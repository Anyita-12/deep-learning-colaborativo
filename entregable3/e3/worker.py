"""Ejecuta UNA tarea en un proceso aislado (el orquestador lo lanza con subprocess).

    python -m e3.worker '<json de la tarea>' <worker_id>

Familias:
  e2  : modelos publicados del E2 con el código congelado de entregable2/src (semillas nuevas 8-11).
  nag : NAGFuse y sus variantes, con el MISMO bucle de entrenamiento del E2 (registro y pérdida añadidos aquí).
  e1  : modelos base del E1 con el código congelado de src/ (raíz del repositorio).
Prep: tareas de preparación (datos, estadísticos de ruido, conjuntos adicionales).

Códigos de salida: 0 terminada, 3 bloqueada por una dependencia, 4 error, 5 memoria de GPU insuficiente.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import cfg
from . import sync as S

EXIT_OK, EXIT_BLOCKED, EXIT_ERROR, EXIT_OOM = 0, 3, 4, 5


class Blocked(Exception):
    pass


def _log_factory(name: str):
    def log(msg: str) -> None:
        S.log_line(msg, name)
    return log


# ================================================================== utilidades comunes
def run_path(t: dict) -> Path:
    """Carpeta de la corrida, sin importar el código del E1/E2 (lo usa también el orquestador)."""
    fam = "e1" if t["fam"] == "e1" else "e2"
    root = cfg.RESULTS[fam]
    if t["kind"] == "hpo":
        return root / t["model"] / "hpo" / f"trial_{t['trial']:02d}"
    if t["fam"] == "nag" and t["model"] not in (cfg.MODEL, cfg.MODEL_B):
        return root / "ablation" / t["model"] / f"seed_{t['seed']:02d}"
    return root / t["model"] / f"seed_{t['seed']:02d}"


def _probs_batches(fn, x_uint8, bs: int = 64):
    """Aplica fn a lotes de imágenes uint8 (N, H, W, 3) convertidas a tensores [0, 1] (N, 3, H, W)."""
    import numpy as np
    import torch
    out = []
    for k in range(0, len(x_uint8), bs):
        x01 = torch.from_numpy(np.ascontiguousarray(x_uint8[k:k + bs])).permute(0, 3, 1, 2).float() / 255
        out.append(fn(x01))
    return np.concatenate(out)


def _extra_metrics(y, probs) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    present = sorted(set(int(v) for v in np.unique(y)))
    pred = probs.argmax(1)
    return {"f1_macro": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_per_class": {cfg.CLASSES[k]: float(v) for k, v in
                             zip(present, f1_score(y, pred, labels=present, average=None, zero_division=0))},
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(cfg.CLASSES)))),
            "clases_presentes": [cfg.CLASSES[k] for k in present], "n": int(len(y)),
            "y_pred": pred.astype(np.int64), "probs": probs.astype(np.float16)}


def extra_eval(run: Path, prob_fn, log, extra_fn=None) -> list[str]:
    """Evalúa la corrida en cada conjunto adicional disponible. Devuelve los conjuntos que faltan."""
    from . import extra_data as X
    f = run / "extra_eval.pkl"
    res = _load_pickle_any(f) if f.exists() else {}
    missing = []
    for name in cfg.EXTRA_DATASETS:
        if name in res:
            continue
        if not (cfg.EXTRA_DIR / f"{name}.npz").exists():
            missing.append(name)
            continue
        d = X.load(name)
        x = d["x"][:48] if cfg.SMOKE else d["x"]
        y = d["y"][:48] if cfg.SMOKE else d["y"]
        res[name] = _extra_metrics(y, _probs_batches(prob_fn, x))
        if extra_fn is not None:
            res[name].update(extra_fn(x))
        _save_pickle_any(res, f)
        log(f"  {run.parent.name}/{run.name} en {name}: F1 macro {res[name]['f1_macro']:.4f} (n = {len(y)})")
    return missing


def _save_pickle_any(obj, path: Path) -> None:
    import pickle
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_pickle_any(path: Path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def keep_weights(t: dict, run: Path, missing_extra: list[str], log) -> None:
    """Conserva en Drive los pesos de las semillas previstas y de las corridas con evaluación pendiente.
    Cuando una corrida pendiente ya se evaluó en todos los conjuntos, borra su copia de pendientes."""
    import shutil
    keep = t.get("seed") in cfg.KEEP_WEIGHTS.get(t["model"], [])
    pend_dir = cfg.DRIVE_WEIGHTS / "pendientes" / t["model"] / run.name
    if not missing_extra and pend_dir.exists():
        shutil.rmtree(pend_dir, ignore_errors=True)
    if not (keep or missing_extra):
        return
    dst = pend_dir if (missing_extra and not keep) else cfg.DRIVE_WEIGHTS / t["model"] / run.name
    if (dst / "best_model.pt").exists() and missing_extra:
        S.write_json_atomic({"tarea": t, "faltan": missing_extra}, dst / "pendiente.json")
        return
    for name in ("best_model.pt", "xgb_model.json", "config.json"):
        if (run / name).exists():
            S.copy_atomic(run / name, dst / name)
    if missing_extra:
        S.write_json_atomic({"tarea": t, "faltan": missing_extra}, dst / "pendiente.json")
    log(f"  pesos de {t['model']} {run.name} conservados en {dst}")


def install_mirror(T, fam: str, tid: str, worker: str, max_epochs: int) -> None:
    """Envuelve save_torch del módulo de entrenamiento para reflejar el checkpoint en Drive y dar latidos."""
    orig = T.save_torch

    def save_torch_mirrored(obj, path, *a, **k):
        r = orig(obj, path, *a, **k)
        p = Path(path)
        if p.name == "checkpoint_last.pt":
            S.mirror_checkpoint(p.parent, fam, p)
            S.heartbeat(tid, worker)
        return r

    T.save_torch = save_torch_mirrored


# ================================================================== familia E2 y NAGFuse (código de entregable2/src)
def _setup_e2_env():
    os.environ["TRASHNET_RESULTS"] = str(cfg.RESULTS["e2"])
    os.environ["TRASHNET_DATA"] = str(cfg.DATA)
    os.environ["TRASHNET_BACKUP"] = str(cfg.DRIVE_RESULTS["e2"])
    os.environ["E3_NOISE_STATS"] = str(cfg.NOISE_STATS)
    if cfg.SMOKE:
        os.environ["TRASHNET_SMOKE"] = "1"
    sys.path.insert(0, str(cfg.E2_ROOT))


def _register_nagfuse():
    """Registra NAGFuse en el registro de modelos y en la pérdida del E2 sin modificar sus archivos."""
    import torch.nn.functional as F
    from src import losses as L
    from src import models as M
    from src import train as T
    from .nagfuse import build_nagfuse

    def spec(name, loss, family):
        return M.ModelSpec(name, family, "Este trabajo (Entregable 3)", "", "nagfuse", base_lr=cfg.BASE_LR_NAG,
                           normalization="imagenet", loss=loss, defines_architecture="propio",
                           deviations=())
    M.SPECS["NAGFuse"] = spec("NAGFuse", "mix", "Fusión ViT-B/16 + EfficientNet-B0 con compuerta adaptativa al ruido")
    M.SPECS["NAGFuse/sin_denoising"] = spec("NAGFuse/sin_denoising", "mix", "Ablación: sin contracción de Haar")
    M.SPECS["NAGFuse/solo_vit"] = spec("NAGFuse/solo_vit", "ce", "Ablación: solo la rama ViT-B/16")
    M.SPECS["NAGFuse/solo_cnn"] = spec("NAGFuse/solo_cnn", "ce", "Ablación: solo la rama EfficientNet-B0 con contracción")
    # Opción B (protocolo v4): mismo mecanismo y misma pérdida; solo cambian las redes de cada rama
    M.SPECS["NAGFuse-B1"] = spec("NAGFuse-B1", "mix", "Opción B, candidato 1: Swin-S + ConvNeXt-T")
    M.SPECS["NAGFuse-B2"] = spec("NAGFuse-B2", "mix", "Opción B, candidato 2: Swin-S + EfficientNet-B0")
    M.SPECS["NAGFuse-B3"] = spec("NAGFuse-B3", "mix", "Opción B, candidato 3: ViT-B/16 + ConvNeXt-T")
    M.SPECS["NAGFuse-B"] = spec("NAGFuse-B", "mix", f"Opción B confirmatoria ({cfg.ELEGIDO_B})")

    orig_build = T.build_model

    def build(name, dropout=0.3, num_classes=6):
        if name.startswith("NAGFuse"):
            return build_nagfuse(name, dropout=dropout, num_classes=num_classes, pretrained=M.PRETRAINED,
                                 rho=cfg.RHO)
        return orig_build(name, dropout=dropout, num_classes=num_classes)
    T.build_model = build

    class E3Criterion(L.Criterion):
        def __init__(self, kind, weight=None, teacher=None, **kw):
            super().__init__("ce" if kind == "mix" else kind, weight, teacher, **kw)
            self.kind = kind

        def forward(self, out, x, y):
            if self.kind == "mix":
                logp, zv, zc = out
                loss = self.ce(logp, y) + cfg.LAMBDA_BRANCH * (self.ce(zv, y) + self.ce(zc, y))
                return loss, logp
            return super().forward(out, x, y)

        def eval_loss(self, logits, y):
            if self.kind == "mix":
                return F.cross_entropy(logits.float(), y, weight=self.weight)
            return super().eval_loss(logits, y)
    T.Criterion = E3Criterion
    return build


def _pull_drive_file(rel: str) -> None:
    src = cfg.DRIVE_RESULTS["e2"] / rel
    dst = cfg.RESULTS["e2"] / rel
    if src.exists():
        S.copy_atomic(src, dst)


def _nag_analysis(model, common, mean, std, dev, run: Path, log) -> None:
    """Compuerta apagada, sensibilidad a rho, sigma_hat y salidas de cada rama (exploratorio y ablación)."""
    import numpy as np
    import torch
    from src import analysis as A
    from src.evaluate import compute_metrics
    f = run / "nagfuse_analisis.pkl"
    if f.exists() or not model.fused:
        return
    cache = np.load(common["cache"], mmap_mode="r")
    rows, y = common["splits"]["test"]["rows"], common["splits"]["test"]["y"]
    if cfg.SMOKE:
        rows, y = rows[:24], y[:24]
    out = {"s_lo": float(model.s_lo), "rho": float(model.rho), "pi0": float(torch.sigmoid(model.pi_logit)),
           "robustez": {}, "detalles": {}}
    rho0 = float(model.rho)
    configs = [("off", rho0)] + [("on", r) for r in cfg.RHO_SENSITIVITY]
    for mode, rho in configs:
        model.gate_mode = mode
        model.rho.fill_(rho)
        key = f"compuerta_{mode}" if mode == "off" else f"rho_{rho:g}"
        out["robustez"][key] = {}
        for kind in A.PERTURBATIONS:
            x01 = A.perturbed_test_set(cache, rows, kind)      # mismas perturbaciones y semillas del E2
            p = A.probs_from_images(model, x01, mean, std, dev)
            if not np.isfinite(p).all():
                raise ValueError(f"probabilidades no finitas en {key}/{kind}")
            out["robustez"][key][kind] = {"f1_macro": compute_metrics(y, p)["f1_macro"],
                                          "y_pred": p.argmax(1).astype(np.int64)}
    model.gate_mode = "on"
    model.rho.fill_(rho0)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    for kind in A.PERTURBATIONS:
        x01 = A.perturbed_test_set(cache, rows, kind)
        acc = {"sigma": [], "pi": [], "p_v": [], "p_c": []}
        for k in range(0, len(x01), 64):
            d = model.details(((x01[k:k + 64] - m) / s).to(dev))
            for key in acc:
                acc[key].append(d[key].float().cpu().numpy())
        out["detalles"][kind] = {k: np.concatenate(v).astype(np.float32 if k in ("sigma", "pi") else np.float16)
                                 for k, v in acc.items()}
    _save_pickle_any(out, f)
    log(f"  análisis NAGFuse: pi0 = {out['pi0']:.3f}, F1 con ruido 0.05: compuerta on "
        f"{_f1_kind(run, 'ruido_0.05'):.4f} / off {out['robustez']['compuerta_off']['ruido_0.05']['f1_macro']:.4f}")


def _f1_kind(run: Path, kind: str) -> float:
    f = run / "robustness.pkl"
    if not f.exists():
        return float("nan")
    return float(_load_pickle_any(f)["results"][kind]["f1_macro"])


def task_e2(t: dict, worker: str) -> None:
    _setup_e2_env()
    import torch
    from src import config as C
    from src import data as D
    from src import design as DS
    from src import train as T
    from src.persist import is_done, load_pickle, load_torch, save_pickle
    build = _register_nagfuse()
    log = _log_factory(f"worker{worker}")
    tid = t["id"]
    subset = {"train": 48, "val": 24, "test": 24} if cfg.SMOKE else None
    common = D.prepare_common(log=log)
    dev = T.device()
    name = t["model"]
    run = run_path(t)
    if t["fam"] == "nag" and not cfg.NOISE_STATS.exists():
        _pull_drive_file(f"common/{cfg.NOISE_STATS.name}")
        if not cfg.NOISE_STATS.exists():
            raise Blocked("falta nagfuse_noise_stats.json: ejecuta antes la etapa prepare")

    if t["kind"] == "hpo":
        hp = DS.trial_hp(name, t["trial"])
        install_mirror(T, "e2", tid, worker, hp["max_epochs"])
        r = T.train_run(name, cfg.HPO_SEED, hp, run, common, stage="hpo", subset=subset, log=log,
                        extra_config={"trial": t["trial"], "entregable": 3})
        if r != "done" and not is_done(run):
            raise RuntimeError(f"train_run devolvió {r}")
        S.sync_run_files(run, "e2")
        S.drop_mirror(run, "e2")
        # selección cuando estén los 8 intentos (pueden venir de otra sesión vía Drive)
        trials = [0] if cfg.SMOKE else cfg.HPO_TRIALS
        for tr in trials:
            d = run.parent / f"trial_{tr:02d}"
            if not is_done(d):
                for fn in ("DONE", "result.pkl", "config.pkl", "config.json", "history.pkl", "val_predictions.pkl"):
                    _pull_drive_file(str((d / fn).relative_to(cfg.RESULTS["e2"])))
        if all(is_done(run.parent / f"trial_{tr:02d}") for tr in trials):
            _pull_drive_file("common/hpo_selected.pkl")
            _pull_drive_file("common/hpo_space.pkl")
            if cfg.SMOKE:
                C.HPO_TRIALS_USED = trials
            sel = DS.select_hpo([name])
            log(f"HPO {name}: trial {sel[name]['trial']} (F1 val {sel[name]['val_f1_macro']:.4f}) -> lr "
                f"{sel[name]['hp']['lr']:g}, wd {sel[name]['hp']['weight_decay']:g}, dropout {sel[name]['hp']['dropout']}")
            for fn in ("hpo_selected.pkl", "hpo_space.pkl"):
                S.copy_atomic(C.COMMON / fn, cfg.DRIVE_RESULTS["e2"] / "common" / fn)
        return

    # ---------------- corrida final
    base = cfg.MODEL if t["fam"] == "nag" else name
    f_sel = C.COMMON / "hpo_selected.pkl"
    sel = load_pickle(f_sel) if f_sel.exists() else {}
    if base not in sel:
        _pull_drive_file("common/hpo_selected.pkl")
        sel = load_pickle(f_sel) if f_sel.exists() else {}
    if base not in sel:
        raise Blocked(f"falta la búsqueda de hiperparámetros de {base}")
    hp = sel[base]["hp"]
    if cfg.SMOKE:
        hp = {**hp, "max_epochs": 2, "warmup_epochs": 1}
    if name == "FocusRCNet" and not is_done(run):
        _ensure_teacher(T, C, common, subset, log, tid, worker)
    install_mirror(T, "e2", tid, worker, hp["max_epochs"])
    extra = {"hpo_trial": sel[base]["trial"], "entregable": 3}
    if t["fam"] == "nag" and name not in (cfg.MODEL, cfg.MODEL_B):
        extra["ablation_of"] = cfg.MODEL
    if name == cfg.MODEL_B:
        extra.update({"opcion": "B", "protocolo": "v4", "candidato": cfg.ELEGIDO_B,
                      "hp_de": "búsqueda de la versión A (intento %d)" % sel[base]["trial"]})
    if not is_done(run):
        ck = S.restore_checkpoint(run, "e2")
        if ck:
            log(f"  checkpoint de {tid} restaurado desde Drive")
    r = T.train_run(name, t["seed"], hp, run, common, stage="final", subset=subset, log=log, extra_config=extra)
    if r != "done" and not is_done(run):
        raise RuntimeError(f"train_run devolvió {r}")

    # ---------------- evaluación ampliada con los pesos de la corrida
    spec = T.get_spec(name)
    mean, std = D.normalization(common, spec.normalization)
    w = run / "best_model.pt"
    if not w.exists():
        mirror = S.mirror_dir(run, "e2") / "checkpoint_last.pt"
        kept = cfg.DRIVE_WEIGHTS / name / run.name / "best_model.pt"
        pend = cfg.DRIVE_WEIGHTS / "pendientes" / name / run.name / "best_model.pt"
        if kept.exists() or pend.exists():
            S.copy_atomic(kept if kept.exists() else pend, w)
        elif mirror.exists():
            torch.save(load_torch(mirror)["best_model"], w)
        else:
            log(f"  AVISO: {tid} terminó en otra sesión sin pesos disponibles; se omite la evaluación ampliada.")
            S.write_json_atomic({"motivo": "pesos no disponibles"}, run / "E3_HOOK.json")
            S.sync_run_files(run, "e2")
            return
    model = build(name, dropout=hp["dropout"]).to(dev)
    model.load_state_dict(load_torch(w))
    model.eval()
    xgb = T.load_xgb(run) if spec.post_hoc == "xgb" else None
    from src.analysis import probs_from_images

    def prob_fn(x01):
        return probs_from_images(model, x01, mean, std, dev, xgb)

    extra_fn = None
    if getattr(model, "fused", False):
        import numpy as np
        m_t, s_t = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)

        def extra_fn(x):
            def sig(x01):
                return model.details(((x01 - m_t) / s_t).to(dev))["pi"].float().cpu().numpy()[:, None]
            pis = _probs_batches(sig, x)[:, 0]
            return {"pi": pis.astype(np.float32)}
        try:
            _nag_analysis(model, common, mean, std, dev, run, log)
        except Exception as e:      # el análisis exploratorio no debe tumbar una corrida ya entrenada y evaluada
            import traceback
            log(f"  AVISO: el análisis exploratorio de NAGFuse falló ({e!r}); la corrida continúa sin nagfuse_analisis.pkl")
            S.write_json_atomic({"error": repr(e), "traza": traceback.format_exc()}, run / "nagfuse_analisis_ERROR.json")
    missing = extra_eval(run, prob_fn, log, extra_fn)
    keep_weights(t, run, missing, log)
    S.write_json_atomic({"tarea": t, "faltan_extra": missing, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                        run / "E3_HOOK.json")
    S.sync_run_files(run, "e2")
    S.drop_mirror(run, "e2")


def _ensure_teacher(T, C, common, subset, log, tid, worker) -> None:
    """El profesor de FocusRCNet (EfficientNet-B4) no está versionado. Si no se aportan sus pesos del E2 en
    Drive (manual/teacher_effnetb4_best_model.pt), se reentrena con la misma configuración y semilla y se
    declara como desviación."""
    import shutil
    from src.persist import is_done, load_pickle
    tdir = T.teacher_path().parent
    w = tdir / "best_model.pt"
    if w.exists():
        return
    manual = cfg.DRIVE_BASE / "manual" / "teacher_effnetb4_best_model.pt"
    if manual.exists() and is_done(tdir):
        S.copy_atomic(manual, w)
        log("Profesor de FocusRCNet: pesos del E2 restaurados desde Drive/manual.")
        return
    if is_done(tdir):
        old = C.COMMON / "teacher_effnetb4_registro_e2"
        if not old.exists():
            shutil.move(str(tdir), str(old))
        log("Profesor de FocusRCNet sin pesos: se reentrena con TEACHER_HP y la semilla 100 (desviación declarada).")
    install_mirror(T, "e2", tid, worker, C.TEACHER_HP["max_epochs"])
    T.train_run("EffNetB4-Teacher", C.HPO_SEED, dict(C.TEACHER_HP), tdir, common, stage="teacher", subset=subset,
                log=log, extra_config={"entregable": 3, "motivo": "pesos del E2 no versionados"})
    S.sync_run_files(tdir, "e2", include_weights=True)
    S.drop_mirror(tdir, "e2")
    r_old = C.COMMON / "teacher_effnetb4_registro_e2" / "result.pkl"
    if r_old.exists() and (tdir / "result.pkl").exists():
        log(f"Profesor: F1 val E2 {load_pickle(r_old)['selection_val_f1_macro']:.4f} / E3 "
            f"{load_pickle(tdir / 'result.pkl')['selection_val_f1_macro']:.4f}")


# ================================================================== familia E1 (código de src/ en la raíz)
def task_e1(t: dict, worker: str) -> None:
    os.environ["TRASHNET_RESULTS"] = str(cfg.RESULTS["e1"])
    os.environ["TRASHNET_BACKUP"] = str(cfg.DRIVE_RESULTS["e1"])
    if cfg.SMOKE:
        os.environ["TRASHNET_SMOKE"] = "1"
    sys.path.insert(0, str(cfg.REPO))
    import torch
    from src import analysis as A1
    from src import config as C
    from src import data as D1
    from src import train as T1
    from src.persist import is_done, load_pickle
    log = _log_factory(f"worker{worker}")
    tid = t["id"]
    subset = {"train": 48, "val": 24, "test": 24} if cfg.SMOKE else None
    common = D1.prepare_common()
    sel = load_pickle(C.COMMON / "hpo_selected.pkl")
    name, seed = t["model"], t["seed"]
    hp = sel[name]["hp"]
    if cfg.SMOKE:
        hp = {**hp, "max_epochs": 2, "warmup_epochs": 1}
    run = C.run_dir(name, seed)
    install_mirror(T1, "e1", tid, worker, hp["max_epochs"])
    if not is_done(run) and S.restore_checkpoint(run, "e1"):
        log(f"  checkpoint de {tid} restaurado desde Drive")
    r = T1.train_run(name, seed, hp, run, common, stage="final", eval_test=True, subset=subset, log=log,
                     extra_config={"hpo_trial": sel[name]["trial"], "entregable": 3})
    if r != "done" and not is_done(run):
        raise RuntimeError(f"train_run devolvió {r}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = D1.load_cache()
    A1.robustness_run(name, run, common, cache, dev)
    model = A1.load_best(name, run, dev)
    mean, std = common["norm"]["mean"], common["norm"]["std"]

    def prob_fn(x01):
        return torch.softmax(torch.from_numpy(A1._logits(model, x01, mean, std, dev)).float(), 1).numpy()

    missing = extra_eval(run, prob_fn, log)
    keep_weights(t, run, missing, log)
    S.write_json_atomic({"tarea": t, "faltan_extra": missing, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                        run / "E3_HOOK.json")
    S.sync_run_files(run, "e1", include_weights=True)
    S.drop_mirror(run, "e1")


# ================================================================== preparación
def prep_e2(worker: str) -> None:
    """Descarga TrashNet, verifica la partición, construye la caché, calcula s_lo y prepara los conjuntos adicionales."""
    _setup_e2_env()
    import numpy as np
    import torch
    from src import data as D
    from src import design as DS
    from .nagfuse import mad_sigma
    log = _log_factory(f"worker{worker}")
    common = D.prepare_common(log=log)
    cache = np.load(common["cache"], mmap_mode="r")
    if not cfg.NOISE_STATS.exists():
        rows = np.sort(common["splits"]["train"]["rows"])
        sig = []
        for k in range(0, len(rows), 128):
            x01 = torch.from_numpy(np.array(cache[rows[k:k + 128]])).permute(0, 3, 1, 2).float() / 255
            sig.append(mad_sigma(x01).numpy())
        sig = np.concatenate(sig)
        st = {"s_lo": float(np.quantile(sig, cfg.NOISE_QUANTILE)), "cuantil": cfg.NOISE_QUANTILE,
              "calculado_en": "train (1800 imágenes limpias, sin aumentación)", "n": int(len(sig)),
              "mediana": float(np.median(sig)), "p95": float(np.quantile(sig, 0.95)), "max": float(sig.max()),
              "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        S.write_json_atomic(st, cfg.NOISE_STATS)
        log(f"s_lo = {st['s_lo']:.5f} (cuantil {cfg.NOISE_QUANTILE} de sigma_hat en train; mediana {st['mediana']:.5f})")
    S.copy_atomic(cfg.NOISE_STATS, cfg.DRIVE_RESULTS["e2"] / "common" / cfg.NOISE_STATS.name)
    lock = cfg.RESULTS["e2"] / "common" / "requirements-lock-e3.txt"
    if not lock.exists():
        DS.write_requirements_lock(lock)
        S.copy_atomic(lock, cfg.DRIVE_RESULTS["e2"] / "common" / lock.name)
    from . import extra_data as X
    all_cache = np.load(common["cache"], mmap_mode="r")
    for name in cfg.EXTRA_DATASETS:
        try:
            X.build(name, all_cache, log)
        except Exception as e:
            log(f"AVISO: no se pudo preparar {name}: {e!r}. El entrenamiento sigue; las corridas guardan sus "
                f"pesos para evaluarlo después (orquestador extra-pass).")


def prep_e1(worker: str) -> None:
    os.environ["TRASHNET_RESULTS"] = str(cfg.RESULTS["e1"])
    sys.path.insert(0, str(cfg.REPO))
    from src import data as D1
    D1.prepare_common()
    _log_factory(f"worker{worker}")("Caché y partición del E1 verificadas.")


def main() -> int:
    t = json.loads(sys.argv[1])
    worker = sys.argv[2] if len(sys.argv) > 2 else "0"
    log = _log_factory(f"worker{worker}")
    try:
        if t["kind"] == "prep":
            (prep_e1 if t["fam"] == "e1" else prep_e2)(worker)
        elif t["fam"] == "e1":
            task_e1(t, worker)
        else:
            task_e2(t, worker)
        return EXIT_OK
    except Blocked as e:
        log(f"Bloqueada {t.get('id')}: {e}")
        return EXIT_BLOCKED
    except Exception as e:
        tb = traceback.format_exc()
        try:
            run = run_path(t) if t["kind"] != "prep" else cfg.WORK / "logs"
            run.mkdir(parents=True, exist_ok=True)
            (run / f"ERROR_E3_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(tb, encoding="utf-8")
            cfg.DRIVE_LOGS.mkdir(parents=True, exist_ok=True)
            (cfg.DRIVE_LOGS / f"ERROR_{t.get('id', 'prep')}_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(
                tb, encoding="utf-8")
        except OSError:
            pass
        log(f"ERROR en {t.get('id')}: {e!r}\n{tb}")
        if "OutOfMemoryError" in type(e).__name__ or "out of memory" in str(e).lower():
            return EXIT_OOM
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
