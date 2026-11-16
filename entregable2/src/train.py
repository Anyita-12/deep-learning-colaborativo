"""Entrenamiento reanudable de una corrida (modelo, semilla) y generación de todas sus salidas.

Cada época termina con un checkpoint atómico (modelo, optimizador, planificador, GradScaler,
historia, early stopping y estados aleatorios). Una corrida con DONE se salta; una con
checkpoint se reanuda desde la última época completa. El conjunto de prueba se evalúa una
sola vez, al final, con el mejor modelo según F1 macro de validación.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from .analysis import gradcam_eval, robustness_eval
from .data import CachedImages, EpochSampler, eval_transform, make_loader, normalization, subset_rows, train_transform
from .evaluate import calibration_summary, compute_metrics, env_info, mc_dropout, predict, time_inference
from .losses import Criterion
from .models import build_model, get_spec, model_info
from .persist import (acquire_lock, backup_run, is_done, load_pickle, load_torch, release_lock, save_json, save_npz,
                      save_pickle, save_torch, touch_done)
from .seeds import configure_determinism, get_rng_state, set_rng_state, set_seed


def lr_lambda_factory(warmup: int, max_epochs: int, eta_min_factor: float):
    def f(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = min(1.0, (epoch - warmup) / max(1, max_epochs - warmup))
        return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def code_version() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob("*.py")):
        if p.name != "runner.py":
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def teacher_path() -> Path:
    return C.COMMON / "teacher_effnetb4" / "best_model.pt"


def load_teacher(dev) -> torch.nn.Module:
    p = teacher_path()
    if not p.exists():
        raise FileNotFoundError("Falta el profesor de FocusRCNet: ejecuta antes la etapa 'teacher'.")
    cfg = load_pickle(p.parent / "config.pkl")
    t = build_model("EffNetB4-Teacher", dropout=cfg["hp"]["dropout"])
    t.load_state_dict(load_torch(p))
    t.to(dev).eval()
    for q in t.parameters():
        q.requires_grad_(False)
    return t


def _load_resume(run: Path, log):
    for name in ("checkpoint_last.pt", "checkpoint_last_prev.pt"):
        p = run / name
        if p.exists():
            try:
                return load_torch(p), name
            except Exception as e:
                log(f"  [aviso] {name} no se pudo leer ({e}); se intenta el anterior.")
    return None, None


def _positions(split_rows: np.ndarray, used_rows: np.ndarray, idx: np.ndarray) -> list[int]:
    where = {r: i for i, r in enumerate(split_rows)}
    return [where[r] for r in used_rows[idx]]


def _fit_xgb(model, loaders: dict, dev, use_amp: bool, run: Path, seed: int):
    from xgboost import XGBClassifier
    feats = {k: predict(model, dl, dev, amp=use_amp) for k, dl in loaders.items()}
    xgb = XGBClassifier(random_state=C.REPLICATION["xgb_random_state"] + seed, n_jobs=-1, tree_method="hist")
    xgb.fit(feats["train"]["embedding"], feats["train"]["y_true"])
    xgb.save_model(str(run / "xgb_model.json"))
    save_npz(run / "embeddings.npz", **{f"{k}_X": v["embedding"] for k, v in feats.items()},
             **{f"{k}_y": v["y_true"] for k, v in feats.items()})
    return xgb


def load_xgb(run: Path):
    from xgboost import XGBClassifier
    p = run / "xgb_model.json"
    if not p.exists():
        return None
    xgb = XGBClassifier()
    xgb.load_model(str(p))
    return xgb


def train_run(model_name: str, seed: int, hp: dict, run: Path, common: dict, stage: str = "final",
              subset: dict | None = None, log=print, extra_config: dict | None = None) -> str:
    run = Path(run)
    if is_done(run):
        return "skip"
    run.mkdir(parents=True, exist_ok=True)
    if not acquire_lock(run):
        log(f"  {model_name} {run.name}: otro proceso la está entrenando; se omite.")
        return "locked"
    try:
        return _train_run(model_name, seed, hp, run, common, stage, subset, log, extra_config or {})
    finally:
        release_lock(run)


def _train_run(model_name, seed, hp, run, common, stage, subset, log, extra):
    dev = device()
    det = configure_determinism()
    spec = get_spec(model_name)
    mean, std = normalization(common, spec.normalization)
    cfg = {"model": model_name, "seed": int(seed), "stage": stage, "hp": dict(hp), "img_size": C.IMG_SIZE,
           "resize": C.RESIZE_METHOD, "augmentation": C.AUGMENTATION, "split_csv": C.SPLIT_CSV.name,
           "test_fingerprint": common["splits"]["test_fingerprint"], "subset": subset or None,
           "pretrained": spec.pretrained, "normalization": {"kind": spec.normalization, "mean": list(mean), "std": list(std)},
           "loss": spec.loss, "post_hoc": spec.post_hoc, "accum_steps": C.ACCUM_STEPS, "code_version": code_version(),
           **extra}
    if spec.loss == "kd":
        cfg["kd"] = {"temperature": C.KD_TEMPERATURE, "alpha": C.KD_ALPHA, "teacher": "EffNetB4-Teacher"}
    f_cfg = run / "config.pkl"
    if f_cfg.exists():
        old = load_pickle(f_cfg)
        for k in ("model", "seed", "hp", "img_size", "test_fingerprint", "subset"):
            if old.get(k) != cfg.get(k):
                raise RuntimeError(f"{run}: la configuración guardada difiere en '{k}'. Borra la carpeta o restaura la original.")
        cfg["date_start"] = old.get("date_start")
    else:
        cfg["date_start"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_pickle(cfg, f_cfg)
        save_json(cfg, run / "config.json")
        save_pickle(env_info(det), run / "env.pkl")

    sp, sub = common["splits"], subset or {}
    tr_rows, tr_y = subset_rows(sp["train"], sub.get("train"))
    va_rows, va_y = subset_rows(sp["val"], sub.get("val"))
    te_rows, te_y = subset_rows(sp["test"], sub.get("test"))
    cache = common["cache"]
    bs = hp["batch_size"]
    micro = max(1, bs // C.ACCUM_STEPS)
    ds_tr = CachedImages(cache, tr_rows, tr_y, train_transform(mean, std))
    sampler = EpochSampler(len(ds_tr), seed)
    dl_tr = make_loader(ds_tr, micro, sampler=sampler, seed=seed)
    dl_va = make_loader(CachedImages(cache, va_rows, va_y, eval_transform(mean, std)), 2 * micro, num_workers=0, seed=seed)
    try:
        return _loop(model_name, spec, seed, hp, run, common, stage, subset, log, dev, mean, std, cache,
                     tr_rows, tr_y, va_rows, te_rows, te_y, sampler, dl_tr, dl_va, micro)
    finally:
        it = getattr(dl_tr, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
        dl_tr._iterator = None


def _loop(model_name, spec, seed, hp, run, common, stage, subset, log, dev, mean, std, cache,
          tr_rows, tr_y, va_rows, te_rows, te_y, sampler, dl_tr, dl_va, micro):
    set_seed(seed)
    model = build_model(model_name, dropout=hp["dropout"]).to(dev)
    if not (run / "model_info.pkl").exists():
        save_pickle(model_info(model, model_name), run / "model_info.pkl")
    w = torch.tensor(common["class_weights"]["weights"], dtype=torch.float32, device=dev) \
        if hp.get("class_weighted_loss", True) else None
    teacher = load_teacher(dev) if spec.loss == "kd" else None
    crit = Criterion(spec.loss, w, teacher)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(hp["warmup_epochs"], hp["max_epochs"],
                                                                     hp["eta_min_factor"]))
    use_amp = bool(hp.get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = C.ACCUM_STEPS

    keys = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1_macro", "val_f1_macro", "lr",
            "grad_norm_mean", "grad_norm_max", "epoch_time_s", "gpu_mem_max_MB")
    hist = {k: [] for k in keys}
    start, best_f1, best_epoch, bad, train_time, stopped = 0, -1.0, -1, 0, 0.0, False
    ck, ck_name = _load_resume(run, log)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        hist, start = ck["history"], ck["epoch"] + 1
        best_f1, best_epoch, bad = ck["best_f1"], ck["best_epoch"], ck["bad_epochs"]
        train_time, stopped = ck["train_time_s"], ck.get("stopped_early", False)
        set_rng_state(ck["rng"])
        log(f"  reanudando {model_name} {run.name} desde la época {start} ({ck_name})")
    best_state = ck.get("best_model") if ck is not None else None
    ck_last, ck_best = run / "checkpoint_last.pt", run / "best_model.pt"

    for epoch in range(start, hp["max_epochs"]):
        if stopped:
            break
        t0 = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sampler.set_epoch(epoch)
        model.train()
        tot, n, gnorms, preds, ys = 0.0, 0, [], [], []
        opt.zero_grad(set_to_none=True)
        n_batches = len(dl_tr)
        for i, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss, out = crit(model(x), x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Pérdida no finita en la época {epoch}")
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == n_batches:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gnorms.append(float(gn))
            tot += loss.item() * len(y)
            n += len(y)
            preds.append(out.detach().argmax(1).cpu())
            ys.append(y.cpu())
        p_tr, y_tr = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()

        pv = predict(model, dl_va, dev, amp=use_amp)
        val_loss = float(crit.eval_loss(torch.from_numpy(pv["logits"]).to(dev), torch.from_numpy(pv["y_true"]).to(dev)))
        val_f1 = float(f1_score(pv["y_true"], pv["y_pred"], average="macro", labels=range(C.NUM_CLASSES), zero_division=0))
        dt = time.perf_counter() - t0
        train_time += dt
        gfin = [g for g in gnorms if math.isfinite(g)]
        vals = (epoch, tot / n, val_loss, float(accuracy_score(y_tr, p_tr)), float(accuracy_score(pv["y_true"], pv["y_pred"])),
                float(f1_score(y_tr, p_tr, average="macro", labels=range(C.NUM_CLASSES), zero_division=0)), val_f1, cur_lr,
                float(np.mean(gfin)) if gfin else float("nan"), float(np.max(gfin)) if gfin else float("nan"), dt,
                torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0)
        for k, v in zip(keys, vals):
            hist[k].append(v)

        improved = val_f1 > best_f1 + hp["min_delta"]
        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(best_state, ck_best)
        else:
            bad += 1
        stopped = bad >= hp["patience"]
        save_pickle(hist, run / "history.pkl")
        save_torch({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": hist, "best_f1": best_f1,
                    "best_epoch": best_epoch, "bad_epochs": bad, "train_time_s": train_time, "stopped_early": stopped,
                    "rng": get_rng_state(), "best_model": best_state}, ck_last, keep_previous=True)
        log(f"  {model_name:<13} {run.name} ep {epoch:3d} | loss {tot / n:.3f}/{val_loss:.3f} | F1 val {val_f1:.4f} "
            f"(mejor {best_f1:.4f} @ {best_epoch}) | {dt:5.1f}s{' *' if improved else ''}")

    if not ck_best.exists() and best_state is not None:
        save_torch(best_state, ck_best)
    model.load_state_dict(load_torch(ck_best))
    sp = common["splits"]
    xgb = None
    if spec.post_hoc == "xgb":
        dl_tr_eval = make_loader(CachedImages(cache, tr_rows, tr_y, eval_transform(mean, std)), 2 * micro,
                                 num_workers=0, seed=seed)
        xgb = load_xgb(run) or _fit_xgb(model, {"train": dl_tr_eval, "val": dl_va}, dev, use_amp, run, seed)
    pv = predict(model, dl_va, dev, amp=use_amp, xgb=xgb)
    pv["filepaths"] = [sp["val"]["relpath"][i] for i in _positions(sp["val"]["rows"], va_rows, pv["idx"])]
    pv["best_epoch"] = best_epoch
    save_pickle(pv, run / "val_predictions.pkl")
    val_metrics = compute_metrics(pv["y_true"], pv["probs"])
    result = {"best_val_f1_macro": best_f1, "best_epoch": best_epoch, "epochs_run": len(hist["epoch"]),
              "stopped_early": stopped, "train_time_s": train_time,
              "s_per_epoch_mean": float(np.mean(hist["epoch_time_s"])) if hist["epoch_time_s"] else float("nan"),
              "selection_val_f1_macro": val_metrics["f1_macro"]}

    if stage in ("hpo", "teacher"):
        save_pickle({**result, "hp": dict(hp), "val_metrics": val_metrics}, run / "result.pkl")
    else:
        _finalize(model, spec, run, common, cache, mean, std, dev, use_amp, xgb, te_rows, te_y, micro, pv,
                  result, hist, subset, log)
    touch_done(run)
    if stage == "hpo" and not C.KEEP_HPO_CHECKPOINTS:
        for name in ("checkpoint_last.pt", "checkpoint_last_prev.pt", "best_model.pt"):
            (run / name).unlink(missing_ok=True)
    backup_run(run, C.BACKUP_DIR, C.RESULTS, C.BACKUP_WEIGHTS)
    return "done"


def _finalize(model, spec, run, common, cache, mean, std, dev, use_amp, xgb, te_rows, te_y, micro, pv, result, hist,
              subset, log):
    sp = common["splits"]
    quick = subset is not None
    f_test = run / "test_predictions.pkl"
    dl_te = make_loader(CachedImages(cache, te_rows, te_y, eval_transform(mean, std)), 2 * micro, num_workers=0, seed=0)
    if f_test.exists():
        pt = load_pickle(f_test)
    else:
        pt = predict(model, dl_te, dev, amp=use_amp, xgb=xgb)
        pt["filepaths"] = [sp["test"]["relpath"][i] for i in _positions(sp["test"]["rows"], te_rows, pt["idx"])]
        pt["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_pickle(pt, f_test)
    save_pickle(compute_metrics(pt["y_true"], pt["probs"]), run / "metrics_test.pkl")
    if "probs_head" in pt:
        save_pickle(compute_metrics(pt["y_true"], pt["probs_head"]), run / "metrics_test_head.pkl")
    save_pickle(calibration_summary(pv, pt), run / "calibration.pkl")
    save_pickle(mc_dropout(model, dl_te, dev, amp=use_amp, xgb=xgb), run / "uncertainty_mc_dropout.pkl")
    if not quick:
        save_pickle(robustness_eval(model, common, cache_array(cache), mean, std, dev, xgb), run / "robustness.pkl")
        save_npz(run / "gradcam.npz", **gradcam_eval(model, common, cache_array(cache), mean, std, dev, pt["y_pred"]))
    tim = time_inference(model, dev, batch_size=micro, warmup=2 if quick else C.TIMING_WARMUP,
                         n_batches=3 if quick else C.TIMING_BATCHES, amp=use_amp)
    save_pickle({**result, "inference": tim}, run / "timing.pkl")
    log(f"  {spec.name} {run.name}: F1 macro prueba {compute_metrics(pt['y_true'], pt['probs'])['f1_macro']:.4f}")


def cache_array(cache_file) -> np.ndarray:
    return np.load(cache_file, mmap_mode="r")


def regenerate_gradcam(model_name: str, run: Path, common: dict, retrain_root: Path, subset: dict | None = None,
                       log=print) -> dict:
    """Recalcula gradcam.npz de una corrida final. Si sus pesos no se conservaron, reentrena la misma
    configuración y semilla en retrain_root y registra cuánto coinciden sus predicciones de prueba con las
    originales; las métricas de la corrida no se modifican."""
    import shutil
    run = Path(run)
    cfg = load_pickle(run / "config.pkl")
    hp, seed = cfg["hp"], int(cfg["seed"])
    spec = get_spec(model_name)
    dev = device()
    mean, std = normalization(common, spec.normalization)
    weights, rdir = run / "best_model.pt", None
    origin = {"modelo": model_name, "semilla": seed, "reentrenado": not weights.exists(), "code_version": code_version()}
    if origin["reentrenado"]:
        rdir = Path(retrain_root) / model_name / run.name
        train_run(model_name, seed, hp, rdir, common, stage="teacher", subset=subset, log=log)
        weights = rdir / "best_model.pt"
        origin["mejor_epoca_original"] = int(load_pickle(run / "timing.pkl")["best_epoch"])
        origin["mejor_epoca_reentrenado"] = int(load_pickle(rdir / "result.pkl")["best_epoch"])
    model = build_model(model_name, dropout=hp["dropout"]).to(dev)
    model.load_state_dict(load_torch(weights))
    model.eval()
    pt = load_pickle(run / "test_predictions.pkl")
    if origin["reentrenado"]:
        te_rows, te_y = subset_rows(common["splits"]["test"], (subset or {}).get("test"))
        dl = make_loader(CachedImages(common["cache"], te_rows, te_y, eval_transform(mean, std)), 64, num_workers=0, seed=0)
        p = predict(model, dl, dev, amp=bool(hp.get("amp", True)) and dev.type == "cuda")
        ref = pt["probs_head"].argmax(1) if "probs_head" in pt else pt["y_pred"]
        ref_by_idx = dict(zip(pt["idx"].tolist(), ref.tolist()))
        same = np.array([ref_by_idx[i] == y for i, y in zip(p["idx"].tolist(), p["y_pred"].tolist())])
        origin["concordancia_predicciones_prueba"] = float(same.mean())
        origin["f1_macro_prueba_reentrenado"] = float(f1_score(p["y_true"], p["y_pred"], average="macro"))
        origin["f1_macro_prueba_original"] = float(load_pickle(run / "metrics_test.pkl")["f1_macro"])
    out = gradcam_eval(model, common, cache_array(common["cache"]), mean, std, dev, pt["y_pred"])
    origin["capa"] = str(out["layer"])
    origin["max_mapa"] = float(np.asarray(out["cam_pred"], dtype=np.float32).max())
    save_npz(run / "gradcam.npz", **out)
    save_json(origin, run / "gradcam_origen.json")
    backup_run(run, C.BACKUP_DIR, C.RESULTS, C.BACKUP_WEIGHTS)
    if rdir is not None:
        shutil.rmtree(rdir, ignore_errors=True)
        for parent in (rdir.parent, rdir.parent.parent):
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
    return origin
