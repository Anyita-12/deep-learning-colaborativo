"""Entrenamiento reanudable de una corrida (modelo, semilla).

Protección ante cortes:
- Al terminar CADA época se guarda checkpoint_last.pt de forma atómica, con
  modelo, optimizador, planificador, GradScaler, historia, contador de early
  stopping, mejor métrica, tiempo acumulado y estados aleatorios. La versión
  anterior queda como checkpoint_last_prev.pt.
- Al relanzar, una corrida con DONE se salta; una con checkpoint se reanuda
  desde la última época completa. Solo se pierde la época en curso.
- La aumentación y el orden de los lotes dependen solo de (semilla, época), así
  que la historia reanudada coincide con la de una corrida sin cortes.
- El test se evalúa una sola vez: si test_predictions.pkl ya existe, se reutiliza.
- DONE se escribe al final, después de todos los demás archivos.
"""
from __future__ import annotations

import copy
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config as C
from .data import CachedImages, EpochSampler, eval_transform, make_loader, train_transform
from .evaluate import compute_metrics, env_info, predict, time_inference
from .models import build_model, model_info
from .persist import (acquire_lock, backup_run, is_done, release_lock, load_pickle, load_torch, save_json, save_pickle,
                      save_torch, touch_done)
from .seeds import configure_determinism, get_rng_state, set_rng_state, set_seed
from sklearn.metrics import f1_score, accuracy_score


def lr_lambda_factory(warmup: int, max_epochs: int, eta_min_factor: float):
    """Calentamiento lineal en `warmup` épocas y coseno hasta lr * eta_min_factor."""
    def f(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        span = max(1, max_epochs - warmup)
        t = min(1.0, (epoch - warmup) / span)
        return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_resume(run: Path):
    for name in ("checkpoint_last.pt", "checkpoint_last_prev.pt"):
        p = run / name
        if p.exists():
            try:
                return load_torch(p), name
            except Exception as e:                 # archivo dañado: probar el anterior
                print(f"  [aviso] {name} no se pudo leer ({e}); se intenta el anterior.")
    return None, None


def train_run(model_name: str, seed: int, hp: dict, run: Path, common: dict, stage: str = "final",
              eval_test: bool = True, subset: dict | None = None, log=print, extra_config: dict | None = None) -> str:
    """Entrena (o reanuda) una corrida. Devuelve 'skip', 'done'."""
    run = Path(run)
    if is_done(run):
        return "skip"
    run.mkdir(parents=True, exist_ok=True)
    if not acquire_lock(run):
        log(f"  {model_name} {run.name}: otro proceso la está entrenando; se omite.")
        return "locked"
    try:
        return _train_run(model_name, seed, hp, run, common, stage, eval_test, subset, log, extra_config)
    finally:
        release_lock(run)


def _train_run(model_name, seed, hp, run, common, stage, eval_test, subset, log, extra_config):
    device = _device()
    det = configure_determinism()

    cfg = {"model": model_name, "seed": int(seed), "stage": stage, "hp": dict(hp), "img_size": C.IMG_SIZE,
           "resize": C.RESIZE_METHOD, "augmentation": C.AUGMENTATION, "split_csv": C.SPLIT_CSV.name,
           "test_fingerprint": common["splits"]["test_fingerprint"], "subset": subset or None,
           "regime": "desde cero (weights=None)", "code_version": _code_version()}
    if extra_config:
        cfg.update(extra_config)
    f_cfg = run / "config.pkl"
    if f_cfg.exists():                            # no mezclar configuraciones en una misma carpeta
        old = load_pickle(f_cfg)
        for k in ("model", "seed", "hp", "img_size", "test_fingerprint", "subset"):
            if old.get(k) != cfg.get(k):
                raise RuntimeError(f"{run}: la configuración guardada difiere en '{k}'. "
                                   f"Borra la carpeta o restaura la configuración original.")
        cfg["date_start"] = old.get("date_start")
    else:
        cfg["date_start"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_pickle(cfg, f_cfg)
        save_json(cfg, run / "config.json")
        save_pickle(env_info(det), run / "env.pkl")

    # ---------------- datos
    sp, norm = common["splits"], common["norm"]
    sub = subset or {}
    from .data import subset_rows
    tr_rows, tr_y = subset_rows(sp["train"], sub.get("train"), seed=0)
    va_rows, va_y = subset_rows(sp["val"], sub.get("val"), seed=0)
    te_rows, te_y = subset_rows(sp["test"], sub.get("test"), seed=0)
    cache = common["cache"]
    ds_tr = CachedImages(cache, tr_rows, tr_y, train_transform(norm["mean"], norm["std"]))
    ds_va = CachedImages(cache, va_rows, va_y, eval_transform(norm["mean"], norm["std"]))
    sampler = EpochSampler(len(ds_tr), seed)
    bs = hp["batch_size"]
    dl_tr = make_loader(ds_tr, bs, sampler=sampler, seed=seed)
    dl_va = make_loader(ds_va, 2 * bs, num_workers=0, seed=seed)     # evaluación sin transformaciones aleatorias
    try:
        return _train_loop(model_name, seed, hp, run, common, stage, eval_test, subset, log, device,
                           sp, norm, cache, tr_rows, va_rows, te_rows, te_y, sampler, dl_tr, dl_va, bs)
    finally:
        _close_loader(dl_tr)                     # libera los procesos workers aunque haya error


def _close_loader(dl) -> None:
    it = getattr(dl, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()
    dl._iterator = None


def _train_loop(model_name, seed, hp, run, common, stage, eval_test, subset, log, device,
                sp, norm, cache, tr_rows, va_rows, te_rows, te_y, sampler, dl_tr, dl_va, bs):

    # ---------------- modelo y optimización
    set_seed(seed)
    model = build_model(model_name, dropout=hp["dropout"]).to(device)
    if not (run / "model_info.pkl").exists():
        save_pickle(model_info(model, model_name), run / "model_info.pkl")
    w = torch.tensor(common["class_weights"]["weights"], dtype=torch.float32, device=device) \
        if hp.get("class_weighted_loss", True) else None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(hp["warmup_epochs"], hp["max_epochs"], hp["eta_min_factor"]))
    use_amp = bool(hp.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1_macro",
                            "val_f1_macro", "lr", "grad_norm_mean", "grad_norm_max", "epoch_time_s", "gpu_mem_max_MB")}
    start_epoch, best_f1, best_epoch, bad, train_time = 0, -1.0, -1, 0, 0.0
    stopped_early = False

    ck, ck_name = _load_resume(run)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        hist, start_epoch = ck["history"], ck["epoch"] + 1
        best_f1, best_epoch, bad = ck["best_f1"], ck["best_epoch"], ck["bad_epochs"]
        train_time, stopped_early = ck["train_time_s"], ck.get("stopped_early", False)
        set_rng_state(ck["rng"])
        log(f"  reanudando {model_name} seed {seed} desde la época {start_epoch} ({ck_name})")
    best_state = ck.get("best_model") if ck is not None else None
    ckpt_last = run / "checkpoint_last.pt"
    ckpt_best = run / "best_model.pt"

    # ---------------- bucle
    for epoch in range(start_epoch, hp["max_epochs"]):
        if stopped_early:
            break
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sampler.set_epoch(epoch)
        model.train()
        tot, n, gnorms, preds, ys = 0.0, 0, [], [], []
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Pérdida no finita en la época {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
            gnorms.append(float(gn))
            preds.append(out.argmax(1).cpu())
            ys.append(y.cpu())
        p_tr, y_tr = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()

        pv = predict(model, dl_va, device, amp=use_amp)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(pv["logits"]), torch.from_numpy(pv["y_true"]),
                                                     weight=w.cpu() if w is not None else None))
        val_f1 = float(f1_score(pv["y_true"], pv["y_pred"], average="macro", labels=range(C.NUM_CLASSES), zero_division=0))
        dt = time.perf_counter() - t0
        train_time += dt
        gfin = [g for g in gnorms if math.isfinite(g)]
        for k, v in (("epoch", epoch), ("train_loss", tot / n), ("val_loss", val_loss),
                     ("train_acc", float(accuracy_score(y_tr, p_tr))), ("val_acc", float(accuracy_score(pv["y_true"], pv["y_pred"]))),
                     ("train_f1_macro", float(f1_score(y_tr, p_tr, average="macro", labels=range(C.NUM_CLASSES), zero_division=0))),
                     ("val_f1_macro", val_f1), ("lr", cur_lr),
                     ("grad_norm_mean", float(np.mean(gfin)) if gfin else float("nan")),
                     ("grad_norm_max", float(np.max(gfin)) if gfin else float("nan")),
                     ("epoch_time_s", dt),
                     ("gpu_mem_max_MB", torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0)):
            hist[k].append(v)

        improved = val_f1 > best_f1 + hp["min_delta"]
        if improved:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(best_state, ckpt_best)
        else:
            bad += 1
        stopped_early = bad >= hp["patience"]

        save_pickle(hist, run / "history.pkl", verify=False)
        save_torch({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": hist, "best_f1": best_f1,
                    "best_epoch": best_epoch, "bad_epochs": bad, "train_time_s": train_time,
                    "stopped_early": stopped_early, "rng": get_rng_state(), "best_model": best_state},
                   ckpt_last, keep_previous=True)
        log(f"  {model_name:<11} s{seed:02d} ep {epoch:3d} | loss {tot / n:.3f}/{val_loss:.3f} | "
            f"F1 val {val_f1:.4f} (mejor {best_f1:.4f} @ {best_epoch}) | {dt:5.1f}s{' *' if improved else ''}")

    # ---------------- cierre: predicciones con el mejor modelo
    if not ckpt_best.exists() and best_state is not None:      # best_model.pt perdido: se restaura del checkpoint
        save_torch(best_state, ckpt_best)
    model.load_state_dict(load_torch(ckpt_best))
    pv = predict(model, dl_va, device, amp=use_amp)
    pv["filepaths"] = [sp["val"]["relpath"][i] for i in _positions(sp["val"]["rows"], va_rows, pv["idx"])]
    pv["best_epoch"] = best_epoch
    save_pickle(pv, run / "val_predictions.pkl")
    result = {"best_val_f1_macro": best_f1, "best_epoch": best_epoch, "epochs_run": len(hist["epoch"]),
              "stopped_early": stopped_early, "train_time_s": train_time}

    if eval_test:
        f_test = run / "test_predictions.pkl"
        if f_test.exists():                       # el test ya se evaluó antes de un corte: no se repite
            pt = load_pickle(f_test)
        else:
            ds_te = CachedImages(cache, te_rows, te_y, eval_transform(norm["mean"], norm["std"]))
            pt = predict(model, make_loader(ds_te, 2 * bs, num_workers=0, seed=seed), device, amp=use_amp)
            pt["filepaths"] = [sp["test"]["relpath"][i] for i in _positions(sp["test"]["rows"], te_rows, pt["idx"])]
            pt["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_pickle(pt, f_test)
        save_pickle(compute_metrics(pt["y_true"], pt["probs"]), run / "metrics_test.pkl")
        quick = subset is not None
        tim = time_inference(model, device, batch_size=bs, warmup=2 if quick else C.TIMING_WARMUP,
                             n_batches=3 if quick else C.TIMING_BATCHES, amp=use_amp)
        save_pickle({**result, "inference": tim, "s_per_epoch_mean": float(np.mean(hist["epoch_time_s"]))},
                    run / "timing.pkl")
    else:                                         # búsqueda de hiperparámetros: el test no se toca
        save_pickle({**result, "hp": dict(hp), "val_metrics": compute_metrics(pv["y_true"], pv["probs"])},
                    run / "result.pkl")
    touch_done(run)
    if stage == "hpo" and not C.KEEP_HPO_CHECKPOINTS:
        for name in ("checkpoint_last.pt", "checkpoint_last_prev.pt"):
            (run / name).unlink(missing_ok=True)
    backup_run(run, C.BACKUP_DIR, C.RESULTS, C.BACKUP_INCLUDE_CHECKPOINT)
    return "done"


def _positions(split_rows: np.ndarray, used_rows: np.ndarray, idx: np.ndarray) -> list[int]:
    """Convierte posiciones dentro del subconjunto usado en posiciones dentro de la partición."""
    where = {r: i for i, r in enumerate(split_rows)}
    return [where[r] for r in used_rows[idx]]


def _code_version() -> str:
    """Hash corto del código de src/ (para saber con qué versión se entrenó cada corrida)."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name in ("runner.py",):
            continue
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
