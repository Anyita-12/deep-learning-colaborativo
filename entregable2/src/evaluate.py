"""Predicción, métricas, calibración, incertidumbre, tiempos de inferencia y entorno."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from . import config as C


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def predict(model, loader, device, amp: bool = True, xgb=None) -> dict:
    """Logits, probabilidades y etiquetas. Con xgb, las probabilidades finales salen del clasificador
    XGBoost sobre el embedding de 64 dimensiones y se guardan también las de la cabeza softmax."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    logits, embs, ys, idx = [], [], [], []
    for x, y, pos in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x, return_embedding=True) if xgb is not None or getattr(model, "embedding_dim", None) else model(x)
        if isinstance(out, tuple):
            out, emb = out
            embs.append(emb.float().cpu())
        logits.append(out.float().cpu())
        ys.append(y)
        idx.append(pos)
    lg = torch.cat(logits).numpy().astype(np.float32)
    res = {"y_true": torch.cat(ys).numpy().astype(np.int64), "logits": lg,
           "idx": torch.cat(idx).numpy().astype(np.int64)}
    head = _softmax(lg.astype(np.float64))
    if embs:
        res["embedding"] = torch.cat(embs).numpy().astype(np.float32)
    if xgb is not None:
        res["probs_head"] = head
        res["probs"] = xgb.predict_proba(res["embedding"]).astype(np.float64)
    else:
        res["probs"] = head
    res["y_pred"] = res["probs"].argmax(1).astype(np.int64)
    return res


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = C.ECE_BINS) -> tuple[float, dict]:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    acc, cnf, cnt = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    ece = 0.0
    for k in range(n_bins):
        m = b == k
        cnt[k] = m.sum()
        if cnt[k]:
            acc[k], cnf[k] = correct[m].mean(), conf[m].mean()
            ece += cnt[k] / len(y) * abs(acc[k] - cnf[k])
    return float(ece), {"edges": edges, "acc": acc, "conf": cnf, "count": cnt}


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def compute_metrics(y: np.ndarray, probs: np.ndarray, classes=C.CLASSES) -> dict:
    k = len(classes)
    pred = probs.argmax(1)
    onehot = np.eye(k)[y]
    ece, rel = ece_score(probs, y)
    p = np.clip(probs, 1e-12, 1)
    return {
        "f1_macro": float(f1_score(y, pred, labels=range(k), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, labels=range(k), average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y, pred, labels=range(k), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=range(k), average="macro", zero_division=0)),
        "auc_roc_macro": _safe(roc_auc_score, y, probs, multi_class="ovr", average="macro", labels=list(range(k))),
        "auc_pr_macro": _safe(average_precision_score, onehot, probs, average="macro"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "ece": ece,
        "nll": _safe(log_loss, y, p / p.sum(1, keepdims=True), labels=list(range(k))),
        "brier": float(np.mean([brier_score_loss(onehot[:, j], probs[:, j]) for j in range(k)])),
        "reliability": rel,
        "f1_per_class": {c: float(v) for c, v in zip(classes, f1_score(y, pred, labels=range(k), average=None, zero_division=0))},
        "precision_per_class": dict(zip(classes, precision_score(y, pred, labels=range(k), average=None, zero_division=0).astype(float).tolist())),
        "recall_per_class": dict(zip(classes, recall_score(y, pred, labels=range(k), average=None, zero_division=0).astype(float).tolist())),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(k)),
        "n": int(len(y)),
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Escalado de temperatura (Guo et al., 2017) ajustado en validación por máxima verosimilitud."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calibration_summary(val: dict, test: dict) -> dict:
    """ECE, NLL y Brier en prueba antes y después del escalado de temperatura ajustado en validación."""
    if "probs_head" in test:
        return {"aplicable": False, "motivo": "las probabilidades finales provienen de XGBoost"}
    t = fit_temperature(val["logits"], val["y_true"])
    scaled = _softmax(test["logits"].astype(np.float64) / t)
    before, after = compute_metrics(test["y_true"], test["probs"]), compute_metrics(test["y_true"], scaled)
    keys = ("ece", "nll", "brier", "f1_macro", "accuracy")
    return {"aplicable": True, "temperature": t, "antes": {k: before[k] for k in keys},
            "despues": {k: after[k] for k in keys}, "reliability_despues": after["reliability"], "probs_escaladas": scaled}


def enable_mc_dropout(model) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) and m.p > 0:
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_dropout(model, loader, device, n_samples: int = C.MC_DROPOUT_SAMPLES, amp: bool = True, xgb=None) -> dict:
    """Incertidumbre por MC Dropout (Gal y Ghahramani, 2016): media predictiva, entropía total,
    entropía aleatoria esperada e información mutua (incertidumbre epistémica)."""
    model.eval()
    n_drop = enable_mc_dropout(model)
    if n_drop == 0:
        model.eval()
        return {"aplicable": False, "motivo": "el modelo no tiene capas de dropout activas"}
    use_amp = amp and device.type == "cuda"
    runs = []
    for s in range(n_samples):
        torch.manual_seed(C.ROBUSTNESS_SEED + s)
        probs, ys = [], []
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x, return_embedding=True) if xgb is not None else model(x)
            if xgb is not None:
                probs.append(xgb.predict_proba(out[1].float().cpu().numpy()))
            else:
                probs.append(torch.softmax(out.float(), 1).cpu().numpy())
            ys.append(y.numpy())
        runs.append(np.concatenate(probs))
    model.eval()
    P = np.stack(runs).astype(np.float64)
    mean = P.mean(0)
    eps = 1e-12
    total = -(mean * np.log(mean + eps)).sum(1)
    aleatoric = -(P * np.log(P + eps)).sum(2).mean(0)
    y = np.concatenate(ys)
    met = compute_metrics(y, mean)
    return {"aplicable": True, "n_samples": n_samples, "n_dropout_layers": n_drop, "probs_mean": mean,
            "probs_std": P.std(0), "entropy_total": total, "entropy_aleatoric": aleatoric,
            "mutual_information": total - aleatoric, "y_true": y,
            "metrics": {k: met[k] for k in ("f1_macro", "accuracy", "ece", "nll", "brier")}}


@torch.no_grad()
def time_inference(model, device, batch_size: int = 32, warmup: int = C.TIMING_WARMUP,
                   n_batches: int = C.TIMING_BATCHES, amp: bool = True) -> dict:
    model.eval()
    use_amp = amp and device.type == "cuda"
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def run(bs, n, w):
        x = torch.randn(bs, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
        for _ in range(w):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)
            sync()
            ts.append(time.perf_counter() - t0)
        return np.array(ts)

    tb, t1 = run(batch_size, n_batches, warmup), run(1, n_batches, warmup)
    per_img = tb / batch_size * 1000
    sd = (lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0)
    return {"batch_size": batch_size, "n_batches": n_batches, "warmup": warmup,
            "ms_per_image_mean": float(per_img.mean()), "ms_per_image_sd": sd(per_img),
            "latency_bs1_ms_mean": float(t1.mean() * 1000), "latency_bs1_ms_sd": sd(t1 * 1000),
            "throughput_img_s": float(batch_size / tb.mean()), "device": str(device), "amp": use_amp}


def _ram_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = MS()
            s.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.ullTotalPhys / 2**30, 1)
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 1)
    except Exception:
        return None


def _driver() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _platform_name() -> str:
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.exists("/content"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def env_info(determinism: dict | None = None) -> dict:
    import torchvision
    cuda = torch.cuda.is_available()
    try:
        import xgboost
        xgb_v = xgboost.__version__
    except ImportError:
        xgb_v = None
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
        "xgboost": xgb_v, "cuda_available": cuda, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
        "driver": _driver() if cuda else None, "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(), "os": platform.platform(), "platform": _platform_name(),
        "num_workers": C.NUM_WORKERS, "accum_steps": C.ACCUM_STEPS, "determinism": determinism or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect_runs(models, results=None, stage_glob: str = "seed_*") -> pd.DataFrame:
    """Una fila por corrida final terminada con las métricas de prueba, tiempos y tamaño del modelo."""
    from .persist import is_done, load_pickle
    results = results or C.RESULTS
    rows = []
    for m in models:
        mdir = results / m
        for d in sorted(mdir.glob(stage_glob)) if mdir.exists() else []:
            if not is_done(d):
                continue
            met = load_pickle(d / "metrics_test.pkl")
            tim = load_pickle(d / "timing.pkl")
            inf = load_pickle(d / "model_info.pkl")
            row = {"model": m, "seed": int(d.name.split("_")[1])}
            row.update({k: v for k, v in met.items() if isinstance(v, (int, float))})
            row.update({"val_f1_macro": tim["best_val_f1_macro"], "best_epoch": tim["best_epoch"],
                        "epochs_run": tim["epochs_run"], "train_time_s": tim["train_time_s"],
                        "s_per_epoch": tim.get("s_per_epoch_mean"),
                        "ms_per_image": tim["inference"]["ms_per_image_mean"],
                        "params": inf["params_total"], "gmacs": inf.get("gmacs"), "size_mb": inf["size_mb"]})
            rows.append(row)
    return pd.DataFrame(rows)
