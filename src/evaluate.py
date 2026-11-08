"""Predicción, métricas, calibración, tiempos de inferencia y entorno."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from . import config as C


# ------------------------------------------------------------------ predicción
@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> dict:
    model.eval()
    logits, ys, idx = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, pos in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        logits.append(out.float().cpu())
        ys.append(y)
        idx.append(pos)
    lg = torch.cat(logits)
    probs = torch.softmax(lg, dim=1).numpy().astype(np.float64)
    y_true = torch.cat(ys).numpy().astype(np.int64)
    return {"y_true": y_true, "y_pred": probs.argmax(1).astype(np.int64), "probs": probs,
            "logits": lg.numpy().astype(np.float32), "idx": torch.cat(idx).numpy().astype(np.int64)}


# ------------------------------------------------------------------ métricas
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
    per = f1_score(y, pred, labels=range(k), average=None, zero_division=0)
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
        "reliability": rel,
        "f1_per_class": {c: float(v) for c, v in zip(classes, per)},
        "precision_per_class": dict(zip(classes, precision_score(y, pred, labels=range(k), average=None, zero_division=0).astype(float).tolist())),
        "recall_per_class": dict(zip(classes, recall_score(y, pred, labels=range(k), average=None, zero_division=0).astype(float).tolist())),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(k)),
        "n": int(len(y)),
    }


# ------------------------------------------------------------------ tiempos
@torch.no_grad()
def time_inference(model, device, batch_size: int = 32, warmup: int = C.TIMING_WARMUP,
                   n_batches: int = C.TIMING_BATCHES, amp: bool = True) -> dict:
    """Tiempo por imagen (ms) en lotes de batch_size tras calentamiento, y latencia con lote 1."""
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

    tb = run(batch_size, n_batches, warmup)
    t1 = run(1, n_batches, warmup)
    per_img = tb / batch_size * 1000
    return {"batch_size": batch_size, "n_batches": n_batches, "warmup": warmup,
            "ms_per_image_mean": float(per_img.mean()), "ms_per_image_sd": float(per_img.std(ddof=1)),
            "latency_bs1_ms_mean": float(t1.mean() * 1000), "latency_bs1_ms_sd": float(t1.std(ddof=1) * 1000),
            "throughput_img_s": float(batch_size / tb.mean()), "device": str(device), "amp": use_amp}


# ------------------------------------------------------------------ entorno
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


def env_info(determinism: dict | None = None) -> dict:
    import torchvision
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_available": cuda, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None, "driver": _driver() if cuda else None,
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
        "os": platform.platform(), "num_workers": C.NUM_WORKERS, "determinism": determinism or {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------------------------------------------------------ lectura de resultados
def collect_runs(models=C.MODELS, results=None, seeds=None) -> pd.DataFrame:
    """Una fila por corrida terminada (modelo, semilla) con las métricas de prueba."""
    from .persist import is_done, load_pickle
    results = results or C.RESULTS
    rows = []
    for m in models:
        mdir = results / m
        if not mdir.exists():
            continue
        for d in sorted(mdir.glob("seed_*")):
            if not is_done(d):
                continue
            s = int(d.name.split("_")[1])
            if seeds is not None and s not in seeds:
                continue
            met = load_pickle(d / "metrics_test.pkl")
            tim = load_pickle(d / "timing.pkl")
            inf = load_pickle(d / "model_info.pkl")
            row = {"model": m, "seed": s}
            row.update({k: v for k, v in met.items() if isinstance(v, (int, float))})
            row.update({"val_f1_macro": tim["best_val_f1_macro"], "best_epoch": tim["best_epoch"],
                        "epochs_run": tim["epochs_run"], "train_time_s": tim["train_time_s"],
                        "ms_per_image": tim["inference"]["ms_per_image_mean"],
                        "params": inf["params_total"], "gmacs": inf["gmacs"], "size_mb": inf["size_mb"]})
            rows.append(row)
    return pd.DataFrame(rows)
