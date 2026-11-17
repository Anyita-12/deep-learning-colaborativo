"""Utilidades comunes de los análisis complementarios del Entregable 2.

Estos análisis no entrenan redes profundas: leen las predicciones, historias y mapas guardados por
cada corrida (results/<modelo>/seed_XX) y escriben tablas en entregable2/results/complementario/.
Todos los análisis que no figuran en el preregistro del 14 de noviembre se declaran exploratorios.
"""
from __future__ import annotations

import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
E2 = REPO / "entregable2"
RES = E2 / "results"
E1_RES = REPO / "results"
OUT = RES / "complementario"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = REPO / "data" / "splits" / "trashnet_split_v2_s1234.csv"
# Copia completa de los resultados (con gradcam.npz), que no se versiona por tamaño.
FULL = Path(os.environ.get("TRASHNET_E2_FULL", str(RES)))

if str(E2) not in sys.path:
    sys.path.insert(0, str(E2))

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
E2_MODELS = ["Celik-HC", "MbXception", "ViT-B16", "GFN", "FocusRCNet", "ResMsCapsule", "RWCNet"]
E1_MODELS = ["CNN", "MobileNetV2", "ResNet-18"]
MODELS = E2_MODELS + E1_MODELS
SEEDS = list(range(8))
DELTA = 0.02
ALPHA = 0.05
MACHINE = {"ViT-B16": "A100", "GFN": "A100", "Celik-HC": "A100",
           "MbXception": "RTX 4060", "FocusRCNet": "RTX 4060", "ResMsCapsule": "RTX 4060", "RWCNet": "RTX 4060",
           "CNN": "RTX 4060", "MobileNetV2": "RTX 4060", "ResNet-18": "RTX 4060"}


def load_pickle(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def run_dir(model: str, seed: int) -> Path:
    if model in E1_MODELS:
        return E1_RES / model / f"seed_{seed:02d}"
    if "/" in model:
        return RES / "ablation" / model / f"seed_{seed:02d}"
    return RES / model / f"seed_{seed:02d}"


def relpath(fp: str) -> str:
    parts = Path(str(fp).replace("\\", "/")).parts
    return "/".join(parts[-2:])


@lru_cache(maxsize=1)
def split_table() -> pd.DataFrame:
    return pd.read_csv(SPLIT)


@lru_cache(maxsize=None)
def preds(model: str, seed: int, which: str = "test") -> dict:
    d = load_pickle(run_dir(model, seed) / f"{which}_predictions.pkl")
    out = {k: d[k] for k in ("y_true", "y_pred", "probs") if k in d}
    out["relpath"] = np.array([relpath(f) for f in d["filepaths"]])
    return out


def aligned(model: str, seed: int, which: str = "test") -> dict:
    """Predicciones ordenadas por relpath para poder parear modelos imagen a imagen."""
    d = preds(model, seed, which)
    o = np.argsort(d["relpath"])
    return {k: v[o] for k, v in d.items()}


@lru_cache(maxsize=2)
def groups_for(which: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Relpaths ordenados del conjunto y su grupo de casi-duplicados (unidad de independencia)."""
    s = split_table()
    s = s[s["split"] == ("test" if which == "test" else "val")].sort_values("relpath")
    return s["relpath"].to_numpy(), s["group"].to_numpy()


def group_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int = 2026) -> list[np.ndarray]:
    """Índices de remuestreo por grupo: se remuestrean grupos completos con reemplazo."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def f1m(y, p) -> float:
    return f1_score(y, p, average="macro", labels=list(range(6)), zero_division=0)


def fast_f1_macro(y: np.ndarray, p: np.ndarray, k: int = 6) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den == 0, 1, den), 0.0)
    return float(f1.mean())


def save(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    df.to_csv(p, index=False)
    return p
