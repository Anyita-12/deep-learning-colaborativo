"""Baselines no profundos del Entregable 2 (Sección 3.2 y Paso 5 de la Sección 7.2 del enunciado).

- Trivial: predice siempre la clase mayoritaria del entrenamiento.
- Clásico fuerte: gradient boosting (HistGradientBoosting de scikit-learn) sobre descriptores de color
  (histogramas HSV) y de forma (histogramas de orientación del gradiente por celdas, tipo HOG), con el
  mismo presupuesto de ajuste que los modelos profundos: 8 configuraciones, semilla 100, selección por
  F1 macro de validación y 8 semillas finales (0-7) sobre la misma partición v2.
- Prueba de sanidad de fuga (Sección 10.1): el mismo pipeline con las etiquetas de entrenamiento
  permutadas debe dar un F1 de prueba cercano al azar.

Uso: python entregable2/analisis/baselines.py   (requiere las imágenes en data/raw/trashnet/dataset-resized)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from comun import CLASSES, OUT, REPO, SEEDS, split_table

RAW = Path(os.environ.get("TRASHNET_RAW", REPO / "data" / "raw" / "trashnet" / "dataset-resized"))
CACHE = OUT / "_features_clasicas.npz"


def descriptor(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((128, 96), Image.BICUBIC)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    feats = []
    # Color: histogramas HSV (16 bins por canal) y medias y DE por canal RGB
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=16, range=(0, 1))
        feats.append(h / h.sum())
    feats.append(rgb.reshape(-1, 3).mean(0))
    feats.append(rgb.reshape(-1, 3).std(0))
    # Forma: histogramas de orientación del gradiente (9 bins) en una malla de 4 x 4 celdas
    g = rgb.mean(2)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    H, W = g.shape
    for i in range(4):
        for j in range(4):
            sl = (slice(i * H // 4, (i + 1) * H // 4), slice(j * W // 4, (j + 1) * W // 4))
            h, _ = np.histogram(ang[sl], bins=9, range=(0, 180), weights=mag[sl])
            feats.append(h / (h.sum() + 1e-8))
    # Textura global: miniatura en escala de grises 16 x 12
    feats.append(np.asarray(im.convert("L").resize((16, 12), Image.BILINEAR), dtype=np.float32).ravel() / 255.0)
    return np.concatenate(feats).astype(np.float32)


def features() -> tuple[np.ndarray, pd.DataFrame]:
    s = split_table().sort_values("relpath").reset_index(drop=True)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["relpath"]) == list(s["relpath"]):
            return z["X"], s
    X = np.stack([descriptor(RAW / r) for r in s["relpath"]])
    np.savez_compressed(CACHE, X=X, relpath=s["relpath"].to_numpy())
    return X, s


def metrics(y, p) -> dict:
    return {"f1_macro": f1_score(y, p, average="macro", labels=list(range(6)), zero_division=0),
            "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "mcc": matthews_corrcoef(y, p)}


GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.1, 0.05) for ml in (15, 31) for l2 in (0.0, 1.0)]


def model(cfg: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=False, class_weight="balanced",
                                          random_state=seed, **cfg)


def main():
    t0 = time.time()
    X, s = features()
    y = s["label_id"].to_numpy()
    tr, va, te = (s["split"] == "train").to_numpy(), (s["split"] == "val").to_numpy(), (s["split"] == "test").to_numpy()
    print(f"descriptores: {X.shape} ({time.time() - t0:.0f} s)")

    # Trivial
    maj = np.bincount(y[tr], minlength=6).argmax()
    triv = metrics(y[te], np.full(te.sum(), maj))
    triv.update({"model": "Trivial (clase mayoritaria)", "seed": -1, "clase_mayoritaria": CLASSES[maj]})

    # Búsqueda de hiperparámetros: 8 configuraciones, semilla 100, selección en validación
    hpo = []
    for i, cfg in enumerate(GRID):
        clf = model(cfg, 100).fit(X[tr], y[tr])
        hpo.append({"trial": i, **cfg, "val_f1_macro": f1_score(y[va], clf.predict(X[va]), average="macro")})
    hpo = pd.DataFrame(hpo)
    best = GRID[int(hpo.val_f1_macro.idxmax())]
    print("configuración elegida:", best)

    # 8 semillas finales
    rows, preds = [triv], {}
    for sd in SEEDS:
        clf = model(best, sd).fit(X[tr], y[tr])
        p = clf.predict(X[te])
        preds[sd] = p
        r = metrics(y[te], p)
        r.update({"model": "HistGradientBoosting (color + HOG)", "seed": sd})
        rows.append(r)
    res = pd.DataFrame(rows)

    # Prueba de sanidad con etiquetas de entrenamiento permutadas (B = 30 réplicas del pipeline completo)
    rng = np.random.default_rng(2026)
    perm = []
    for b in range(30):
        clf = model(best, b).fit(X[tr], rng.permutation(y[tr]))
        perm.append(metrics(y[te], clf.predict(X[te]))["f1_macro"])
    perm = pd.DataFrame({"replica": range(30), "f1_macro_test": perm})

    hpo.to_csv(OUT / "baseline_clasico_busqueda.csv", index=False)
    res.to_csv(OUT / "baselines.csv", index=False)
    perm.to_csv(OUT / "sanidad_etiquetas_permutadas.csv", index=False)
    np.savez_compressed(OUT / "baseline_clasico_predicciones.npz",
                        relpath=s.loc[te, "relpath"].to_numpy(), y_true=y[te],
                        y_pred=np.stack([preds[sd] for sd in SEEDS]))
    print(res.groupby("model")[["f1_macro", "accuracy"]].agg(["mean", "std"]).round(4))
    print("permutadas: media F1 =", round(perm.f1_macro_test.mean(), 4), "máx =", round(perm.f1_macro_test.max(), 4))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
