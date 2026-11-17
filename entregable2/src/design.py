"""Preregistro, configuración de la búsqueda de hiperparámetros y selección por validación."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import config as C
from .models import SPECS, get_spec
from .persist import is_done, load_pickle, save_pickle, save_text

LOCK_PACKAGES = ["torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pillow",
                 "xgboost", "scikit-posthocs", "matplotlib"]


def trial_hp(model: str, trial: int, hpo: bool = True) -> dict:
    s = get_spec(model)
    g = C.HPO_GRID[trial]
    hp = {**C.DEFAULT_HP, "lr": s.base_lr * g["lr_factor"], "weight_decay": g["weight_decay"], "dropout": g["dropout"]}
    if hpo:
        hp["max_epochs"] = C.HPO_MAX_EPOCHS
    return hp


def preregistro_path() -> Path | None:
    found = sorted(C.COMMON.glob("preregistro_*.md"))
    return found[0] if found else None


def write_preregistro(overwrite: bool = False) -> Path:
    old = preregistro_path()
    if old and not overwrite:
        return old
    lines = ["| modelo | familia | preentrenado | lr base | normalización | pérdida | arquitectura definida |",
             "|---|---|---|---|---|---|---|"]
    for m in C.MODELS:
        s = SPECS[m]
        lines.append(f"| {m} | {s.family} | {'sí' if s.pretrained else 'no'} | {s.base_lr:g} | {s.normalization} | "
                     f"{s.loss}{' + XGBoost' if s.post_hoc else ''} | {s.defines_architecture} |")
    grid = pd.DataFrame(C.HPO_GRID)
    grid_md = "\n".join(["| trial | factor lr | weight decay | dropout |", "|---|---|---|---|"] +
                        [f"| {i} | {r.lr_factor} | {r.weight_decay} | {r.dropout} |" for i, r in grid.iterrows()])
    dev = "\n".join(f"- {m}: " + "; ".join(SPECS[m].deviations) for m in C.MODELS)
    txt = f"""# Preregistro del benchmark del Entregable 2

Fecha: {time.strftime('%Y-%m-%d %H:%M')}. Se escribe antes de la búsqueda de hiperparámetros y de las corridas definitivas.

## Elementos heredados del Entregable 1 (sin cambios)
- Métrica primaria: F1 macro en prueba. alpha = {C.ALPHA}, delta = {C.DELTA}, potencia objetivo = {C.POWER}.
- Unidad de réplica: la semilla. Semillas {C.SEEDS} (R = {len(C.SEEDS)}, el R_final del Entregable 1), compartidas por todos los modelos.
- Partición v2 (`{C.SPLIT_CSV.name}`, 1800/360/361), huella del test verificada antes de cada corrida.
- Entrada {C.IMG_SIZE}x{C.IMG_SIZE} por redimensionado directo, aumentación solo en train: {C.AUGMENTATION}.
- Entropía cruzada ponderada por la inversa de la frecuencia de clase (o la pérdida propia del modelo con los mismos pesos).

## Modelos
{chr(10).join(lines)}

Normalización: estadísticos de ImageNet para los modelos con pesos preentrenados (exigido por los pesos) y media y DE de train para los entrenados desde cero.

## Desviaciones respecto a los papers
{dev}

## Entrenamiento (igual para los siete)
- AdamW, calentamiento lineal de {C.DEFAULT_HP['warmup_epochs']} épocas y coseno hasta lr x {C.DEFAULT_HP['eta_min_factor']}, lote efectivo {C.DEFAULT_HP['batch_size']}, recorte de gradiente {C.DEFAULT_HP['grad_clip']}, precisión mixta.
- Máximo {C.DEFAULT_HP['max_epochs']} épocas en las corridas definitivas y {C.HPO_MAX_EPOCHS} en la búsqueda; parada temprana con paciencia {C.DEFAULT_HP['patience']} sobre F1 macro de validación.
- Búsqueda de hiperparámetros: malla de {len(C.HPO_GRID)} configuraciones para cada modelo (mismo número de intentos), semilla {C.HPO_SEED}. La tasa de aprendizaje es lr base x factor, con lr base según el régimen (desde cero 1e-3; ajuste fino de CNN 3e-4; ajuste fino de Transformers 1e-4), de modo que la amplitud relativa del espacio es la misma. Selección por F1 macro de validación; el test no se usa.

{grid_md}

- FocusRCNet: profesor EfficientNet-B4 (ImageNet) entrenado una vez con la partición de train y seleccionado en validación; destilación con T = {C.KD_TEMPERATURE} y alpha = {C.KD_ALPHA}.
- Celik-HC: la parte convolucional se entrena con el protocolo común; después XGBoost se ajusta solo con los embeddings de train. Se reportan ambas salidas (XGBoost y cabeza softmax).

## Análisis planeado
- Global: Friedman (bloques = semillas) sobre los diez modelos (siete nuevos y los tres base) con p por permutación y W de Kendall.
- Pares: t pareada y Wilcoxon, IC 95 % (t y bootstrap), d_z, r rango-biserial, Holm. Relevante si |diferencia| >= delta y p Holm < alpha.
- Por imagen: McNemar por par y semilla con Holm.
- Calibración: ECE, NLL y Brier antes y después del escalado de temperatura ajustado en validación.
- Incertidumbre: MC Dropout ({C.MC_DROPOUT_SAMPLES} pasadas) e IC bootstrap por semilla.
- Robustez: ruido gaussiano (0.05 y 0.10), desenfoque, brillo +-30 %, oclusión y fondo sustituido (hallazgo del EDA).
- Interpretabilidad: Grad-CAM sobre {C.CAM_PER_CLASS} imágenes de prueba por clase y fracción de la atención sobre el fondo.
- Replicación del paper guía (Celik, 2025) bajo su protocolo original y bajo el protocolo estricto.
"""
    p = C.COMMON / f"preregistro_{time.strftime('%Y-%m-%d')}.md"
    save_text(txt, p)
    return p


def hpo_table(models) -> pd.DataFrame:
    rows = []
    for m in models:
        for t in C.HPO_TRIALS_USED:
            d = C.hpo_dir(m, t)
            hp = trial_hp(m, t)
            row = {"model": m, "trial": t, "lr": hp["lr"], "weight_decay": hp["weight_decay"], "dropout": hp["dropout"],
                   "done": is_done(d)}
            if is_done(d):
                r = load_pickle(d / "result.pkl")
                row.update({"val_f1_macro": r["selection_val_f1_macro"], "val_f1_macro_head": r["best_val_f1_macro"],
                            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "train_time_s": r["train_time_s"]})
            rows.append(row)
    return pd.DataFrame(rows)


def select_hpo(models) -> dict:
    tab = hpo_table(models)
    f_sel = C.COMMON / "hpo_selected.pkl"
    sel = load_pickle(f_sel) if f_sel.exists() else {}
    for m, g in tab.groupby("model"):
        if not g["done"].all():
            continue
        best = g.sort_values(["val_f1_macro", "trial"], ascending=[False, True]).iloc[0]
        t = int(best["trial"])
        sel[m] = {"trial": t, "val_f1_macro": float(best["val_f1_macro"]), "hp": trial_hp(m, t, hpo=False)}
    full = load_pickle(C.COMMON / "hpo_space.pkl")["table"] if (C.COMMON / "hpo_space.pkl").exists() else pd.DataFrame()
    if len(full):
        full = full[~full["model"].isin(tab["model"].unique())]
    save_pickle({"space": C.HPO_SPACE, "grid": C.HPO_GRID, "trials_used": C.HPO_TRIALS_USED, "seed": C.HPO_SEED,
                 "max_epochs": C.HPO_MAX_EPOCHS, "table": pd.concat([full, tab], ignore_index=True)},
                C.COMMON / "hpo_space.pkl")
    save_pickle(sel, f_sel)
    return sel


def write_requirements_lock(path: Path) -> Path:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    names = {p.lower().replace("_", "-") for p in LOCK_PACKAGES}
    keep = [l.strip() for l in out.splitlines()
            if l.split("==")[0].split(" @ ")[0].strip().lower().replace("_", "-") in names]
    return save_text("\n".join(sorted(keep)) + "\n", path)
