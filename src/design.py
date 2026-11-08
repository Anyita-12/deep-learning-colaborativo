"""Preregistro, selección de hiperparámetros, piloto y archivos de entorno."""
from __future__ import annotations

import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .persist import is_done, load_pickle, save_pickle, save_text

REQ_PACKAGES = ["torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
                "matplotlib", "pillow", "scikit-posthocs", "nbformat", "ipykernel", "jupyter-book"]


def _md_table(df: pd.DataFrame) -> str:
    cols = ["trial"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def preregistro_path() -> Path | None:
    found = sorted(C.COMMON.glob("preregistro_*.md"))
    return found[0] if found else None


def write_preregistro(overwrite: bool = False) -> Path:
    """Escribe el preregistro fechado. Si ya existe uno, no se reescribe: la fecha
    debe ser anterior a las corridas definitivas."""
    old = preregistro_path()
    if old and not overwrite:
        return old
    hp = C.DEFAULT_HP
    grid = pd.DataFrame(C.HPO_GRID)
    txt = f"""# Preregistro del experimento de modelos base

Fecha: {time.strftime('%Y-%m-%d %H:%M')}. Se escribe antes de la búsqueda de hiperparámetros y de las corridas definitivas.

## Hipótesis
- H0 global: las tres arquitecturas (CNN estilo VGG reducida, MobileNetV2 y ResNet-18), entrenadas desde cero con el mismo protocolo, tienen la misma distribución de F1 macro en prueba.
- H1 global: al menos una difiere.
- Por pares: H0: la diferencia media de F1 macro pareada por semilla es 0.

## Métrica, umbrales y unidad de réplica
- Métrica primaria: F1 macro en el conjunto de prueba. Secundarias: accuracy, balanced accuracy, F1 ponderado y por clase, precisión y recall macro, AUC-ROC macro (OvR), AUC-PR macro, MCC y ECE.
- alpha = {C.ALPHA}; potencia objetivo = {C.POWER}; delta (relevancia práctica) = {C.DELTA} en F1 macro.
- Unidad de réplica: la semilla (inicialización, orden de lotes, aumentación y dropout). Las tres arquitecturas comparten partición y semillas: datos pareados por semilla.

## Datos
- Partición v2 del EDA (`{C.SPLIT_CSV.name}`): 1800/360/361, estratificada y agrupada por casi-duplicados, sin las 6 imágenes con etiqueta contradictoria. Huella del test verificada antes de cada corrida.
- Entrada: {C.IMG_SIZE}x{C.IMG_SIZE}, redimensionado directo (todas las imágenes son 512x384). Normalización con media y DE por canal de train. Pesos de clase con train.
- Aumentación solo en train: {C.AUGMENTATION}.

## Entrenamiento (igual para los tres)
- Régimen: desde cero (weights=None) para los tres. La CNN no tiene pesos preentrenados; preentrenar solo dos rompería la equidad. La transferencia desde ImageNet queda para el Entregable 2.
- Valores por defecto: {hp}.
- Búsqueda de hiperparámetros: malla de {len(C.HPO_GRID)} configuraciones, idéntica para los tres modelos, semilla {C.HPO_SEED}, máximo {C.HPO_MAX_EPOCHS} épocas, selección por F1 macro de validación. El test no se usa.

{_md_table(grid)}

## Tamaño de muestra (piloto interno)
- Orden: búsqueda de hiperparámetros, luego piloto con las semillas {C.PILOT_SEEDS} y la configuración elegida. Las corridas del piloto son las primeras corridas definitivas (misma configuración).
- R = (z_(1-alpha/2) + z_(1-beta))^2 * sd_d^2 / delta^2 por par de modelos, con sd_d de las diferencias pareadas de F1 macro de validación del piloto. Se toma el máximo y se redondea hacia arriba.
- R_final = min(max(R, {C.R_MIN}), {C.R_MAX}). El mínimo de {C.R_MIN} permite pruebas por rangos; el máximo de {C.R_MAX} lo impone el tiempo de GPU. Si R supera {C.R_MAX}, se reporta la diferencia mínima detectable (MDE) como limitación.

## Análisis planeado
- Global: Friedman (bloques = semillas, tratamientos = 3 modelos) con p exacto por permutación y W de Kendall.
- Pares (3): t pareada y Wilcoxon de rangos con signo; diferencia media con IC 95 % (t y bootstrap); d_z y r rango-biserial; p sin ajustar y ajustados por Holm. Nemenyi y Conover como complemento.
- Por imagen: McNemar por par y semilla (exacto si b + c < 25), con Holm dentro de cada semilla.
- AUC: bootstrap estratificado pareado sobre las imágenes de prueba ({C.N_BOOT} remuestreos).
- Una diferencia es relevante si |diferencia media| >= delta y significativa tras Holm.
- Robustez: ruido gaussiano (0.05 y 0.10), desenfoque, brillo +-30 %, oclusión y fondo sustituido (hallazgo del EDA, 2.7).
"""
    p = C.COMMON / f"preregistro_{time.strftime('%Y-%m-%d')}.md"
    save_text(txt, p)
    return p


def write_requirements_lock(path: Path) -> Path:
    """Versiones exactas del entorno actual (pip freeze filtrado) y un environment.yml."""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    keep = []
    names = {p.lower().replace("_", "-") for p in REQ_PACKAGES}
    for line in out.splitlines():
        name = line.split("==")[0].split(" @ ")[0].strip().lower().replace("_", "-")
        if name in names:
            keep.append(line.strip())
    save_text("\n".join(sorted(keep)) + "\n", path)
    yml = "name: trashnet-dl\nchannels:\n  - defaults\ndependencies:\n" \
          f"  - python={sys.version_info.major}.{sys.version_info.minor}\n  - pip\n  - pip:\n" + \
          "".join(f"      - {k}\n" for k in sorted(keep))
    save_text(yml, path.with_name("environment.yml"))
    return path


# ------------------------------------------------------------------ búsqueda de hiperparámetros
def hpo_table(models=C.MODELS) -> pd.DataFrame:
    rows = []
    for m in models:
        for t in C.HPO_TRIALS_USED:
            d = C.hpo_dir(m, t)
            row = {"model": m, "trial": t, **C.HPO_GRID[t], "done": is_done(d)}
            if is_done(d):
                r = load_pickle(d / "result.pkl")
                row.update({"val_f1_macro": r["best_val_f1_macro"], "best_epoch": r["best_epoch"],
                            "epochs_run": r["epochs_run"], "train_time_s": r["train_time_s"]})
            rows.append(row)
    return pd.DataFrame(rows)


def select_hpo(models=C.MODELS) -> dict:
    tab = hpo_table(models)
    if not tab["done"].all():
        raise RuntimeError("La búsqueda de hiperparámetros no está completa para todos los modelos.")
    sel = {}
    for m, g in tab.groupby("model"):
        best = g.sort_values(["val_f1_macro", "trial"], ascending=[False, True]).iloc[0]
        hp = dict(C.DEFAULT_HP)
        hp.update(C.HPO_GRID[int(best["trial"])])
        sel[m] = {"trial": int(best["trial"]), "val_f1_macro": float(best["val_f1_macro"]), "hp": hp}
    save_pickle({"space": C.HPO_SPACE, "grid": C.HPO_GRID, "trials_used": C.HPO_TRIALS_USED,
                 "seed": C.HPO_SEED, "max_epochs": C.HPO_MAX_EPOCHS, "table": tab}, C.COMMON / "hpo_space.pkl")
    save_pickle(sel, C.COMMON / "hpo_selected.pkl")
    return sel


# ------------------------------------------------------------------ piloto y tamaño de muestra
def pilot_and_power(models=C.MODELS) -> dict:
    from .stats import sample_size
    vals = {}
    for m in models:
        for s in C.PILOT_SEEDS:
            d = C.run_dir(m, s)
            if not is_done(d):
                raise RuntimeError(f"Falta la corrida piloto {m} seed {s}.")
            vals[(m, s)] = load_pickle(d / "timing.pkl")["best_val_f1_macro"]
    df = pd.DataFrame([{"model": m, "seed": s, "val_f1_macro": v} for (m, s), v in vals.items()])
    w = df.pivot(index="seed", columns="model", values="val_f1_macro")[list(models)]
    desc = w.agg(["mean", "std"]).T
    desc["cv"] = desc["std"] / desc["mean"]
    diffs = {f"{a} - {b}": (w[a] - w[b]).to_numpy() for a, b in itertools.combinations(models, 2)}
    pa = sample_size(diffs)
    pilot = {"tabla_larga": df, "tabla_ancha": w, "descriptivos": desc, "semillas": C.PILOT_SEEDS,
             "metrica": "F1 macro de validación (mejor época)"}
    save_pickle(pilot, C.COMMON / "pilot.pkl")
    save_pickle(pa, C.COMMON / "power_analysis.pkl")
    return {"pilot": pilot, "power": pa}
