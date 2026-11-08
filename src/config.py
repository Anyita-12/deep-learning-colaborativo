"""Configuración central de la implementación (Secciones 4 y 5).

Todo lo que define un experimento vive aquí como diccionarios serializables.
Este módulo se importa ANTES que torch en todos los puntos de entrada, porque
CUBLAS_WORKSPACE_CONFIG debe existir antes de inicializar CUDA (determinismo).
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ------------------------------------------------------------------ rutas
ROOT = Path(__file__).resolve().parents[1]                 # raíz del repositorio
RAW_DIR = ROOT / "data" / "raw" / "trashnet"               # imágenes (ignorado por git)
INTERIM = ROOT / "data" / "interim"                        # cachés (ignorado por git)
SPLIT_CSV = ROOT / "data" / "splits" / "trashnet_split_v2_s1234.csv"   # partición v2 del EDA
SUMS = ROOT / "data" / "SHA256SUMS.txt"                    # huella del test registrada en el EDA

# Los resultados pueden redirigirse (p. ej. para la prueba de humo) con TRASHNET_RESULTS.
RESULTS = Path(os.environ.get("TRASHNET_RESULTS", str(ROOT / "results")))
COMMON = RESULTS / "common"
SUMMARY = RESULTS / "summary"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"

# Copia de respaldo de cada corrida terminada (carpeta sincronizada con OneDrive o
# Google Drive). None la desactiva. Se puede fijar sin tocar el código con la
# variable de entorno TRASHNET_BACKUP.
BACKUP_DIR = os.environ.get("TRASHNET_BACKUP") or None
BACKUP_INCLUDE_CHECKPOINT = False      # checkpoint_last.pt pesa ~3x el modelo; no se respalda

# ------------------------------------------------------------------ problema
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
MODELS = ["CNN", "MobileNetV2", "ResNet-18"]
IMG_SIZE = 224
RESIZE_METHOD = "resize"          # 512x384 (4:3) -> 224x224, misma deformación para todas las imágenes

# ------------------------------------------------------------------ inferencia (declarado antes de entrenar)
PRIMARY_METRIC = "f1_macro"
ALPHA = 0.05
DELTA = 0.02                      # diferencia mínima de relevancia práctica en F1 macro
POWER = 0.80
R_MIN = 6                         # mínimo para pruebas por rangos (Wilcoxon/Friedman)
R_MAX = 10                        # techo por tiempo de GPU; si R calculado lo supera, se declara
PILOT_SEEDS = [0, 1, 2, 3, 4]
HPO_SEED = 100
N_BOOT = 10_000

# ------------------------------------------------------------------ entrenamiento (igual para los tres)
DEFAULT_HP = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "batch_size": 32,
    "max_epochs": 100,
    "warmup_epochs": 3,
    "eta_min_factor": 1e-2,       # el coseno baja hasta lr * 1e-2
    "patience": 15,
    "min_delta": 1e-4,
    "grad_clip": 5.0,
    "amp": True,
    "class_weighted_loss": True,
}

HPO_MAX_EPOCHS = 50
HPO_SPACE = {"lr": [1e-3, 3e-4], "weight_decay": [1e-4, 5e-4], "dropout": [0.2, 0.4]}
HPO_GRID = [dict(zip(HPO_SPACE, v)) for v in itertools.product(*HPO_SPACE.values())]   # 8 configuraciones
# Si falta tiempo, se usa un subconjunto, SIEMPRE el mismo para los tres modelos (p. ej. [0,1,2,3,4,5]).
HPO_TRIALS_USED = list(range(len(HPO_GRID)))
KEEP_HPO_CHECKPOINTS = False      # al terminar un trial se borra su checkpoint (ahorra disco)

# ------------------------------------------------------------------ aumentación (solo train)
AUGMENTATION = {
    "RandomResizedCrop": {"size": IMG_SIZE, "scale": (0.7, 1.0)},
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomVerticalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 15},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
}

# ------------------------------------------------------------------ DataLoader
# En Windows los workers se crean con "spawn": por eso Dataset y transformaciones
# viven en src/ y no en los cuadernos. Si falla, TRASHNET_WORKERS=0.
NUM_WORKERS = int(os.environ.get("TRASHNET_WORKERS", "4"))

# ------------------------------------------------------------------ evaluación
ECE_BINS = 15
TIMING_WARMUP = 20
TIMING_BATCHES = 100
ROBUSTNESS_SEED = 2026

# ------------------------------------------------------------------ figuras
PALETTE = ["#0B132B", "#1C2541", "#3A506B", "#5BC0BE", "#6FFFE9"]
MODEL_COLORS = {"CNN": "#3A506B", "MobileNetV2": "#5BC0BE", "ResNet-18": "#1C2541"}
FIG_DPI = 200


# ------------------------------------------------------------------ prueba de humo
# `python -m src.runner ... --smoke` fija TRASHNET_SMOKE=1: pocas semillas, 2 épocas y results_smoke/.
SMOKE = os.environ.get("TRASHNET_SMOKE") == "1"
if SMOKE:
    PILOT_SEEDS, R_MIN, R_MAX = [0, 1, 2], 4, 4
    HPO_TRIALS_USED = [0, 7]
    HPO_MAX_EPOCHS = 2
    DEFAULT_HP = {**DEFAULT_HP, "max_epochs": 2, "warmup_epochs": 1}
    TIMING_WARMUP, TIMING_BATCHES = 2, 3


def run_dir(model: str, seed: int) -> Path:
    return RESULTS / model / f"seed_{seed:02d}"


def hpo_dir(model: str, trial: int) -> Path:
    return RESULTS / model / "hpo" / f"trial_{trial:02d}"


def ensure_dirs() -> None:
    for d in (COMMON, SUMMARY, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
