"""Configuración del experimento del Entregable 2.

Las decisiones heredadas del Entregable 1 (partición v2, métrica primaria, delta,
semillas, aumentación y pérdida ponderada) se mantienen sin cambios. Este módulo
se importa antes que torch porque CUBLAS_WORKSPACE_CONFIG debe existir antes de
inicializar CUDA.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "splits" / "trashnet_split_v2_s1234.csv"
SUMS = ROOT / "data" / "SHA256SUMS.txt"
DATA_DIR = Path(os.environ.get("TRASHNET_DATA", str(ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DATASET_ZIP = RAW_DIR / "dataset-resized.zip"

RESULTS = Path(os.environ.get("TRASHNET_RESULTS", str(ROOT / "results")))
COMMON = RESULTS / "common"
SUMMARY = RESULTS / "summary"
LOGS = RESULTS / "logs"
BACKUP_DIR = os.environ.get("TRASHNET_BACKUP") or None
BACKUP_WEIGHTS = os.environ.get("TRASHNET_BACKUP_WEIGHTS", "0") == "1"
E1_RESULTS = Path(os.environ.get("TRASHNET_E1_RESULTS", str(ROOT / "data" / "e1_results")))

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
RESIZE_METHOD = "resize"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODELS = ["Celik-HC", "MbXception", "ViT-B16", "GFN", "FocusRCNet", "ResMsCapsule", "RWCNet"]
BASE_MODELS = ["CNN", "MobileNetV2", "ResNet-18"]
GUIDE_MODEL = "Celik-HC"

PRIMARY_METRIC = "f1_macro"
ALPHA = 0.05
DELTA = 0.02
POWER = 0.80
SEEDS = list(range(8))
HPO_SEED = 100
N_BOOT = 10_000

DEFAULT_HP = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "batch_size": 32,
    "max_epochs": 60,
    "warmup_epochs": 3,
    "eta_min_factor": 1e-2,
    "patience": 12,
    "min_delta": 1e-4,
    "grad_clip": 5.0,
    "amp": True,
    "class_weighted_loss": True,
}

HPO_MAX_EPOCHS = 30
HPO_SPACE = {"lr_factor": [1.0, 0.3], "weight_decay": [1e-4, 5e-4], "dropout": [0.2, 0.4]}
HPO_GRID = [dict(zip(HPO_SPACE, v)) for v in itertools.product(*HPO_SPACE.values())]
HPO_TRIALS_USED = list(range(len(HPO_GRID)))
KEEP_HPO_CHECKPOINTS = False

ACCUM_STEPS = int(os.environ.get("TRASHNET_ACCUM", "1"))
NUM_WORKERS = int(os.environ.get("TRASHNET_WORKERS", str(min(4, os.cpu_count() or 1))))

AUGMENTATION = {
    "RandomResizedCrop": {"size": IMG_SIZE, "scale": (0.7, 1.0)},
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomVerticalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 15},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
}

KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7
TEACHER_HP = {**DEFAULT_HP, "lr": 3e-4, "weight_decay": 1e-4, "dropout": 0.3}

ECE_BINS = 15
TIMING_WARMUP = 20
TIMING_BATCHES = 100
ROBUSTNESS_SEED = 2026
MC_DROPOUT_SAMPLES = 20
CAM_PER_CLASS = 10
CAM_SIZE = 112

REPLICATION = {
    "test_size": 0.20,
    "random_state": 32,
    "epochs": 40,
    "batch_size": 128,
    "lr": 1e-3,
    "xgb_random_state": 101,
    "seeds": [0, 1, 2, 3, 4],
    "inner_val_fraction": 0.10,
    "reported": {"accuracy": 0.9940, "precision": 0.9949, "recall": 0.9954, "f1": 0.9951},
}

SMOKE = os.environ.get("TRASHNET_SMOKE") == "1"
if SMOKE:
    SEEDS = [0, 1]
    HPO_TRIALS_USED = [0, 7]
    HPO_MAX_EPOCHS = 1
    DEFAULT_HP = {**DEFAULT_HP, "max_epochs": 2, "warmup_epochs": 1}
    TEACHER_HP = {**TEACHER_HP, "max_epochs": 1, "warmup_epochs": 1}
    TIMING_WARMUP, TIMING_BATCHES = 1, 2
    MC_DROPOUT_SAMPLES = 2
    CAM_PER_CLASS = 1
    REPLICATION = {**REPLICATION, "epochs": 1, "seeds": [0]}


def run_dir(model: str, seed: int) -> Path:
    return RESULTS / model / f"seed_{seed:02d}"


def hpo_dir(model: str, trial: int) -> Path:
    return RESULTS / model / "hpo" / f"trial_{trial:02d}"


def ablation_dir(variant: str, seed: int) -> Path:
    return RESULTS / "ablation" / variant / f"seed_{seed:02d}"


def ensure_dirs() -> None:
    for d in (COMMON, SUMMARY, LOGS):
        d.mkdir(parents=True, exist_ok=True)
