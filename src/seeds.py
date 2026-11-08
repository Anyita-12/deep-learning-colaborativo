"""Semillas, determinismo y estados aleatorios (Sección 4.3).

Se busca determinismo completo, pero algunas operaciones de cuDNN (por ejemplo,
el backward de MaxPool y de AdaptiveAvgPool) no tienen versión determinista y
AMP introduce redondeos: por eso use_deterministic_algorithms va con warn_only.
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def configure_determinism() -> dict:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Semilla de numpy y random en cada worker del DataLoader. La aumentación no
    depende de esto (usa una semilla por imagen y época, ver data.EpochSampler),
    pero evita que alguna librería use un estado no controlado."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
