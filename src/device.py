"""Configuración de dispositivo y semillas. Funciona igual con GPU o solo CPU."""
from __future__ import annotations

import platform

import keras
import tensorflow as tf


def setup_device(seed: int = 42, mixed_precision: bool = False,
                 deterministic: bool = False) -> dict:
    """Usa GPU si existe; si no, CPU. Devuelve info para registrar en cada corrida.

    deterministic=True activa ops deterministas (más lento; reduce, no elimina,
    diferencias entre CPU y GPU: los resultados NO son idénticos entre hardware).
    """
    keras.utils.set_random_seed(seed)  # fija random, numpy y backend
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)  # no reservar toda la VRAM
        except RuntimeError:
            pass  # la GPU ya estaba inicializada (p. ej. al re-ejecutar la celda)
    if gpus and mixed_precision:  # solo con GPU y si se declara explícitamente
        keras.mixed_precision.set_global_policy("mixed_float16")
    return {
        "tf": tf.__version__,
        "keras": keras.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "device": "GPU" if gpus else "CPU",
        "gpus": [g.name for g in gpus],
        "seed": seed,
        "mixed_precision": bool(gpus and mixed_precision),
        "deterministic": deterministic,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(setup_device(), indent=2))
