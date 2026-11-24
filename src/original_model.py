"""Modelo original NAGFuse (punto de entrada pedido en la Sección 12, ítem 4, del enunciado del E3).

La implementación vive en ``entregable3/e3/nagfuse.py`` (versionada con el tag ``e3-protocolo-v3`` y probada en
``entregable3/tests/test_nagfuse.py``). Este módulo solo la reexporta para no duplicar código ni alterar el
artefacto congelado del protocolo.

Uso::

    from src.original_model import NAGFuse, build_nagfuse, mad_sigma, gate_value, mixture_log_prob
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "entregable3"))

from e3.nagfuse import (NAGFuse, build_nagfuse, gate_value, haar_dwt2, haar_idwt2,  # noqa: E402,F401
                        mad_sigma, mixture_log_prob, soft_threshold, wavelet_shrink)

__all__ = ["NAGFuse", "build_nagfuse", "gate_value", "haar_dwt2", "haar_idwt2", "mad_sigma", "mixture_log_prob",
           "soft_threshold", "wavelet_shrink"]
