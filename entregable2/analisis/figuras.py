"""Estilo común de las figuras del Entregable 2 (matplotlib)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# Paleta categórica en orden fijo (azul, naranja, aqua, amarillo, magenta, verde, violeta, rojo)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#b5b4ae"

# Dos categorías para los 10 modelos: régimen de entrenamiento (la identidad se da con etiquetas directas)
PREENTRENADOS = {"Celik-HC", "ViT-B16", "GFN", "ResMsCapsule", "RWCNet"}
DESDE_CERO = {"MbXception", "FocusRCNet", "CNN", "MobileNetV2", "ResNet-18"}
ORIGEN_E1 = {"CNN", "MobileNetV2", "ResNet-18"}


def color_regimen(m: str) -> str:
    return SERIES[0] if m in PREENTRENADOS else SERIES[1]


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
        "figure.dpi": 110, "text.color": INK, "axes.axisbelow": True,
    })


def leyenda_regimen(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=SERIES[0], markersize=8, label="Preentrenado (ImageNet)"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor=SERIES[1], markersize=8, label="Desde cero")]
    ax.legend(handles=h, loc=loc)
