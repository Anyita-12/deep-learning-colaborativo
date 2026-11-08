"""Estilo de figuras y guardado (PNG 200 dpi + datos fuente en pickle)."""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import config as C
from .persist import save_pickle


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#3A506B", "axes.labelcolor": "#0B132B", "text.color": "#0B132B",
        "xtick.color": "#1C2541", "ytick.color": "#1C2541", "axes.grid": True, "grid.color": "#E3E8EE",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 100,
    })


def save(fig, name: str, data=None) -> None:
    """Guarda results/figures/<name>.png y, si se da, results/summary/fig_<name>.pkl."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", dpi=C.FIG_DPI, bbox_inches="tight")
    if data is not None:
        save_pickle(data, C.SUMMARY / f"fig_{name}.pkl", verify=False)


def color(model: str) -> str:
    return C.MODEL_COLORS[model]
