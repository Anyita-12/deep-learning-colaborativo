"""Datos de diagnóstico para la página Modelos del dashboard (sección 9.4.3 del enunciado).

Lee las corridas versionadas en entregable3/results y escribe en dashboard/data:
  curvas.csv             historial por época (pérdida y F1 macro, train y val) de cada modelo y semilla
  probs_prueba.csv.gz    probabilidades en prueba de una semilla por modelo (la menor disponible)
  costo.csv              parámetros, GFLOPs y latencia medidos en el entrenamiento (mediana entre semillas)
No reentrena ni recalcula métricas: solo reorganiza archivos ya guardados.
Uso: python entregable3/analisis/datos_dashboard_diagnostico.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "entregable3" / "results"
OUT = REPO / "dashboard" / "data"
CARPETAS = sorted([p for p in (RES / "results_e2").iterdir() if p.is_dir() and p.name not in ("ablation", "common")]
                  + [p for p in (RES / "results_e1").iterdir() if p.is_dir()])


def leer(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def main() -> None:
    curvas, probs, costo = [], [], []
    for carpeta in CARPETAS:
        modelo = carpeta.name
        semillas = sorted(s for s in carpeta.glob("seed_*") if (s / "DONE").exists())
        for s in semillas:
            sd = int(s.name.split("_")[1])
            h = leer(s / "history.pkl")
            mejor = leer(s / "timing.pkl").get("best_epoch") if (s / "timing.pkl").exists() else None
            curvas.append(pd.DataFrame({"modelo": modelo, "semilla": sd, "epoca": h["epoch"],
                                        "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                                        "train_f1": h["train_f1_macro"], "val_f1": h["val_f1_macro"],
                                        "mejor_epoca": mejor}))
            if (s / "model_info.pkl").exists() and (s / "timing.pkl").exists():
                mi, t = leer(s / "model_info.pkl"), leer(s / "timing.pkl")
                inf = t.get("inference", {})
                costo.append({"modelo": modelo, "semilla": sd, "params": mi.get("params_total"),
                              "gflops": mi.get("gflops"), "lat_bs1_ms": inf.get("latency_bs1_ms_mean"),
                              "ms_img_lote": inf.get("ms_per_image_mean"), "dispositivo": inf.get("device"),
                              "amp": inf.get("amp"), "t_entreno_s": t.get("train_time_s")})
        s0 = semillas[0]
        tp = leer(s0 / "test_predictions.pkl")
        d = pd.DataFrame(np.asarray(tp["probs"], dtype=float), columns=[f"p_{k}" for k in range(6)])
        d.insert(0, "y", np.asarray(tp["y_true"], dtype=int)); d.insert(0, "semilla", int(s0.name.split("_")[1]))
        d.insert(0, "modelo", modelo)
        probs.append(d)
    pd.concat(curvas).to_csv(OUT / "curvas.csv", index=False, float_format="%.5f")
    pd.concat(probs).to_csv(OUT / "probs_prueba.csv.gz", index=False, float_format="%.6f")
    c = pd.DataFrame(costo)
    c = c.groupby("modelo").agg(params=("params", "median"), gflops=("gflops", "median"),
                                lat_bs1_ms=("lat_bs1_ms", "median"), ms_img_lote=("ms_img_lote", "median"),
                                t_entreno_s=("t_entreno_s", "median"), dispositivo=("dispositivo", "first"),
                                amp=("amp", "first"), n=("semilla", "count")).reset_index()
    c.to_csv(OUT / "costo.csv", index=False, float_format="%.4f")
    print(c.to_string(index=False))


if __name__ == "__main__":
    main()
