"""Prepara los datos precalculados del dashboard (sin pesos): copia la tabla maestra y extrae salidas por imagen."""
import json, shutil, numpy as np, pandas as pd, pickle
from pathlib import Path
from cargar import REPO, dirs, lp
R = Path(REPO); OUT = R / "dashboard" / "data"; OUT.mkdir(parents=True, exist_ok=True)
for f in (R / "entregable3" / "resultados").glob("*.csv"):
    shutil.copy2(f, OUT / f.name)
# salidas por imagen de NAGFuse semilla 0 en prueba, por perturbación
d = dirs("NAGFuse")[0]; a = lp(f"{d}/nagfuse_analisis.pkl"); tp = lp(f"{d}/test_predictions.pkl")
rows = []
for cond, v in a["detalles"].items():
    p = v["pi"][:, None] * v["p_v"] + (1 - v["pi"][:, None]) * v["p_c"]
    for i in range(len(v["sigma"])):
        rows.append(dict(cond=cond, i=i, archivo=tp["filepaths"][i], y=int(tp["y_true"][i]), sigma=float(v["sigma"][i]),
                         pi=float(v["pi"][i]), pred=int(p[i].argmax()), conf=float(p[i].max()),
                         **{f"p_{k}": float(p[i, k]) for k in range(6)}, **{f"pv_{k}": float(v["p_v"][i, k]) for k in range(6)},
                         **{f"pc_{k}": float(v["p_c"][i, k]) for k in range(6)}))
pd.DataFrame(rows).to_csv(OUT / "nagfuse_s00_por_imagen.csv.gz", index=False, compression="gzip")
# umbral conforme LAC (alfa 0.10) de NAGFuse semilla 0, calibrado en validación
v = lp(f"{d}/val_predictions.pkl"); pv, yv = np.asarray(v["probs"]), np.asarray(v["y_true"])
sc = 1 - pv[np.arange(len(yv)), yv]; n = len(sc); q = float(np.quantile(sc, np.ceil((n + 1) * 0.9) / n, method="higher"))
# EDA: conteos por clase y partición
sp = pd.read_csv(R / "data" / "splits" / "trashnet_split_v2_s1234.csv")
sp.groupby(["split", "label"]).size().rename("n").reset_index().to_csv(OUT / "eda_conteos.csv", index=False)
cfg = dict(clases=["cardboard", "glass", "metal", "paper", "plastic", "trash"], s_lo=float(a["s_lo"]), rho=float(a["rho"]),
           pi0=float(a["pi0"]), q_conforme=q, alfa=0.10, media=[0.485, 0.456, 0.406], desv=[0.229, 0.224, 0.225],
           sigma_limpio_p99=float(np.percentile(a["detalles"]["limpio"]["sigma"], 99)))
json.dump(cfg, open(OUT / "config_modelo.json", "w"), indent=2)
print(cfg); print(sorted(p.name for p in OUT.iterdir()))
