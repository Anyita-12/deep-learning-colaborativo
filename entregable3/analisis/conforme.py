"""Predicción conforme dividida (LAC, alfa = 0.10): calibración en validación, cobertura en prueba, por clase."""
import numpy as np, pandas as pd
from cargar import dirs, lp, REPO
CL = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]; A = 0.10
rows = []
for m in ["NAGFuse", "ViT-B16", "RWCNet", "GFN", "Celik-HC"]:
    for s, d in dirs(m).items():
        v, t = lp(f"{d}/val_predictions.pkl"), lp(f"{d}/test_predictions.pkl")
        pv, yv, pt, yt = np.asarray(v["probs"]), np.asarray(v["y_true"]), np.asarray(t["probs"]), np.asarray(t["y_true"])
        sc = 1 - pv[np.arange(len(yv)), yv]; n = len(sc)
        q = np.quantile(sc, np.ceil((n + 1) * (1 - A)) / n, method="higher")
        S = (1 - pt) <= q; cov = S[np.arange(len(yt)), yt]
        r = dict(modelo=m, semilla=s, cobertura=cov.mean(), tamano=S.sum(1).mean())
        for k, c in enumerate(CL): r[f"cob_{c}"] = cov[yt == k].mean()
        rows.append(r)
d = pd.DataFrame(rows); g = d.groupby("modelo").mean(numeric_only=True).drop(columns="semilla")
g["R"] = d.groupby("modelo").size(); g["peor_clase"] = g[[f"cob_{c}" for c in CL]].min(1)
g.round(3).to_csv(f"{REPO}/entregable3/resultados/conforme.csv")
print(g.round(3).to_string())
