"""H3, ablación y predicciones P1-P5, conjuntos adicionales, Friedman/Nemenyi."""
import numpy as np, pandas as pd, pickle
from scipy import stats
from sklearn.metrics import f1_score
from cargar import dirs, lp
import estad as E
from confirmatorio import contraste, serie, Y, DELTA, RIVALES
df = pd.read_pickle("maestra.pkl"); out = {}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

# ---------- H3: no inferioridad / equivalencia frente al mejor rival en limpio ----------
med = df[df.model.isin(RIVALES)].groupby("model").f1.mean().sort_values()
mejor = med.index[-1]
a, b = serie("NAGFuse", "f1"), serie(mejor, "f1"); s = sorted(set(a.index) & set(b.index)); d = (a[s] - b[s]).values
se = d.std(ddof=1) / np.sqrt(len(d)); tq = stats.t.ppf(0.95, len(d) - 1)
p_inf = stats.t.sf((d.mean() + DELTA) / se, len(d) - 1); p_sup = stats.t.cdf((d.mean() - DELTA) / se, len(d) - 1)
out["H3"] = dict(mejor_rival=mejor, R=len(d), dif=d.mean(), ic90=(d.mean() - tq*se, d.mean() + tq*se),
                 p_no_inferioridad=p_inf, p_tost=max(p_inf, p_sup))
print("H3", out["H3"])

# ---------- Ablación (semillas 0-11) ----------
def nag_an(seed, var="NAGFuse"):
    return lp(f"{dirs(var)[seed]}/nagfuse_analisis.pkl")
rows = []
S = list(range(12))
for cond, col in [("limpio", "f1"), ("ruido_0.05", "rob_ruido_0.05"), ("ruido_0.10", "rob_ruido_0.10")]:
    full = serie("NAGFuse", col)[S]
    off = np.array([nag_an(s)["robustez"]["compuerta_off"][cond]["f1_macro"] for s in S])
    var = {"compuerta apagada": off}
    for v in ["sin_denoising", "solo_vit", "solo_cnn"]:
        var[v] = serie(f"abl:{v}", col)[S].values
    for k, x in var.items():
        dd = full.values - x
        rows.append(dict(cond=cond, variante=k, completo=full.mean(), variante_f1=x.mean(), dif=dd.mean(),
                         ic=tuple(np.round(E.boot_semillas(dd)[2], 4)), p_t=E.t_pair(dd), p_w=E.wilc(dd), cliff=E.cliff(dd)))
ab = pd.DataFrame(rows)
for c in ["limpio", "ruido_0.05", "ruido_0.10"]:
    m = ab.cond == c
    ab.loc[m, "p_max_holm"] = np.maximum(E.holm(ab.loc[m, "p_t"]), E.holm(ab.loc[m, "p_w"]))
out["ablacion"] = ab; print(ab.round(4).to_string(index=False))

# ---------- P1-P5 ----------
P = {}
x05 = ab[(ab.cond == "ruido_0.05") & (ab.variante == "compuerta apagada")].iloc[0]
x0 = ab[(ab.cond == "limpio") & (ab.variante == "compuerta apagada")].iloc[0]
P["P1 compuerta reduce F1 ruido0.05 >= delta"] = (round(x05.dif, 4), x05.dif >= DELTA and x05.p_max_holm < 0.05)
P["P2 compuerta no cambia F1 limpio > 0.005"] = (round(x0.dif, 4), abs(x0.dif) <= 0.005)
best_cnn = "RWCNet"
d5 = (serie("NAGFuse", "rob_ruido_0.05") - serie(best_cnn, "rob_ruido_0.05")).dropna()
d10 = (serie("NAGFuse", "rob_ruido_0.10") - serie(best_cnn, "rob_ruido_0.10")).dropna()
g = (d10 - d5).values
P["P3 ventaja sobre RWCNet crece 0.05->0.10"] = (f"{d5.mean():.3f} -> {d10.mean():.3f}", g.mean() > 0 and E.wilc(g) < 0.05)
sig = {c: [] for c in ["limpio", "ruido_0.05", "ruido_0.10"]}; pi1 = []; pis = []
for s in range(20):
    a_ = nag_an(s)
    for c in sig: sig[c].append(np.median(a_["detalles"][c]["sigma"]))
    pi1.append((a_["detalles"]["ruido_0.05"]["pi"] >= 0.999).mean())
    pis.append({c: a_["detalles"][c]["pi"].mean() for c in a_["detalles"]})
P["P4 sigma mediana limpio/0.05/0.10"] = tuple(np.round([np.mean(sig[c]) for c in sig], 4))
P["P4 frac pi=1 con ruido 0.05 (media, min)"] = (round(np.mean(pi1), 4), round(np.min(pi1), 4), np.min(pi1) >= 0.95)
# P5: rama CNN completa vs rama CNN de sin_denoising, ruido 0.05
fc, fs = [], []
S5 = [s for s in S if nag_an(s, "abl:sin_denoising") is not None]
for s in S5:
    fc.append(f1_score(Y, nag_an(s)["detalles"]["ruido_0.05"]["p_c"].argmax(1), average="macro"))
    fs.append(f1_score(Y, nag_an(s, "abl:sin_denoising")["detalles"]["ruido_0.05"]["p_c"].argmax(1), average="macro"))
P["P5 semillas usadas"] = S5
dd = np.array(fc) - np.array(fs)
P["P5 rama CNN con Haar vs sin, ruido0.05"] = (f"{np.mean(fc):.3f} vs {np.mean(fs):.3f}", dd.mean() > 0 and E.wilc(dd) < 0.05)
out["P"] = P; out["pi_medio"] = pd.DataFrame(pis).mean()
for k, v in P.items(): print(k, v)
print(out["pi_medio"].round(3).to_dict())

# ---------- Conjuntos adicionales ----------
ext = []
for ds in ["realwaste", "gd_v2", "warp_c"]:
    col = f"ext_{ds}"; rr = []
    for r in RIVALES[:10]:
        a, b = serie("NAGFuse", col).dropna(), serie(r, col).dropna(); s = sorted(set(a.index) & set(b.index))
        dd = (a[s] - b[s]).values
        rr.append(dict(ds=ds, rival=r, R=len(s), nag=a[s].mean(), riv=b[s].mean(), dif=dd.mean(),
                       ic=tuple(np.round(E.boot_semillas(dd)[2], 4)), p_t=E.t_pair(dd), p_w=E.wilc(dd)))
    t = pd.DataFrame(rr); t["p_max_holm"] = np.maximum(E.holm(t.p_t), E.holm(t.p_w)); ext.append(t)
ext = pd.concat(ext); out["extra"] = ext; print(ext.round(4).to_string(index=False))
rank = df[~df.model.str.startswith("abl")].groupby("model")[["ext_realwaste", "ext_gd_v2", "ext_warp_c"]].mean()
print(rank.round(4).sort_values("ext_gd_v2", ascending=False)); out["extra_medias"] = rank

# ---------- Friedman + Iman-Davenport + Nemenyi (11 modelos profundos, semillas 0-11) ----------
M = ["NAGFuse"] + RIVALES[:10]
X = np.stack([serie(m, "f1")[S].values for m in M], 1)           # (N=12, k=11)
N, k = X.shape; R_ = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, X).mean(0)
chi2 = 12*N/(k*(k+1)) * (np.sum(R_**2) - k*(k+1)**2/4)
FF = (N-1)*chi2 / (N*(k-1) - chi2); pF = stats.f.sf(FF, k-1, (k-1)*(N-1))
q = 3.219  # Nemenyi, alfa 0.05, k = 11 (Demšar, 2006)
CD = q*np.sqrt(k*(k+1)/(6*N))
out["friedman"] = dict(chi2=chi2, FF=FF, p=pF, CD=CD, rangos=dict(zip(M, R_)))
print("Friedman", round(chi2, 2), "F_ID", round(FF, 2), "p", pF, "CD", round(CD, 3))
print(sorted(zip(R_.round(2), M)))
pd.to_pickle(out, "secundario.pkl")
