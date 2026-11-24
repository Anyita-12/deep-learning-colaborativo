"""Tabla maestra única (libro y dashboard) y figuras del E3. Ejecutar tras confirmatorio.py y secundario.py."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from cargar import REPO, dirs, lp
OUT = Path(REPO) / "entregable3" / "resultados"; OUT.mkdir(exist_ok=True)
FIG = Path(REPO) / "docs" / "images" / "e3"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_pickle("maestra.pkl"); C = pd.read_pickle("confirmatorio.pkl"); S = pd.read_pickle("secundario.pkl")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"NAGFuse": "#c0392b", "ViT-B16": "#2e86c1", "RWCNet": "#27ae60", "GFN": "#8e44ad", "Celik-HC": "#d68910"}

def ic_t(x):
    x = np.asarray(x)[~np.isnan(x)]; n = len(x)
    if n < 2: return np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return x.mean() - h, x.mean() + h

# ---------- tabla maestra resumen ----------
cols = {"f1": "F1 limpio", "mcc": "MCC", "ece": "ECE", "nll": "NLL", "rob_ruido_0.05": "F1 ruido 0.05",
        "rob_ruido_0.10": "F1 ruido 0.10", "rob_desenfoque": "F1 desenfoque", "rob_oclusion": "F1 oclusión",
        "rob_fondo_sustituido": "F1 fondo", "ext_realwaste": "F1 RealWaste", "ext_gd_v2": "F1 Garbage v2",
        "ext_warp_c": "F1 WaRP-C"}
rows = []
for m, g in df.groupby("model"):
    r = {"modelo": m.replace("abl:", "NAGFuse/"), "R": len(g)}
    for c, n in cols.items():
        x = g[c].astype(float).values; lo, hi = ic_t(x)
        r[n] = np.nanmean(x); r[n + " IC_lo"] = lo; r[n + " IC_hi"] = hi; r[n + " n"] = int((~np.isnan(x)).sum())
    rows.append(r)
M = pd.DataFrame(rows).sort_values("F1 limpio", ascending=False)
M.to_csv(OUT / "tabla_maestra.csv", index=False)
df.to_csv(OUT / "por_corrida.csv", index=False)
for k, t in C.items(): t.to_csv(OUT / f"contrastes_{k}.csv", index=False)
S["ablacion"].to_csv(OUT / "ablacion.csv", index=False); S["extra"].to_csv(OUT / "contrastes_extra.csv", index=False)
pd.DataFrame([S["H3"]]).to_csv(OUT / "H3_no_inferioridad.csv", index=False)
pd.DataFrame([{"prediccion": k, "resultado": str(v)} for k, v in S["P"].items()]).to_csv(OUT / "predicciones.csv", index=False)
fr = S["friedman"]; pd.DataFrame([{"modelo": k, "rango_medio": v} for k, v in fr["rangos"].items()]).assign(
    chi2=fr["chi2"], F_ID=fr["FF"], p=fr["p"], CD=fr["CD"]).to_csv(OUT / "friedman_nemenyi.csv", index=False)

# ---------- Fig 1: forest H1 y H2 ----------
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, k, tit in [(axs[0], "H1_f1", "F1 macro limpio (H1)"), (axs[1], "H2_ruido005", "F1 macro con ruido σ = 0.05 (H2)")]:
    t = C[k].set_index("rival").reindex(C["H1_f1"].rival); y = np.arange(len(t))
    for i, (riv, r) in enumerate(t.iterrows()):
        if np.isnan(r.media_dif): ax.text(0.01, i, "sin datos", va="center", fontsize=7, color="gray"); continue
        c = "#c0392b" if r.supera else ("#7f8c8d" if not r.mejor_sig else "#e67e22")
        ax.errorbar(r.media_dif, i, xerr=[[r.media_dif - r.ic_lo], [r.ic_hi - r.media_dif]], fmt="o", color=c, ms=4, capsize=2)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(0.02, color="k", ls="--", lw=0.8)
    ax.set_xscale("symlog", linthresh=0.05); ax.set_title(tit); ax.set_xlabel("Diferencia NAGFuse − rival (IC 95 % bootstrap)")
axs[0].set_yticks(y, t.index); axs[0].invert_yaxis()
fig.text(0.5, -0.02, "Rojo: supera con diferencia > δ = 0.02 (línea discontinua) y p de Holm < 0.05; naranja: significativo pero < δ; gris: no significativo.", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "forest_H1_H2.png", bbox_inches="tight"); plt.close()

# ---------- Fig 2: curva de robustez al ruido ----------
fig, ax = plt.subplots(figsize=(6, 3.8)); xs = [0, 0.05, 0.10]
for m in ["NAGFuse", "ViT-B16", "RWCNet", "GFN", "Celik-HC", "ResMsCapsule", "ResNet-18"]:
    g = df[df.model == m]; mu = [g[c].mean() for c in ["f1", "rob_ruido_0.05", "rob_ruido_0.10"]]
    ci = np.array([ic_t(g[c].values) for c in ["f1", "rob_ruido_0.05", "rob_ruido_0.10"]])
    ax.errorbar(xs, mu, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], marker="o", ms=4, capsize=2,
                label=f"{m} (R={len(g)})", color=COL.get(m), lw=2 if m == "NAGFuse" else 1)
ax.set_xticks(xs, ["limpio", "σ = 0.05", "σ = 0.10"]); ax.set_ylabel("F1 macro en prueba (media, IC 95 %)")
ax.set_xlabel("Ruido gaussiano añadido a la imagen en [0, 1]"); ax.set_title("Robustez al ruido gaussiano"); ax.set_ylim(0, 1)
ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(FIG / "curva_ruido.png"); plt.close()

# ---------- Fig 3: conjuntos adicionales ----------
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
mods = ["NAGFuse", "ViT-B16", "GFN", "RWCNet", "Celik-HC", "ResMsCapsule", "ResNet-18", "CNN"]
for ax, (c, n) in zip(axs, [("ext_realwaste", "RealWaste"), ("ext_gd_v2", "Garbage v2"), ("ext_warp_c", "WaRP-C")]):
    for i, m in enumerate(mods):
        x = df[df.model == m][c].dropna().values; lo, hi = ic_t(x)
        ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", ms=4, capsize=2, color=COL.get(m, "#555"))
        ax.text(1.0, i, f"R={len(x)}", fontsize=6, va="center", ha="right", color="gray")
    ax.set_title(n); ax.set_xlim(0, 1); ax.set_xlabel("F1 macro (clases presentes)")
axs[0].set_yticks(range(len(mods)), mods); axs[0].invert_yaxis()
fig.suptitle("Generalización sin reentrenar a tres conjuntos adicionales (media e IC 95 % t)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "conjuntos_adicionales.png"); plt.close()

# ---------- Fig 4: diagrama de diferencia crítica ----------
rk = sorted(fr["rangos"].items(), key=lambda kv: kv[1]); k = len(rk); CD = fr["CD"]
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.axis("off")
ax.hlines(0.8, 1, k, color="k")
for i in range(1, k + 1):
    ax.vlines(i, 0.78, 0.82, color="k"); ax.text(i, 0.85, str(i), ha="center", fontsize=7)
for j, (m, r) in enumerate(rk):
    left = j < (k + 1) // 2; yy = 0.65 - 0.1 * (j if left else k - 1 - j)
    ax.plot([r, r], [0.8, yy], color=COL.get(m, "#555"), lw=0.8)
    ax.plot([r, 1 if left else k], [yy, yy], color=COL.get(m, "#555"), lw=0.8)
    ax.text(0.95 if left else k + 0.05, yy, f"{m} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=7)
ax.plot([1, 1 + CD], [0.97, 0.97], color="k", lw=2); ax.text(1 + CD / 2, 0.99, f"CD = {CD:.2f}", ha="center", fontsize=7)
# grupos no distinguibles
yb = 0.74; last = -1
for i in range(k):
    j = i
    while j + 1 < k and rk[j + 1][1] - rk[i][1] <= CD: j += 1
    if j > i and j > last:
        ax.plot([rk[i][1] - 0.03, rk[j][1] + 0.03], [yb, yb], color="k", lw=3); yb -= 0.035; last = j
ax.set_title("Diferencia crítica de Nemenyi (F1 limpio, 11 modelos, semillas 0–11; rango 1 = mejor)", fontsize=8)
fig.savefig(FIG / "diferencia_critica.png", bbox_inches="tight"); plt.close()

# ---------- Fig 5: estimador de ruido y compuerta ----------
fig, ax = plt.subplots(figsize=(6, 3.2)); sig = {c: [] for c in ["limpio", "ruido_0.05", "ruido_0.10"]}
for s in range(20):
    a = lp(f"{dirs('NAGFuse')[s]}/nagfuse_analisis.pkl")
    for c in sig: sig[c].append(a["detalles"][c]["sigma"])
s_lo = a["s_lo"]
for c, lab, col in [("limpio", "limpio", "#2e86c1"), ("ruido_0.05", "σ = 0.05", "#e67e22"), ("ruido_0.10", "σ = 0.10", "#c0392b")]:
    ax.hist(np.concatenate(sig[c][:1]), bins=60, range=(0, 0.13), alpha=0.6, label=lab, color=col)
ax.axvline(s_lo, color="k", ls="--", lw=0.8); ax.axvline(2 * s_lo, color="k", ls=":", lw=0.8)
ax.text(s_lo, ax.get_ylim()[1] * 0.9, " s_lo", fontsize=7); ax.text(2 * s_lo, ax.get_ylim()[1] * 0.8, " ρ·s_lo", fontsize=7)
ax.set_xlabel("σ estimado por MAD (361 imágenes de prueba)"); ax.set_ylabel("Número de imágenes")
ax.set_title("Estimador de ruido σ̂ y umbrales de la compuerta (semilla 0)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "estimador_compuerta.png"); plt.close()

# ---------- Fig 6: ablación ----------
ab = S["ablacion"]; fig, ax = plt.subplots(figsize=(7, 3.4)); var = ab.variante.unique()
for j, (cnd, col) in enumerate([("limpio", "#2e86c1"), ("ruido_0.05", "#e67e22"), ("ruido_0.10", "#c0392b")]):
    t = ab[ab.cond == cnd]
    for i, v in enumerate(var):
        r = t[t.variante == v].iloc[0]; lo, hi = r.ic
        ax.errorbar(r.dif, i + (j - 1) * 0.22, xerr=[[r.dif - lo], [hi - r.dif]], fmt="o", ms=4, capsize=2, color=col,
                    label=cnd if i == 0 else None, mfc=col if r.p_max_holm < 0.05 else "white")
ax.axvline(0, color="k", lw=0.8); ax.set_xscale("symlog", linthresh=0.05)
ax.set_yticks(range(len(var)), var); ax.invert_yaxis(); ax.legend(fontsize=7)
ax.set_xlabel("F1 del modelo completo − F1 de la variante (IC 95 %; relleno = Holm p < 0.05)")
ax.set_title("Ablación de NAGFuse (semillas 0–11)"); fig.tight_layout(); fig.savefig(FIG / "ablacion.png"); plt.close()
print("tabla_maestra:", M.shape, "| figuras:", sorted(p.name for p in FIG.iterdir()))
print(M[["modelo", "R", "F1 limpio", "F1 ruido 0.05", "ECE", "F1 RealWaste"]].round(4).to_string(index=False))
