"""Pruebas estadísticas del benchmark (Secciones 5.2–5.4) y tamaño de muestra.

Unidad de réplica: la semilla. Las tres arquitecturas comparten partición y
semillas, así que las métricas están pareadas por semilla (bloques).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from . import config as C

Z = st.norm.ppf


# ------------------------------------------------------------------ tamaño de muestra
def sample_size(diffs: dict[str, np.ndarray], delta: float = C.DELTA, alpha: float = C.ALPHA,
                power: float = C.POWER, r_min: int = C.R_MIN, r_max: int = C.R_MAX) -> dict:
    """R = (z_{1-a/2} + z_{1-b})^2 sd_d^2 / delta^2 por par; se toma el máximo."""
    zsum = Z(1 - alpha / 2) + Z(power)
    rows = []
    for pair, d in diffs.items():
        d = np.asarray(d, float)
        sd = float(d.std(ddof=1))
        r = math.ceil(zsum ** 2 * sd ** 2 / delta ** 2) if sd > 0 else 1
        sw = st.shapiro(d) if len(d) >= 3 else None
        rows.append({"par": pair, "n_piloto": len(d), "media_d": float(d.mean()), "sd_d": sd, "R_par": r,
                     "shapiro_W": float(sw.statistic) if sw else np.nan, "shapiro_p": float(sw.pvalue) if sw else np.nan})
    tab = pd.DataFrame(rows)
    r_calc = int(tab["R_par"].max())
    r_final = min(max(r_calc, r_min), r_max)
    sd_max = float(tab["sd_d"].max())
    mde = zsum * sd_max / math.sqrt(r_final)
    return {"tabla": tab, "R_calculado": r_calc, "R_final": r_final, "R_min": r_min, "R_max": r_max,
            "limitado_por_R_max": r_calc > r_max, "MDE": float(mde), "delta": delta, "alpha": alpha,
            "power": power, "z_sum": float(zsum), "MDE_supera_delta": bool(mde > delta)}


# ------------------------------------------------------------------ utilidades
def wide(df: pd.DataFrame, metric: str = C.PRIMARY_METRIC, models=C.MODELS) -> pd.DataFrame:
    """Tabla semilla x modelo, solo con semillas completas en los tres modelos."""
    w = df.pivot(index="seed", columns="model", values=metric)[list(models)]
    return w.dropna()


def mean_ci(x, alpha: float = C.ALPHA, n_boot: int = C.N_BOOT, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else np.nan
    h = st.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, (n_boot, n))].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"media": m, "de": sd, "n": n, "ic_t": (m - h, m + h), "ic_boot": (float(lo), float(hi))}


# ------------------------------------------------------------------ Friedman exacto
def friedman_exact(w: pd.DataFrame, max_enum: int = 2_000_000, n_mc: int = 200_000, seed: int = 0) -> dict:
    """Friedman con p exacto por enumeración de permutaciones dentro de cada bloque
    (k!^n casos). Si son demasiados, Monte Carlo. Incluye la W de Kendall."""
    X = w.to_numpy(float)
    n, k = X.shape
    R = np.apply_along_axis(st.rankdata, 1, X)
    def q_stat(Rm):
        Rj = Rm.sum(0)
        return 12 / (n * k * (k + 1)) * (Rj ** 2).sum() - 3 * n * (k + 1)
    q_obs = q_stat(R)
    chi = st.friedmanchisquare(*[X[:, j] for j in range(k)])
    perms = np.array(list(itertools.permutations(range(k))))
    total = len(perms) ** n
    if total <= max_enum:
        # suma de rangos por columna para cada combinación de permutaciones (enumeración completa)
        sums = np.zeros((1, k))
        for i in range(n):
            sums = (sums[:, None, :] + R[i][perms][None, :, :]).reshape(-1, k)
        qs = 12 / (n * k * (k + 1)) * (sums ** 2).sum(1) - 3 * n * (k + 1)
        p = float((qs >= q_obs - 1e-12).mean())
        method = f"exacto ({total} permutaciones)"
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(perms), (n_mc, n))
        sums = np.zeros((n_mc, k))
        for i in range(n):
            sums += R[i][perms[idx[:, i]]]
        qs = 12 / (n * k * (k + 1)) * (sums ** 2).sum(1) - 3 * n * (k + 1)
        p = float(((qs >= q_obs - 1e-12).sum() + 1) / (n_mc + 1))
        method = f"Monte Carlo ({n_mc} permutaciones)"
    return {"Q": float(q_obs), "p_exacto": p, "metodo_p": method, "p_chi2": float(chi.pvalue),
            "W_kendall": float(q_obs / (n * (k - 1))), "n_bloques": n, "k": k,
            "rangos_medios": dict(zip(w.columns, R.mean(0).round(3)))}


# ------------------------------------------------------------------ comparaciones por pares
def pairwise(w: pd.DataFrame, alpha: float = C.ALPHA, n_boot: int = C.N_BOOT, seed: int = 0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for a, b in itertools.combinations(w.columns, 2):
        d = (w[a] - w[b]).to_numpy(float)
        n = len(d)
        md, sd = d.mean(), d.std(ddof=1)
        tt = st.ttest_rel(w[a], w[b])
        h = st.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
        boots = d[rng.integers(0, n, (n_boot, n))].mean(1)
        try:
            wx = st.wilcoxon(w[a], w[b], method="exact" if n <= 25 and np.all(d != 0) else "auto")
            w_stat, w_p = float(wx.statistic), float(wx.pvalue)
        except ValueError:
            w_stat, w_p = np.nan, 1.0
        nz = d[d != 0]
        rk = st.rankdata(np.abs(nz))
        rbc = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else 0.0
        rows.append({"A": a, "B": b, "n": n, "dif_media": md, "de_dif": sd,
                     "ic_t_lo": md - h, "ic_t_hi": md + h,
                     "ic_boot_lo": float(np.percentile(boots, 100 * alpha / 2)),
                     "ic_boot_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
                     "t": float(tt.statistic), "p_t": float(tt.pvalue), "d_z": md / sd if sd > 0 else np.nan,
                     "W_wilcoxon": w_stat, "p_wilcoxon": w_p, "r_rango_biserial": rbc,
                     "shapiro_p_dif": float(st.shapiro(d).pvalue) if n >= 3 else np.nan,
                     "supera_delta": bool(abs(md) >= C.DELTA)})
    tab = pd.DataFrame(rows)
    for col in ("p_t", "p_wilcoxon"):
        tab[col + "_holm"] = multipletests(tab[col], alpha=alpha, method="holm")[1]
    return tab


def posthoc(w: pd.DataFrame) -> dict:
    """Nemenyi y Conover (scikit-posthocs), como complemento."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        return {"disponible": False}
    X = w.to_numpy(float)
    return {"disponible": True,
            "nemenyi": pd.DataFrame(sp.posthoc_nemenyi_friedman(X), index=w.columns, columns=w.columns),
            "conover": pd.DataFrame(sp.posthoc_conover_friedman(X, p_adjust="holm"), index=w.columns, columns=w.columns)}


# ------------------------------------------------------------------ McNemar por imagen
def mcnemar_pair(y, pa, pb) -> dict:
    ca, cb = pa == y, pb == y
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    exact = (b + c) < 25
    r = sm_mcnemar([[int((ca & cb).sum()), b], [c, int((~ca & ~cb).sum())]], exact=exact, correction=True)
    return {"b": b, "c": c, "exacto": exact, "estadistico": float(r.statistic), "p": float(r.pvalue)}


def mcnemar_all(preds: dict, alpha: float = C.ALPHA) -> pd.DataFrame:
    """preds[modelo][semilla] = dict con y_true, y_pred, idx (mismo orden de test)."""
    rows = []
    models = list(preds)
    seeds = sorted(set.intersection(*[set(preds[m]) for m in models]))
    for s in seeds:
        for a, b in itertools.combinations(models, 2):
            A, B = preds[a][s], preds[b][s]
            assert np.array_equal(A["idx"], B["idx"])
            r = mcnemar_pair(A["y_true"], A["y_pred"], B["y_pred"])
            rows.append({"seed": s, "A": a, "B": b, **r})
    tab = pd.DataFrame(rows)
    tab["p_holm_semilla"] = np.nan
    for s, g in tab.groupby("seed"):
        tab.loc[g.index, "p_holm_semilla"] = multipletests(g["p"], alpha=alpha, method="holm")[1]
    return tab


# ------------------------------------------------------------------ AUC: bootstrap pareado estratificado
def auc_ovr_macro(y, probs) -> float:
    """AUC-ROC macro uno-contra-resto por la forma de Mann-Whitney (igual a sklearn, más rápida)."""
    aucs = []
    for k in range(probs.shape[1]):
        pos = y == k
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            continue
        r = st.rankdata(probs[:, k])
        aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs))


def auc_bootstrap(y, probs_a, probs_b, n_boot: int = C.N_BOOT, seed: int = 0, alpha: float = C.ALPHA) -> dict:
    """Diferencia de AUC-ROC macro (OvR) entre dos modelos sobre las mismas imágenes,
    remuestreando imágenes dentro de cada clase."""
    rng = np.random.default_rng(seed)
    labels = list(range(probs_a.shape[1]))
    auc = auc_ovr_macro
    obs = auc(y, probs_a) - auc(y, probs_b)
    by_class = [np.flatnonzero(y == k) for k in labels]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ix = np.concatenate([rng.choice(ix_k, len(ix_k), replace=True) for ix_k in by_class])
        diffs[i] = auc(y[ix], probs_a[ix]) - auc(y[ix], probs_b[ix])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"dif_auc": float(obs), "ic_lo": float(lo), "ic_hi": float(hi), "p_bootstrap": float(p), "n_boot": n_boot}


# ------------------------------------------------------------------ descomposición de la varianza
def variance_decomposition(df: pd.DataFrame, metric: str = C.PRIMARY_METRIC) -> dict:
    groups = [g[metric].to_numpy(float) for _, g in df.groupby("model")]
    f = st.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    k, n_tot = len(groups), len(allv)
    n0 = n_tot / k
    msb, msw = ssb / (k - 1), ssw / (n_tot - k)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return {"F": float(f.statistic), "p": float(f.pvalue), "eta2": float(ssb / (ssb + ssw)),
            "icc1": float(icc), "ss_entre": float(ssb), "ss_dentro": float(ssw), "gl": (k - 1, n_tot - k)}
