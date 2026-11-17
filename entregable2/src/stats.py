"""Pruebas estadísticas del benchmark. Unidad de réplica: la semilla (bloques pareados)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from . import config as C

Z = st.norm.ppf


def wide(df: pd.DataFrame, metric: str = C.PRIMARY_METRIC, models=None) -> pd.DataFrame:
    w = df.pivot(index="seed", columns="model", values=metric)
    if models is not None:
        w = w[[m for m in models if m in w.columns]]
    return w.dropna()


def mean_ci(x, alpha: float = C.ALPHA, n_boot: int = C.N_BOOT, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    sd = x.std(ddof=1) if n > 1 else np.nan
    h = st.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
    boots = x[np.random.default_rng(seed).integers(0, n, (n_boot, n))].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"media": m, "de": sd, "n": n, "ic_t": (m - h, m + h), "ic_boot": (float(lo), float(hi))}


def friedman_permutation(w: pd.DataFrame, n_mc: int = 100_000, seed: int = 0) -> dict:
    """Friedman con p por permutación de rangos dentro de cada bloque (exacto si k!^n <= 2e6) y W de Kendall."""
    X = w.to_numpy(float)
    n, k = X.shape
    R = np.apply_along_axis(st.rankdata, 1, X)

    def q_stat(sums):
        return 12 / (n * k * (k + 1)) * (sums ** 2).sum(-1) - 3 * n * (k + 1)

    q_obs = float(q_stat(R.sum(0)))
    chi = st.friedmanchisquare(*[X[:, j] for j in range(k)])
    total = math.factorial(k) ** n
    if total <= 2_000_000:
        perms = np.array(list(itertools.permutations(range(k))))
        sums = np.zeros((1, k))
        for i in range(n):
            sums = (sums[:, None, :] + R[i][perms][None, :, :]).reshape(-1, k)
        p, method = float((q_stat(sums) >= q_obs - 1e-12).mean()), f"exacto ({total} permutaciones)"
    else:
        rng = np.random.default_rng(seed)
        sums = np.zeros((n_mc, k))
        for i in range(n):
            sums += np.take_along_axis(np.broadcast_to(R[i], (n_mc, k)), rng.permuted(np.tile(np.arange(k), (n_mc, 1)), axis=1), 1)
        p = float(((q_stat(sums) >= q_obs - 1e-12).sum() + 1) / (n_mc + 1))
        method = f"Monte Carlo ({n_mc} permutaciones)"
    return {"Q": q_obs, "p_permutacion": p, "metodo_p": method, "p_chi2": float(chi.pvalue),
            "W_kendall": float(q_obs / (n * (k - 1))), "n_bloques": n, "k": k,
            "rangos_medios": dict(zip(w.columns, R.mean(0).round(3)))}


def pairwise(w: pd.DataFrame, alpha: float = C.ALPHA, n_boot: int = C.N_BOOT, seed: int = 0,
             pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for a, b in pairs or list(itertools.combinations(w.columns, 2)):
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
        rows.append({"A": a, "B": b, "n": n, "dif_media": md, "de_dif": sd, "ic_t_lo": md - h, "ic_t_hi": md + h,
                     "ic_boot_lo": float(np.percentile(boots, 100 * alpha / 2)),
                     "ic_boot_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
                     "t": float(tt.statistic), "p_t": float(tt.pvalue), "d_z": md / sd if sd > 0 else np.nan,
                     "W_wilcoxon": w_stat, "p_wilcoxon": w_p, "r_rango_biserial": rbc,
                     "shapiro_p_dif": float(st.shapiro(d).pvalue) if n >= 3 else np.nan,
                     "supera_delta": bool(abs(md) >= C.DELTA)})
    tab = pd.DataFrame(rows)
    for col in ("p_t", "p_wilcoxon"):
        tab[col + "_holm"] = multipletests(tab[col], alpha=alpha, method="holm")[1]
    m = len(tab)
    tab["p_wilcoxon_holm_minimo_posible"] = np.minimum(1.0, m * 2.0 / 2.0 ** tab["n"])
    tab["relevante_y_significativa_wilcoxon"] = tab["supera_delta"] & (tab["p_wilcoxon_holm"] < alpha)
    tab["relevante_y_significativa"] = tab["supera_delta"] & (tab["p_t_holm"] < alpha)
    return tab


def posthoc(w: pd.DataFrame) -> dict:
    try:
        import scikit_posthocs as sp
    except ImportError:
        return {"disponible": False}
    X = w.to_numpy(float)
    return {"disponible": True,
            "nemenyi": pd.DataFrame(np.asarray(sp.posthoc_nemenyi_friedman(X)), index=w.columns, columns=w.columns),
            "conover": pd.DataFrame(np.asarray(sp.posthoc_conover_friedman(X, p_adjust="holm")), index=w.columns,
                                    columns=w.columns)}


def mcnemar_pair(y, pa, pb) -> dict:
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    exact = (b + c) < 25
    r = sm_mcnemar([[int((ca & cb).sum()), b], [c, int((~ca & ~cb).sum())]], exact=exact, correction=True)
    return {"b": b, "c": c, "exacto": exact, "estadistico": float(r.statistic), "p": float(r.pvalue)}


def mcnemar_all(preds: dict, alpha: float = C.ALPHA) -> pd.DataFrame:
    rows = []
    models = list(preds)
    seeds = sorted(set.intersection(*[set(preds[m]) for m in models]))
    for s in seeds:
        for a, b in itertools.combinations(models, 2):
            A, B = preds[a][s], preds[b][s]
            order_a, order_b = np.argsort(A["idx"]), np.argsort(B["idx"])
            ya = A["y_true"][order_a]
            if not np.array_equal(ya, B["y_true"][order_b]):
                raise ValueError("Las predicciones de prueba no están alineadas.")
            rows.append({"seed": s, "A": a, "B": b, **mcnemar_pair(ya, A["y_pred"][order_a], B["y_pred"][order_b])})
    tab = pd.DataFrame(rows)
    tab["p_holm_semilla"] = np.nan
    for _, g in tab.groupby("seed"):
        tab.loc[g.index, "p_holm_semilla"] = multipletests(g["p"], alpha=alpha, method="holm")[1]
    return tab


def overfitting_gap(hist: dict) -> dict:
    """Brecha entre entrenamiento y validación en la mejor época y tendencia de la pérdida de validación."""
    best = int(np.argmax(hist["val_f1_macro"]))
    tail = np.array(hist["val_loss"][best:], float)
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else float("nan")
    return {"best_epoch": best, "gap_f1": float(hist["train_f1_macro"][best] - hist["val_f1_macro"][best]),
            "gap_loss": float(hist["val_loss"][best] - hist["train_loss"][best]), "val_loss_slope_after_best": slope}


def variance_decomposition(df: pd.DataFrame, metric: str = C.PRIMARY_METRIC) -> dict:
    groups = [g[metric].to_numpy(float) for _, g in df.groupby("model")]
    f = st.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    k, n_tot = len(groups), len(allv)
    msb, msw = ssb / (k - 1), ssw / (n_tot - k)
    icc = (msb - msw) / (msb + (n_tot / k - 1) * msw)
    return {"F": float(f.statistic), "p": float(f.pvalue), "eta2": float(ssb / (ssb + ssw)), "icc1": float(icc),
            "ss_entre": float(ssb), "ss_dentro": float(ssw), "gl": (k - 1, n_tot - k)}


def minimum_detectable_effect(w: pd.DataFrame, alpha: float = C.ALPHA, power: float = C.POWER) -> pd.DataFrame:
    zsum = Z(1 - alpha / 2) + Z(power)
    rows = []
    for a, b in itertools.combinations(w.columns, 2):
        d = (w[a] - w[b]).to_numpy(float)
        sd = float(d.std(ddof=1))
        r_needed = math.ceil(zsum ** 2 * sd ** 2 / C.DELTA ** 2) if sd > 0 else 1
        rows.append({"A": a, "B": b, "sd_d": sd, "n": len(d), "MDE": zsum * sd / math.sqrt(len(d)), "R_necesario": r_needed})
    return pd.DataFrame(rows)
