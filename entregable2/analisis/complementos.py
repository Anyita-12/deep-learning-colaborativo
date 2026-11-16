"""Análisis complementarios del Entregable 2 sobre los resultados guardados (sin entrenar redes).

Uso, desde la raíz del repositorio:
    python entregable2/analisis/complementos.py            # todos los análisis
    python entregable2/analisis/complementos.py estadistica calibracion

Las salidas quedan en entregable2/results/complementario/. Las secciones marcadas como exploratorias
no figuran en el preregistro y no sustentan conclusiones confirmatorias.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from statsmodels.stats.multitest import multipletests

from comun import (ALPHA, CLASSES, DELTA, E1_MODELS, E2_MODELS, FULL, MACHINE, MODELS, OUT, RES, SEEDS, aligned,
                   fast_f1_macro, group_bootstrap_indices, groups_for, load_pickle, run_dir, save)

N_BOOT = 2000


# ----------------------------------------------------------------------------------------------
# 1. Inferencia complementaria entre modelos (Tabla C)
# ----------------------------------------------------------------------------------------------
def wide_f1() -> pd.DataFrame:
    return pd.read_csv(RES / "summary" / "f1_macro_por_semilla.csv").set_index("seed")[MODELS]


def bayes_signed_rank(d: np.ndarray, rope: float = DELTA, n_samples: int = 50_000, seed: int = 0) -> tuple:
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017) con previa de Dirichlet y región ROPE.

    Devuelve P(A peor), P(equivalente dentro de la ROPE) y P(A mejor) para d = A - B.
    """
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], d])  # pseudo-observación de la previa en 0
    n = len(z)
    w = rng.dirichlet(np.ones(n), n_samples)
    s = z[:, None] + z[None, :]
    left = (s < -2 * rope).astype(float)
    right = (s > 2 * rope).astype(float)
    mid = 1.0 - left - right
    pl = np.einsum("ki,ij,kj->k", w, left, w)
    pr = np.einsum("ki,ij,kj->k", w, right, w)
    pm = np.einsum("ki,ij,kj->k", w, mid, w)
    winner = np.argmax(np.stack([pl, pm, pr], 1), 1)
    return tuple(np.bincount(winner, minlength=3) / n_samples)


def tost_paired(d: np.ndarray, delta: float = DELTA) -> tuple[float, float, float]:
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    t1 = (m + delta) / se
    t2 = (m - delta) / se
    p1 = 1 - st.t.cdf(t1, n - 1)
    p2 = st.t.cdf(t2, n - 1)
    lo, hi = st.t.interval(0.90, n - 1, loc=m, scale=se)
    return max(p1, p2), lo, hi


def estadistica():
    w = wide_f1()
    n, k = w.shape
    # Iman-Davenport y diferencia crítica de Nemenyi
    fr = st.friedmanchisquare(*[w[c] for c in w.columns])
    q = fr.statistic
    ff = (n - 1) * q / (n * (k - 1) - q)
    p_id = 1 - st.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    q_alpha = 3.164  # Studentized range / sqrt(2), k = 10, alpha = 0.05 (Demšar, 2006)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    ranks = w.rank(axis=1, ascending=False).mean()
    glob = {"friedman_Q": q, "friedman_p_chi2": fr.pvalue, "iman_davenport_F": ff,
            "iman_davenport_gl": [k - 1, (k - 1) * (n - 1)], "iman_davenport_p": p_id,
            "nemenyi_q_alpha": q_alpha, "nemenyi_CD": cd, "rangos_medios": ranks.round(4).to_dict()}
    # Supuestos: normalidad por modelo y homocedasticidad
    sh = {m: st.shapiro(w[m]).pvalue for m in w.columns}
    lev = st.levene(*[w[c] for c in w.columns], center="median")
    glob["levene_W"] = lev.statistic
    glob["levene_p"] = lev.pvalue
    glob["shapiro_p_por_modelo"] = sh
    # ANOVA de medidas repetidas con corrección de Greenhouse-Geisser (complemento paramétrico de Friedman)
    X = w.to_numpy()
    gm = X.mean()
    ss_mod = n * ((X.mean(0) - gm) ** 2).sum()
    ss_blk = k * ((X.mean(1) - gm) ** 2).sum()
    ss_err = ((X - gm) ** 2).sum() - ss_mod - ss_blk
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f_rm = (ss_mod / df1) / (ss_err / df2)
    S = np.cov(X, rowvar=False)
    C = np.eye(k) - 1 / k
    Sc = C @ S @ C
    eps = np.trace(Sc) ** 2 / ((k - 1) * np.trace(Sc @ Sc))
    eps = min(1.0, eps)
    glob.update({"anova_rm_F": f_rm, "anova_rm_gl": [df1, df2], "gg_epsilon": eps,
                 "anova_rm_p_gg": 1 - st.f.cdf(f_rm, df1 * eps, df2 * eps),
                 "eta2_parcial": ss_mod / (ss_mod + ss_err)})
    with open(OUT / "estadistica_complementaria.json", "w", encoding="utf-8") as fh:
        json.dump(glob, fh, indent=2, ensure_ascii=False, default=float)

    # TOST y prueba bayesiana con ROPE para todos los pares (orden: A con mayor media)
    rows = []
    for a, b in itertools.combinations(w.columns, 2):
        if w[a].mean() < w[b].mean():
            a, b = b, a
        d = (w[a] - w[b]).to_numpy()
        p_tost, lo90, hi90 = tost_paired(d)
        pl, pm, pr = bayes_signed_rank(d)
        # Probabilidad de superioridad entre semillas (A12 de Vargha-Delaney sobre las 8 semillas)
        a12 = (np.sum(w[a].to_numpy()[:, None] > w[b].to_numpy()[None, :])
               + 0.5 * np.sum(w[a].to_numpy()[:, None] == w[b].to_numpy()[None, :])) / n ** 2
        rows.append({"A": a, "B": b, "dif_media": d.mean(), "ic90_lo": lo90, "ic90_hi": hi90, "p_tost": p_tost,
                     "equivalente_tost": p_tost < ALPHA, "p_bayes_A_peor": pl, "p_bayes_rope": pm,
                     "p_bayes_A_mejor": pr, "A12": a12})
    save(pd.DataFrame(rows), "equivalencia_tost_rope.csv")

    # Familia confirmatoria del E2: cada modelo del E2 frente al modelo de referencia del E1 (ResNet-18)
    pp = pd.read_csv(RES / "summary" / "pruebas_pares.csv")
    fam = []
    for m in E2_MODELS:
        r = pp[((pp.A == m) & (pp.B == "ResNet-18")) | ((pp.A == "ResNet-18") & (pp.B == m))].iloc[0]
        sign = 1 if r.A == m else -1
        fam.append({"modelo": m, "dif": sign * r.dif_media,
                    "ic_t_lo": sign * (r.ic_t_lo if sign == 1 else r.ic_t_hi),
                    "ic_t_hi": sign * (r.ic_t_hi if sign == 1 else r.ic_t_lo),
                    "ic_boot_lo": sign * (r.ic_boot_lo if sign == 1 else r.ic_boot_hi),
                    "ic_boot_hi": sign * (r.ic_boot_hi if sign == 1 else r.ic_boot_lo),
                    "d_z": sign * r.d_z, "p_t": r.p_t, "p_wilcoxon": r.p_wilcoxon,
                    "p_t_holm_45": r.p_t_holm, "shapiro_p": r.shapiro_p_dif})
    fam = pd.DataFrame(fam)
    fam["p_t_holm_7"] = multipletests(fam.p_t, method="holm")[1]
    fam["p_wilcoxon_holm_7"] = multipletests(fam.p_wilcoxon, method="holm")[1]
    fam["supera_umbral_0036"] = fam["dif"] >= 0.036
    sig = fam.p_t_holm_45 < ALPHA
    rel = fam.dif.abs() >= DELTA
    fam["decision"] = np.select([sig & rel & (fam.dif > 0), sig & rel & (fam.dif < 0), sig & ~rel],
                                ["superior: significativa y relevante", "inferior: significativa y relevante",
                                 "significativa sin relevancia"], "no concluyente")
    save(fam, "familia_vs_resnet18.csv")

    # Sensibilidad de la decisión por pares a la prueba elegida
    con = pd.read_csv(RES / "summary" / "posthoc_conover_holm.csv").set_index("model")
    nem = pd.read_csv(RES / "summary" / "posthoc_nemenyi.csv").set_index("model")
    sens = []
    for _, r in pp.iterrows():
        big = abs(r.dif_media) >= DELTA
        sens.append({"A": r.A, "B": r.B, "dif": r.dif_media,
                     "t_holm": bool(big and r.p_t_holm < ALPHA),
                     "wilcoxon_holm": bool(big and r.p_wilcoxon_holm < ALPHA),
                     "bootstrap_ic": bool(big and (r.ic_boot_lo > 0 or r.ic_boot_hi < 0)),
                     "conover_holm": bool(big and con.loc[r.A, r.B] < ALPHA),
                     "nemenyi": bool(big and nem.loc[r.A, r.B] < ALPHA)})
    save(pd.DataFrame(sens), "sensibilidad_decision.csv")


def por_imagen():
    """Q de Cochran por semilla y DeLong uno contra todos por clase (exploratorio)."""
    rows = []
    for s in SEEDS:
        Xc = np.stack([(aligned(m, s)["y_pred"] == aligned(m, s)["y_true"]).astype(int) for m in MODELS], 1)
        kk = Xc.shape[1]
        rs = Xc.sum(1)
        cs = Xc.sum(0)
        qn = (kk - 1) * (kk * (cs ** 2).sum() - cs.sum() ** 2)
        qd = kk * rs.sum() - (rs ** 2).sum()
        qc = qn / qd
        rows.append({"seed": s, "Q_cochran": qc, "gl": kk - 1, "p": 1 - st.chi2.cdf(qc, kk - 1)})
    save(pd.DataFrame(rows), "cochran_q.csv")

    def delong(y, pa, pb):
        """DeLong et al. (1988) para dos AUC correlacionadas (Sun y Xu, 2014)."""
        pos, neg = pa[y == 1], pa[y == 0]
        posb, negb = pb[y == 1], pb[y == 0]
        def comps(p, n):
            v10 = np.array([(np.sum(x > n) + 0.5 * np.sum(x == n)) / len(n) for x in p])
            v01 = np.array([(np.sum(p > x) + 0.5 * np.sum(p == x)) / len(p) for x in n])
            return v10, v01
        a10, a01 = comps(pos, neg)
        b10, b01 = comps(posb, negb)
        auc_a, auc_b = a10.mean(), b10.mean()
        s10 = np.cov(np.vstack([a10, b10]))
        s01 = np.cov(np.vstack([a01, b01]))
        S = s10 / len(pos) + s01 / len(neg)
        var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
        z = (auc_a - auc_b) / math.sqrt(var) if var > 0 else 0.0
        return auc_a, auc_b, 2 * (1 - st.norm.cdf(abs(z))), math.sqrt(max(var, 0))

    rows = []
    contrastes = [("RWCNet", "ViT-B16"), ("RWCNet", "GFN"), ("RWCNet", "Celik-HC"), ("ViT-B16", "GFN"),
                  ("Celik-HC", "ResNet-18"), ("RWCNet", "ResNet-18")]
    for a, b in contrastes:
        for s in SEEDS:
            da, db = aligned(a, s), aligned(b, s)
            ps = []
            for c in range(6):
                y = (da["y_true"] == c).astype(int)
                auc_a, auc_b, p, se = delong(y, da["probs"][:, c], db["probs"][:, c])
                ps.append(p)
                rows.append({"A": a, "B": b, "seed": s, "clase": CLASSES[c], "auc_A": auc_a, "auc_B": auc_b,
                             "dif": auc_a - auc_b, "se": se, "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby(["A", "B", "seed"])["p"].transform(lambda x: multipletests(x, method="holm")[1])
    save(df, "delong_ovr.csv")


# ----------------------------------------------------------------------------------------------
# 2. Bondad de ajuste (Tabla A) y sobreajuste (Tabla B)
# ----------------------------------------------------------------------------------------------
def bondad_ajuste():
    rel, grp = groups_for("test")
    boots = group_bootstrap_indices(grp, N_BOOT)
    rng = np.random.default_rng(7)
    p_major = None
    rows = []
    for m in MODELS:
        for s in SEEDS:
            d = aligned(m, s)
            y, yp, P = d["y_true"], d["y_pred"], d["probs"]
            if p_major is None:
                p_major = np.bincount(y, minlength=6).max() / len(y)
            hits = int((y == yp).sum())
            p_bin = st.binomtest(hits, len(y), p_major, alternative="greater").pvalue
            # Permutación condicional de etiquetas de prueba con las predicciones fijas
            f_obs = fast_f1_macro(y, yp)
            perm = np.array([fast_f1_macro(rng.permutation(y), yp) for _ in range(2000)])
            p_perm = (1 + (perm >= f_obs).sum()) / (len(perm) + 1)
            auc = roc_auc_score(y, P, multi_class="ovr", average="macro")
            mcc = matthews_corrcoef(y, yp)
            aucs, mccs = [], []
            for b in boots[:500]:
                yb = y[b]
                if len(np.unique(yb)) < 6:
                    continue
                aucs.append(roc_auc_score(yb, P[b], multi_class="ovr", average="macro"))
                mccs.append(matthews_corrcoef(yb, yp[b]))
            # Calibración top-label: regresión logística de acierto sobre logit(confianza)
            conf = np.clip(P.max(1), 1e-6, 1 - 1e-6)
            corr = (yp == y).astype(float)
            lg = np.log(conf / (1 - conf))
            slope, inter, p_lr = _calib_slope(lg, corr)
            zsp = _spiegelhalter(corr, conf)
            rows.append({"model": m, "seed": s, "accuracy": hits / len(y), "p_mayoritaria": p_major,
                         "p_binomial": p_bin, "f1_obs": f_obs, "f1_perm_media": perm.mean(), "p_permutacion": p_perm,
                         "auc_macro": auc, "auc_ic_lo": np.percentile(aucs, 2.5), "auc_ic_hi": np.percentile(aucs, 97.5),
                         "mcc": mcc, "mcc_ic_lo": np.percentile(mccs, 2.5), "mcc_ic_hi": np.percentile(mccs, 97.5),
                         "calib_pendiente": slope, "calib_intercepto": inter, "calib_p_lr": p_lr,
                         "spiegelhalter_z": zsp, "spiegelhalter_p": 2 * (1 - st.norm.cdf(abs(zsp)))})
    save(pd.DataFrame(rows), "bondad_ajuste.csv")


def _calib_slope(x, y):
    """Pendiente e intercepto de calibración (logística) y prueba de razón de verosimilitud de (0, 1)."""
    import statsmodels.api as sm
    X = sm.add_constant(x)
    try:
        fit = sm.Logit(y, X).fit(disp=0)
        ll_full = fit.llf
        p0 = 1 / (1 + np.exp(-x))
        ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        lr = 2 * (ll_full - ll0)
        return fit.params[1], fit.params[0], 1 - st.chi2.cdf(lr, 2)
    except Exception:
        return np.nan, np.nan, np.nan


def _spiegelhalter(y, p):
    num = np.sum((y - p) * (1 - 2 * p))
    den = math.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    return num / den if den > 0 else np.nan


def sobreajuste():
    rows = []
    for m in MODELS:
        for s in SEEDS:
            h = load_pickle(run_dir(m, s) / "history.pkl")
            vf = np.array(h["val_f1_macro"])
            b = int(np.argmax(vf))
            rows.append({"model": m, "seed": s, "best_epoch": b, "epochs_run": len(vf),
                         "train_f1": h["train_f1_macro"][b], "val_f1": vf[b],
                         "gap": h["train_f1_macro"][b] - vf[b]})
    g = pd.DataFrame(rows)
    save(g, "brecha_por_corrida.csv")
    dg = 0.05  # margen de sobreajuste (no preregistrado; valor de referencia del enunciado)
    out = []
    for m, gm in g.groupby("model", sort=False):
        x = gm.gap.to_numpy() - dg
        w = st.wilcoxon(x, alternative="less")
        t = st.ttest_1samp(x, 0, alternative="less")
        out.append({"model": m, "brecha_media": gm.gap.mean(), "brecha_de": gm.gap.std(ddof=1),
                    "delta_g": dg, "p_wilcoxon_unilateral": w.pvalue, "p_t_unilateral": t.pvalue,
                    "llega_tope_epocas": int((gm.epochs_run >= (60 if m in E2_MODELS else 100)).sum())})
    out = pd.DataFrame(out)
    out["p_wilcoxon_holm"] = multipletests(out.p_wilcoxon_unilateral, method="holm")[1]
    out["p_t_holm"] = multipletests(out.p_t_unilateral, method="holm")[1]
    save(out, "brecha_no_inferioridad.csv")

    # Validación frente a prueba: IC bootstrap por grupo de la diferencia s_val - s_test
    rel_t, grp_t = groups_for("test")
    rel_v, grp_v = groups_for("val")
    bt = group_bootstrap_indices(grp_t, N_BOOT, seed=11)
    bv = group_bootstrap_indices(grp_v, N_BOOT, seed=12)
    rows = []
    for m in MODELS:
        dv = [aligned(m, s, "val") for s in SEEDS]
        dt = [aligned(m, s, "test") for s in SEEDS]
        obs = np.mean([fast_f1_macro(v["y_true"], v["y_pred"]) - fast_f1_macro(t["y_true"], t["y_pred"])
                       for v, t in zip(dv, dt)])
        boots = []
        for i in range(N_BOOT):
            boots.append(np.mean([fast_f1_macro(v["y_true"][bv[i]], v["y_pred"][bv[i]])
                                  - fast_f1_macro(t["y_true"][bt[i]], t["y_pred"][bt[i]]) for v, t in zip(dv, dt)]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": m, "val_menos_test": obs, "ic_lo": lo, "ic_hi": hi, "significativa": lo > 0 or hi < 0})
    save(pd.DataFrame(rows), "validacion_vs_prueba.csv")

    w = wide_f1()
    cv = pd.DataFrame({"model": w.columns, "media": w.mean().values, "de": w.std(ddof=1).values,
                       "cv_pct": (100 * w.std(ddof=1) / w.mean()).values})
    save(cv, "sensibilidad_semilla.csv")


# ----------------------------------------------------------------------------------------------
# 3. Calibración e incertidumbre
# ----------------------------------------------------------------------------------------------
def _ece(conf, corr, bins=15, adaptive=False):
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    else:
        idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        mk = idx == b
        if mk.any():
            e += mk.mean() * abs(corr[mk].mean() - conf[mk].mean())
    return e


def calibracion():
    rows = []
    for m in MODELS:
        for s in SEEDS:
            d = aligned(m, s)
            P, y = d["probs"], d["y_true"]
            conf, corr = P.max(1), (P.argmax(1) == y).astype(float)
            cw = np.mean([_ece(P[:, c], (y == c).astype(float), 15) for c in range(6)])
            row = {"model": m, "seed": s, "ece15": _ece(conf, corr), "ece_adaptativo15": _ece(conf, corr, adaptive=True),
                   "ece_por_clase": cw, "brier": np.mean(np.sum((P - np.eye(6)[y]) ** 2, 1)),
                   "nll": -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))}
            cal = run_dir(m, s) / "calibration.pkl"
            if cal.exists():
                c = load_pickle(cal)
                if c.get("aplicable") and "probs_escaladas" in c:
                    d0 = load_pickle(run_dir(m, s) / "test_predictions.pkl")
                    from comun import relpath
                    o = np.argsort([relpath(f) for f in d0["filepaths"]])
                    Ps = c["probs_escaladas"][o]
                    cs, cr = Ps.max(1), (Ps.argmax(1) == y).astype(float)
                    row.update({"temperatura": c["temperature"], "ece15_ts": _ece(cs, cr),
                                "ece_adaptativo15_ts": _ece(cs, cr, adaptive=True)})
            rows.append(row)
    save(pd.DataFrame(rows), "calibracion_ampliada.csv")


def conforme():
    """Predicción conforme dividida (LAC y APS), calibrada en validación y evaluada en prueba."""
    rows, rows_c = [], []
    for m in MODELS:
        for s in SEEDS:
            v, t = aligned(m, s, "val"), aligned(m, s, "test")
            for alpha in (0.10, 0.05):
                n = len(v["y_true"])
                qlev = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
                # LAC: puntaje 1 - p_y
                sc = 1 - v["probs"][np.arange(n), v["y_true"]]
                qh = np.quantile(sc, qlev, method="higher")
                sets = (1 - t["probs"]) <= qh
                cov = sets[np.arange(len(t["y_true"])), t["y_true"]]
                # APS: masa acumulada hasta la clase verdadera (sin aleatorización)
                def aps_scores(P, y):
                    o = np.argsort(-P, 1)
                    cs = np.cumsum(np.take_along_axis(P, o, 1), 1)
                    rank = np.argmax(o == y[:, None], 1)
                    return cs[np.arange(len(y)), rank]
                qa = np.quantile(aps_scores(v["probs"], v["y_true"]), qlev, method="higher")
                o = np.argsort(-t["probs"], 1)
                cs = np.cumsum(np.take_along_axis(t["probs"], o, 1), 1)
                incl = np.zeros_like(cs, dtype=bool)
                incl[:, 0] = True
                incl[:, 1:] = cs[:, :-1] < qa
                sets_a = np.zeros_like(incl)
                np.put_along_axis(sets_a, o, incl, 1)
                cov_a = sets_a[np.arange(len(t["y_true"])), t["y_true"]]
                rows.append({"model": m, "seed": s, "alpha": alpha, "cobertura_lac": cov.mean(),
                             "tam_medio_lac": sets.sum(1).mean(), "vacios_lac": (sets.sum(1) == 0).mean(),
                             "cobertura_aps": cov_a.mean(), "tam_medio_aps": sets_a.sum(1).mean()})
                if alpha == 0.10:
                    for c in range(6):
                        mk = t["y_true"] == c
                        rows_c.append({"model": m, "seed": s, "clase": CLASSES[c], "cobertura_lac": cov[mk].mean(),
                                       "cobertura_aps": cov_a[mk].mean(), "tam_lac": sets[mk].sum(1).mean()})
    save(pd.DataFrame(rows), "conforme.csv")
    save(pd.DataFrame(rows_c), "conforme_por_clase.csv")


# ----------------------------------------------------------------------------------------------
# 4. Robustez
# ----------------------------------------------------------------------------------------------
def robustez():
    rel, grp = groups_for("test")
    boots = group_bootstrap_indices(grp, N_BOOT, seed=21)
    rows, rows_idx = [], []
    for m in MODELS:
        per_seed = []
        for s in SEEDS:
            r = load_pickle(run_dir(m, s) / "robustness.pkl")["results"]
            d0 = load_pickle(run_dir(m, s) / "test_predictions.pkl")
            from comun import relpath
            o = np.argsort([relpath(f) for f in d0["filepaths"]])
            y = np.asarray(d0["y_true"])[o]
            if "y_pred" in r["limpio"]:
                per_seed.append((y, {k: np.asarray(v["y_pred"])[o] for k, v in r.items()}))
                f = {k: fast_f1_macro(y, v) for k, v in per_seed[-1][1].items()}
            else:  # las corridas del E1 guardaron solo el F1 por perturbación (sin predicciones por imagen)
                f = {k: v["f1_macro"] for k, v in r.items()}
            # Índice de robustez: área bajo la curva de degradación normalizada por el F1 limpio
            ir_ruido = np.trapezoid([f["limpio"], f["ruido_0.05"], f["ruido_0.10"]], [0, 0.05, 0.10]) / (0.10 * f["limpio"])
            ir_brillo = np.trapezoid([f["brillo_-30"], f["limpio"], f["brillo_+30"]], [-30, 0, 30]) / (60 * f["limpio"])
            rows_idx.append({"model": m, "seed": s, "ir_ruido": ir_ruido, "ir_brillo": ir_brillo,
                             "ir_global": np.mean([f[k] / f["limpio"] for k in f if k != "limpio"])})
        # IC bootstrap por grupo de la caída por ruido 0.05 y 0.10 (promedio de semillas)
        for pert in ([] if not per_seed else ("ruido_0.05", "ruido_0.10", "desenfoque", "fondo_sustituido")):
            obs = np.mean([fast_f1_macro(y, pr["limpio"]) - fast_f1_macro(y, pr[pert]) for y, pr in per_seed])
            bs = [np.mean([fast_f1_macro(y[b], pr["limpio"][b]) - fast_f1_macro(y[b], pr[pert][b]) for y, pr in per_seed])
                  for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"model": m, "perturbacion": pert, "caida_f1": obs, "ic_lo": lo, "ic_hi": hi})
    save(pd.DataFrame(rows), "robustez_caidas_ic.csv")
    save(pd.DataFrame(rows_idx), "robustez_indice.csv")


# ----------------------------------------------------------------------------------------------
# 5. Errores compartidos, techo de desempeño y ensamble (exploratorio)
# ----------------------------------------------------------------------------------------------
def errores_techo():
    rel, grp = groups_for("test")
    y = aligned(MODELS[0], 0)["y_true"]
    E = {}
    P = {}
    for m in MODELS:
        E[m] = np.stack([aligned(m, s)["y_pred"] != y for s in SEEDS])
        P[m] = np.stack([aligned(m, s)["y_pred"] for s in SEEDS])
    allp = np.concatenate([P[m] for m in MODELS])          # 80 x 361
    err_rate = (allp != y).mean(0)
    # Etiqueta de consenso entre las 80 corridas
    cons = np.array([np.bincount(allp[:, i], minlength=6).argmax() for i in range(len(y))])
    cons_frac = np.array([np.bincount(allp[:, i], minlength=6).max() / allp.shape[0] for i in range(len(y))])
    img = pd.DataFrame({"relpath": rel, "grupo": grp, "clase": [CLASSES[c] for c in y],
                        "tasa_error_80": err_rate, "consenso": [CLASSES[c] for c in cons],
                        "frac_consenso": cons_frac,
                        "tasa_error_top3": np.concatenate([E[m] for m in ("RWCNet", "ViT-B16", "GFN")]).mean(0)})
    save(img.sort_values("tasa_error_80", ascending=False), "errores_por_imagen.csv")

    # Correlación de errores entre modelos (media sobre semillas) y Jaccard de los conjuntos de error
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        cors, jac = [], []
        for s in SEEDS:
            ea, eb = E[a][s].astype(float), E[b][s].astype(float)
            if ea.std() > 0 and eb.std() > 0:
                cors.append(np.corrcoef(ea, eb)[0, 1])
            u = np.logical_or(ea, eb).sum()
            jac.append(np.logical_and(ea, eb).sum() / u if u else np.nan)
        rows.append({"A": a, "B": b, "corr_errores": np.nanmean(cors), "jaccard_errores": np.nanmean(jac)})
    save(pd.DataFrame(rows), "errores_compartidos.csv")

    # Ensamble por voto mayoritario de los 10 modelos y de los 3 mejores, por semilla
    rows = []
    for s in SEEDS:
        for nombre, ms in (("10 modelos", MODELS), ("RWCNet+ViT-B16+GFN", ["RWCNet", "ViT-B16", "GFN"])):
            prob = np.mean([aligned(m, s)["probs"] for m in ms], 0)
            rows.append({"seed": s, "ensamble": nombre, "f1_macro": fast_f1_macro(y, prob.argmax(1))})
        orac = np.any(np.stack([~E[m][s] for m in MODELS]), 0)
        rows.append({"seed": s, "ensamble": "oráculo (alguno acierta)", "f1_macro": orac.mean()})
    save(pd.DataFrame(rows), "ensambles.csv")


# ----------------------------------------------------------------------------------------------
# 6. Estabilidad de Grad-CAM entre semillas (exploratorio)
# ----------------------------------------------------------------------------------------------
def gradcam_estabilidad():
    rows = []
    for m in E2_MODELS:
        maps, keys = [], None
        for s in SEEDS:
            f = FULL / m / f"seed_{s:02d}" / "gradcam.npz"
            if not f.exists():
                break
            z = np.load(f, allow_pickle=True)
            o = np.argsort(z["rows"])
            maps.append(z["cam_true"][o].reshape(len(o), -1))
            keys = z["rows"][o] if keys is None else keys
        if len(maps) < 2:
            continue
        M = np.stack(maps)                                   # semillas x imágenes x píxeles
        for i in range(M.shape[1]):
            rho, iou = [], []
            for a, b in itertools.combinations(range(M.shape[0]), 2):
                xa, xb = M[a, i], M[b, i]
                if xa.std() == 0 or xb.std() == 0:
                    continue
                rho.append(st.spearmanr(xa, xb).statistic)
                ta, tb = xa >= np.quantile(xa, 0.8), xb >= np.quantile(xb, 0.8)
                iou.append((ta & tb).sum() / max(1, (ta | tb).sum()))
            rows.append({"model": m, "imagen": int(keys[i]), "spearman_medio": np.mean(rho) if rho else np.nan,
                         "iou_top20_medio": np.mean(iou) if iou else np.nan})
    save(pd.DataFrame(rows), "gradcam_estabilidad.csv")


# ----------------------------------------------------------------------------------------------
# 7. Eficiencia: Pareto, costo por unidad de mejora y energía estimada
# ----------------------------------------------------------------------------------------------
def eficiencia():
    mc = pd.read_csv(RES / "summary" / "metricas_por_corrida.csv")
    g = mc.groupby("model").agg(f1=("f1_macro", "mean"), params=("params", "first"), gmacs=("gmacs", "first"),
                                ms_img=("ms_per_image", "mean"), train_s=("train_time_s", "mean")).reindex(MODELS)
    g["maquina"] = [MACHINE[m] for m in g.index]
    for c in ("params", "gmacs"):
        nd = []
        for m in g.index:
            dom = ((g.f1 >= g.loc[m, "f1"]) & (g[c] <= g.loc[m, c]) & ((g.f1 > g.loc[m, "f1"]) | (g[c] < g.loc[m, c]))).any()
            nd.append(not dom)
        g[f"pareto_{c}"] = nd
    # Latencia: frente de Pareto solo dentro de cada máquina
    g["pareto_latencia"] = False
    for mq, gg in g.groupby("maquina"):
        for m in gg.index:
            dom = ((gg.f1 >= gg.loc[m, "f1"]) & (gg.ms_img <= gg.loc[m, "ms_img"])
                   & ((gg.f1 > gg.loc[m, "f1"]) | (gg.ms_img < gg.loc[m, "ms_img"]))).any()
            g.loc[m, "pareto_latencia"] = not dom
    # Costo por unidad de mejora frente a ResNet-18 (GMAC adicionales por punto de F1), IC bootstrap por semilla
    w = wide_f1()
    rng = np.random.default_rng(3)
    eta, lo, hi = [], [], []
    for m in g.index:
        dc = g.loc[m, "gmacs"] - g.loc["ResNet-18", "gmacs"]
        d = (w[m] - w["ResNet-18"]).to_numpy()
        e = d.mean() / dc if dc != 0 else np.nan
        bs = [d[rng.integers(0, 8, 8)].mean() / dc for _ in range(2000)] if dc != 0 else [np.nan]
        eta.append(e)
        lo.append(np.nanpercentile(bs, 2.5))
        hi.append(np.nanpercentile(bs, 97.5))
    g["eta_f1_por_gmac"] = eta
    g["eta_ic_lo"] = lo
    g["eta_ic_hi"] = hi
    save(g.reset_index(), "eficiencia_pareto.csv")

    # Energía aproximada: potencia nominal de la GPU x tiempo de entrenamiento (búsqueda + 8 semillas)
    tdp = {"A100": 400.0, "RTX 4060": 75.0}
    hp = pd.read_csv(RES / "summary" / "busqueda_hiperparametros.csv").groupby("model").train_time_s.sum()
    rows = []
    for m in MODELS:
        t_final = mc[mc.model == m].train_time_s.sum()
        t_hpo = hp.get(m, np.nan)
        kwh = tdp[MACHINE[m]] * (t_final + (0 if np.isnan(t_hpo) else t_hpo)) / 3.6e6
        rows.append({"model": m, "maquina": MACHINE[m], "tdp_W": tdp[MACHINE[m]], "horas_finales": t_final / 3600,
                     "horas_busqueda": t_hpo / 3600 if not np.isnan(t_hpo) else np.nan, "kwh_estimado": kwh})
    save(pd.DataFrame(rows), "energia_estimada.csv")


# ----------------------------------------------------------------------------------------------
# 8. Ablaciones: Celik-HC (5 variantes) y RWCNet (sin salidas auxiliares)
# ----------------------------------------------------------------------------------------------
def ablaciones():
    w = wide_f1()
    rows = []
    for base in ("Celik-HC", "RWCNet"):
        root = RES / "ablation" / base
        if not root.exists():
            continue
        for var in sorted(p.name for p in root.iterdir() if p.is_dir()):
            vals = []
            for s in SEEDS:
                f = root / var / f"seed_{s:02d}" / "metrics_test.pkl"
                if (root / var / f"seed_{s:02d}" / "DONE").exists():
                    vals.append((s, load_pickle(f)["f1_macro"], load_pickle(f)["ece"]))
            if len(vals) < len(SEEDS):
                continue
            v = np.array([x[1] for x in vals])
            d = w[base].to_numpy() - v                       # completo - variante, pareado por semilla
            t = st.ttest_1samp(d, 0)
            p_tost, lo90, hi90 = tost_paired(d)
            h = st.t.ppf(0.975, 7) * d.std(ddof=1) / math.sqrt(8)
            rows.append({"base": base, "variante": f"{base}/{var}", "f1_variante": v.mean(), "f1_de": v.std(ddof=1),
                         "f1_completo": w[base].mean(), "ece_variante": np.mean([x[2] for x in vals]),
                         "dif_media": d.mean(), "ic_t_lo": d.mean() - h, "ic_t_hi": d.mean() + h, "d_z": d.mean() / d.std(ddof=1),
                         "p_t": t.pvalue, "p_wilcoxon": st.wilcoxon(d).pvalue, "p_tost": p_tost,
                         "ic90_lo": lo90, "ic90_hi": hi90})
    df = pd.DataFrame(rows)
    df["p_t_holm_familia"] = df.groupby("base").p_t.transform(lambda x: multipletests(x, method="holm")[1])
    df["decision"] = np.where((df.p_t_holm_familia < ALPHA) & (df.dif_media.abs() >= DELTA), "significativa y relevante",
                              np.where(df.p_tost < ALPHA, "equivalente (TOST ±δ)", "no concluyente"))
    save(df, "equivalencia_ablacion.csv")
    print(df[["variante", "f1_variante", "dif_media", "p_t_holm_familia", "p_tost", "decision"]].round(4))


SECCIONES = {"ablaciones": ablaciones,"estadistica": estadistica, "por_imagen": por_imagen, "bondad": bondad_ajuste,
             "sobreajuste": sobreajuste, "calibracion": calibracion, "conforme": conforme, "robustez": robustez,
             "errores": errores_techo, "gradcam": gradcam_estabilidad, "eficiencia": eficiencia}

if __name__ == "__main__":
    pedidas = sys.argv[1:] or list(SECCIONES)
    for nombre in pedidas:
        t0 = time.time()
        SECCIONES[nombre]()
        print(f"{nombre}: {time.time() - t0:.1f} s")
