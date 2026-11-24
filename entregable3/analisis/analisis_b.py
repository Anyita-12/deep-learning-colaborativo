"""Opción B (protocolo v4, experiments/protocolo_v4_opcion_B.md): contrastes de NAGFuse-B contra la misma familia.

Ejecutar desde entregable3/analisis después de `python cargar.py` (que ya incluye NAGFuse-B):
    python analisis_b.py
Escribe entregable3/resultados_b/*.csv y resumen_b.json. No modifica los resultados de la versión A.

Reglas fijadas antes de entrenar (protocolo v4):
- Misma familia de 12 rivales, mismas tres pruebas, Holm (m = 12) y la decisión por el máximo p ajustado.
- alfa_B = 0.025: Bonferroni por haber hecho dos intentos (A y B) sobre la misma familia y el mismo conjunto de prueba.
  También se informa el veredicto con alfa = 0.05, como dato secundario.
- "Supera" = p máx. de Holm < alfa_B, diferencia media > delta = 0.02 y límite inferior del IC 95 % > 0.
- R3 se cumple solo si NAGFuse-B supera a los 12 rivales en H1 (F1 macro limpio).
- B contra A, H3, conjuntos adicionales, Friedman y el análisis del mecanismo son secundarios o exploratorios.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from cargar import REPO, dirs, lp
import confirmatorio as CF            # reutiliza contraste(), serie(), RIVALES y DELTA (lee maestra.pkl)
import estad as E

PROPIO, ALFA_B = "NAGFuse-B", 0.025
OUT = Path(REPO) / "entregable3" / "resultados_b"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)


def familia(col, cond, **kw):
    rows = [r for r in (CF.contraste(PROPIO, b, col, cond, **kw) for b in CF.RIVALES) if r]
    t = pd.DataFrame(rows)
    for k in ["p_t", "p_w", "p_boot", "p_nb"]:
        t[k + "_holm"] = E.holm(t[k])
    t["p_max_holm"] = t[["p_t_holm", "p_w_holm", "p_boot_holm"]].max(1)
    t["supera"] = (t.p_max_holm < ALFA_B) & (t.media_dif > CF.DELTA) & (t.ic_lo > 0)
    t["supera_alfa_005"] = (t.p_max_holm < 0.05) & (t.media_dif > CF.DELTA) & (t.ic_lo > 0)
    t["mejor_sig"] = (t.p_max_holm < ALFA_B) & (t.ic_lo > 0)
    return t


if __name__ == "__main__":
    df = CF.df
    semillas = sorted(df[df.model == PROPIO].seed)
    if len(semillas) < 12:
        raise SystemExit(f"NAGFuse-B tiene {len(semillas)} semillas terminadas; el protocolo v4 exige al menos 12.")
    print(f"NAGFuse-B: {len(semillas)} semillas ({semillas[0]}-{semillas[-1]})")
    out = {"H1_f1": familia("f1", "limpio"), "H1_mcc": familia("mcc", "limpio"),
           "H1_seeds0-9": familia("f1", "limpio", seeds_max=9), "H2_ruido005": familia("rob_ruido_0.05", "ruido_0.05")}
    for k, t in out.items():
        t.to_csv(OUT / f"contrastes_{k}.csv", index=False)
    c = ["rival", "R", "media_dif", "ic_lo", "ic_hi", "p_max_holm", "cliff", "dz", "supera", "supera_alfa_005"]
    for k, t in out.items():
        print(f"\n== {k}"); print(t[c].round(4).to_string(index=False))

    # ---------- B contra A (exploratorio: dos intentos del mismo equipo) ----------
    ba = CF.contraste(PROPIO, "NAGFuse", "f1", "limpio")
    ba05 = CF.contraste(PROPIO, "NAGFuse", "rob_ruido_0.05", "ruido_0.05")
    pd.DataFrame([dict(cond="limpio", **ba), dict(cond="ruido_0.05", **ba05)]).to_csv(OUT / "B_contra_A.csv", index=False)

    # ---------- H3: no inferioridad / equivalencia (TOST) frente al mejor rival ----------
    med = df[df.model.isin(CF.RIVALES)].groupby("model").f1.mean().sort_values(); mejor = med.index[-1]
    a, b = CF.serie(PROPIO, "f1"), CF.serie(mejor, "f1"); s = sorted(set(a.index) & set(b.index)); d = (a[s] - b[s]).values
    se = d.std(ddof=1) / np.sqrt(len(d))
    p_inf = stats.t.sf((d.mean() + CF.DELTA) / se, len(d) - 1); p_sup = stats.t.cdf((d.mean() - CF.DELTA) / se, len(d) - 1)
    h3 = dict(mejor_rival=mejor, R=len(d), dif=d.mean(), p_no_inferioridad=p_inf, p_tost=max(p_inf, p_sup))
    pd.DataFrame([h3]).to_csv(OUT / "H3_no_inferioridad.csv", index=False)

    # ---------- Conjuntos adicionales (secundario) ----------
    ext = []
    for ds in ["realwaste", "gd_v2", "warp_c"]:
        col = f"ext_{ds}"
        if col not in df:
            continue
        rr = []
        for r in CF.RIVALES[:10] + ["NAGFuse"]:
            x, y = CF.serie(PROPIO, col).dropna(), CF.serie(r, col).dropna(); s = sorted(set(x.index) & set(y.index))
            if len(s) < 2:
                continue
            dd = (x[s] - y[s]).values
            rr.append(dict(ds=ds, rival=r, R=len(s), propio=x[s].mean(), riv=y[s].mean(), dif=dd.mean(),
                           ic=tuple(np.round(E.boot_semillas(dd)[2], 4)), p_t=E.t_pair(dd), p_w=E.wilc(dd)))
        t = pd.DataFrame(rr); t["p_max_holm"] = np.maximum(E.holm(t.p_t), E.holm(t.p_w)); ext.append(t)
    if ext:
        pd.concat(ext).to_csv(OUT / "contrastes_extra.csv", index=False)

    # ---------- Friedman + Iman-Davenport + Nemenyi con B en lugar de A (11 modelos, semillas 0-11) ----------
    S = list(range(12))
    M = [PROPIO] + CF.RIVALES[:10]
    X = np.stack([CF.serie(m, "f1")[S].values for m in M], 1)
    N, k = X.shape; R_ = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, X).mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R_ ** 2) - k * (k + 1) ** 2 / 4)
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2); pF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
    CD = 3.219 * np.sqrt(k * (k + 1) / (6 * N))                     # Nemenyi, alfa 0.05, k = 11 (Demšar, 2006)
    pd.DataFrame([{"modelo": m, "rango_medio": r} for m, r in zip(M, R_)]).assign(
        chi2=chi2, F_ID=FF, p=pF, CD=CD).to_csv(OUT / "friedman_nemenyi.csv", index=False)

    # ---------- Mecanismo (exploratorio): compuerta apagada frente a encendida con ruido 0.05 ----------
    mec = []
    for sd, d_ in dirs(PROPIO).items():
        an = lp(f"{d_}/nagfuse_analisis.pkl")
        if an:
            mec.append(dict(semilla=sd, pi0=an["pi0"], f1_ruido005_off=an["robustez"]["compuerta_off"]["ruido_0.05"]["f1_macro"],
                            f1_ruido005_on=float(CF.serie(PROPIO, "rob_ruido_0.05").get(sd, np.nan)),
                            frac_pi1_ruido005=float((an["detalles"]["ruido_0.05"]["pi"] >= 0.999).mean())))
    pd.DataFrame(mec).to_csv(OUT / "mecanismo.csv", index=False)

    # ---------- Veredicto ----------
    h1 = out["H1_f1"]
    res = {"semillas": len(semillas), "alfa_B": ALFA_B, "delta": CF.DELTA,
           "H1_supera_a": int(h1.supera.sum()), "H1_de": int(len(h1)),
           "R3_cumple": bool(h1.supera.all() and len(h1) == 12),
           "R3_cumple_con_alfa_005": bool(h1.supera_alfa_005.all() and len(h1) == 12),
           "no_supera": h1.loc[~h1.supera, "rival"].tolist(),
           "F1_limpio_B": float(CF.serie(PROPIO, "f1").mean()), "F1_limpio_A": float(CF.serie("NAGFuse", "f1").mean()),
           "B_menos_A": {"dif": float(ba["media_dif"]), "ic": [float(ba["ic_lo"]), float(ba["ic_hi"])]},
           "H2_supera_a": int(out["H2_ruido005"].supera.sum()), "H2_de": int(len(out["H2_ruido005"])),
           "H3": {k: (float(v) if not isinstance(v, str) else v) for k, v in h3.items()}}
    json.dump(res, open(OUT / "resumen_b.json", "w"), indent=2, ensure_ascii=False)
    print("\n== VEREDICTO"); print(json.dumps(res, indent=2, ensure_ascii=False))
