"""Análisis confirmatorio del E3 (protocolo.md, secciones 4 y 5)."""
import pickle, json, numpy as np, pandas as pd
from scipy import stats
from cargar import dirs, lp, REPO
import estad as E
DELTA = 0.02
RIVALES = ["Celik-HC", "MbXception", "ViT-B16", "GFN", "FocusRCNet", "ResMsCapsule", "RWCNet",
           "CNN", "MobileNetV2", "ResNet-18", "HistGradientBoosting", "Trivial"]
df = pd.read_pickle("maestra.pkl")
sp = pickle.load(open(f"{REPO}/entregable2/results/common/splits.pkl", "rb"))["test"]
Y, GR = np.asarray(sp["y"]), np.asarray(sp["group"])
bl = pd.read_csv(f"{REPO}/entregable2/results/complementario/baselines.csv")
CONST = {"HistGradientBoosting": bl[bl.model.str.startswith("Hist")].iloc[0],
         "Trivial": bl[bl.model.str.startswith("Trivial")].iloc[0]}

def ypred(model, seed, cond):
    d = dirs(model)[seed]
    if cond == "limpio":
        return np.asarray(lp(f"{d}/test_predictions.pkl")["y_pred"])
    r = lp(f"{d}/robustness.pkl")["results"][cond]
    return np.asarray(r["y_pred"]) if "y_pred" in r else None

def serie(model, col):
    return df[df.model == model].set_index("seed")[col].sort_index()

def contraste(a, b, col, cond, B=10000, seeds_max=None):
    sa = serie(a, col)
    if b in CONST:
        c = float(CONST[b]["f1_macro" if col in ("f1", "rob_ruido_0.05") else col]) if col != "mcc" else float(CONST[b]["mcc"])
        if col == "rob_ruido_0.05":
            c = np.nan  # los baselines deterministas no tienen evaluación con ruido en el E2
        seeds = list(sa.index); d = (sa - c).values; cb = c
    else:
        sb = serie(b, col); seeds = sorted(set(sa.index) & set(sb.index)); cb = None
        d = (sa[seeds] - sb[seeds]).values
    if seeds_max is not None:
        keep = [i for i, s in enumerate(seeds) if s <= seeds_max]; seeds = [seeds[i] for i in keep]; d = d[keep]
    if np.isnan(d).any():
        return None
    YA = [ypred(a, s, cond) for s in seeds]
    YB = None if cb is not None else [ypred(b, s, cond) for s in seeds]
    if col in ("f1", "rob_ruido_0.05") and all(x is not None for x in YA) and (YB is None or all(x is not None for x in YB)):
        dm, pb, ic = E.boot_jer(np.stack(YA), None if YB is None else np.stack(YB), Y, GR, B=B, const_b=cb)
        tipo = "jerárquico"
    else:
        dm, pb, ic = E.boot_semillas(d, B=B); tipo = "semillas"
    return dict(rival=b, R=len(seeds), semillas=f"{seeds[0]}-{seeds[-1]}", media_dif=d.mean(), de_dif=d.std(ddof=1),
                ic_lo=ic[0], ic_hi=ic[1], p_t=E.t_pair(d), p_w=E.wilc(d), p_boot=pb, boot=tipo,
                p_nb=E.t_nb(d), cliff=E.cliff(d), dz=E.dz(d), media_nag=sa[seeds].mean())

def familia(col, cond, **kw):
    rows = [r for r in (contraste("NAGFuse", b, col, cond, **kw) for b in RIVALES) if r]
    t = pd.DataFrame(rows)
    for k in ["p_t", "p_w", "p_boot", "p_nb"]:
        t[k + "_holm"] = E.holm(t[k])
    t["p_max_holm"] = t[["p_t_holm", "p_w_holm", "p_boot_holm"]].max(1)
    t["supera"] = (t.p_max_holm < 0.05) & (t.media_dif > DELTA) & (t.ic_lo > 0)
    t["mejor_sig"] = (t.p_max_holm < 0.05) & (t.ic_lo > 0)
    return t

if __name__ == "__main__":
    out = {}
    out["H1_f1"] = familia("f1", "limpio")
    out["H1_mcc"] = familia("mcc", "limpio")
    out["H1_seeds0-9"] = familia("f1", "limpio", seeds_max=9)
    out["H2_ruido005"] = familia("rob_ruido_0.05", "ruido_0.05")
    pd.to_pickle(out, "confirmatorio.pkl")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    c = ["rival", "R", "media_dif", "ic_lo", "ic_hi", "p_t_holm", "p_w_holm", "p_boot_holm", "p_nb_holm", "cliff", "dz", "supera", "mejor_sig", "boot"]
    for k, t in out.items():
        print("\n==", k); print(t[c].round(4).to_string(index=False))
