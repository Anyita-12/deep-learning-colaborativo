"""Carga todas las corridas (E1, E2, E3) en una tabla maestra por (modelo, semilla)."""
import pickle, os, glob, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
NEW = f"{REPO}/entregable3/results"
E1M = ["CNN", "MobileNetV2", "ResNet-18"]
E2M = ["Celik-HC", "MbXception", "ViT-B16", "GFN", "FocusRCNet", "ResMsCapsule", "RWCNet"]
ABL = ["sin_denoising", "solo_vit", "solo_cnn"]

def dirs(model):
    if model == "NAGFuse":
        roots = [f"{NEW}/results_e2/NAGFuse"]
    elif model.startswith("abl:"):
        roots = [f"{NEW}/results_e2/ablation/NAGFuse/{model[4:]}"]
    elif model in E1M:
        roots = [f"{REPO}/results/{model}", f"{NEW}/results_e1/{model}"]
    else:
        roots = [f"{REPO}/entregable2/results/{model}", f"{NEW}/results_e2/{model}"]
    out = {}
    for r in roots:
        for d in sorted(glob.glob(f"{r}/seed_*")):
            if os.path.exists(f"{d}/metrics_test.pkl"):
                out[int(d[-2:])] = d
    return out

def lp(p):
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

def cargar():
    rows, preds = [], {}
    for m in ["NAGFuse"] + E2M + E1M + [f"abl:{a}" for a in ABL]:
        for s, d in dirs(m).items():
            mt = lp(f"{d}/metrics_test.pkl"); tp = lp(f"{d}/test_predictions.pkl")
            rb = lp(f"{d}/robustness.pkl"); ex = lp(f"{d}/extra_eval.pkl")
            r = dict(model=m, seed=s, f1=mt["f1_macro"], mcc=mt["mcc"], acc=mt["accuracy"],
                     auc=mt.get("auc_roc_macro"), ece=mt.get("ece"), nll=mt.get("nll"), brier=mt.get("brier"))
            if rb:
                for k, v in rb["results"].items():
                    r[f"rob_{k}"] = v.get("f1_macro") if isinstance(v, dict) else np.nan
            if ex:
                for k, v in ex.items():
                    r[f"ext_{k}"] = v["f1_macro"]
            rows.append(r)
            preds[(m, s)] = tp
    return pd.DataFrame(rows), preds

if __name__ == "__main__":
    df, preds = cargar()
    df.to_pickle("maestra.pkl"); pickle.dump(preds, open("preds.pkl", "wb"))
    print(df.groupby("model").seed.agg(["count", "min", "max"]))
    print(df.columns.tolist())
