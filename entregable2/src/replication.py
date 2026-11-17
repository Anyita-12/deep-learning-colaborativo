"""Replicación parcial del paper guía (Celik, 2025).

Protocolo original (repositorio oficial): las 2527 imágenes, partición aleatoria 80/20 con
random_state = 32, sin aumentación, Adam (lr 1e-3), lote 128, 40 épocas, selección del
checkpoint por accuracy sobre el mismo 20 % que se reporta como prueba, y XGBoost
(random_state = 101) sobre el embedding de 64 dimensiones.

Protocolo estricto: misma partición, pero la época se elige con un 10 % de validación
interno tomado del 80 % de entrenamiento; el 20 % se evalúa una sola vez.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from . import config as C
from .data import CachedImages, build_full_cache, eval_transform, make_loader, train_transform
from .evaluate import compute_metrics, env_info, predict
from .models import build_model, model_info
from .persist import is_done, save_csv, save_json, save_pickle, touch_done, backup_tree
from .seeds import configure_determinism, set_seed

PROTOCOLS = ("original", "estricto")


def replication_split(df: pd.DataFrame, seed: int) -> dict:
    R = C.REPLICATION
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=R["test_size"], random_state=R["random_state"])
    if C.SMOKE:
        tr, te = tr[:96], te[:48]
    tr_in, va_in = train_test_split(tr, test_size=R["inner_val_fraction"], random_state=seed,
                                    stratify=None if C.SMOKE else df["label_id"].to_numpy()[tr])
    return {"train": np.sort(tr), "test": np.sort(te), "train_inner": np.sort(tr_in), "val_inner": np.sort(va_in)}


def leakage_audit(df: pd.DataFrame, split: dict) -> dict:
    """Imágenes del 20 % de prueba cuyo grupo de casi duplicados (EDA del Entregable 1) tiene miembros en el 80 %."""
    groups = pd.read_csv(C.SPLIT_CSV)[["relpath", "group"]]
    g = df.merge(groups, on="relpath", how="left")["group"].to_numpy()
    tr_groups = set(g[split["train"]][~pd.isna(g[split["train"]])])
    te = split["test"]
    shared = np.array([not pd.isna(g[i]) and g[i] in tr_groups for i in te])
    dup_groups = pd.Series(g[~pd.isna(g)]).value_counts()
    multi = set(dup_groups[dup_groups > 1].index)
    in_multi = np.array([not pd.isna(g[i]) and g[i] in multi for i in te])
    leaked = shared & in_multi
    return {"n_test": int(len(te)), "test_con_casi_duplicado_en_train": int(leaked.sum()),
            "fraccion": float(leaked.mean()), "excluidas_en_v2_presentes": int(pd.isna(g[te]).sum())}


def _train(seed: int, protocol: str, cache: str, df: pd.DataFrame, split: dict, dev, log) -> dict:
    R = C.REPLICATION
    set_seed(seed)
    model = build_model("Celik-HC", dropout=0.0).to(dev)
    y = df["label_id"].to_numpy()
    tr = split["train"] if protocol == "original" else split["train_inner"]
    sel = split["test"] if protocol == "original" else split["val_inner"]
    mean, std = C.IMAGENET_MEAN, C.IMAGENET_STD
    micro = max(1, R["batch_size"] // C.ACCUM_STEPS)
    g = torch.Generator().manual_seed(seed)
    ds_tr = CachedImages(cache, tr, y[tr], train_transform(mean, std, augment=False))
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=micro, shuffle=True, generator=g, num_workers=0,
                                        pin_memory=dev.type == "cuda")
    ev = lambda rows: make_loader(CachedImages(cache, rows, y[rows], eval_transform(mean, std)), 128, num_workers=0)
    dl_sel, dl_test, dl_tr_eval = ev(sel), ev(split["test"]), ev(tr)
    opt = torch.optim.Adam(model.parameters(), lr=R["lr"])
    crit = torch.nn.CrossEntropyLoss()
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_acc, best_state, best_epoch, hist = -1.0, None, -1, []
    t_start = time.perf_counter()
    for epoch in range(R["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, (x, yb, _) in enumerate(dl_tr):
            x, yb = x.to(dev), yb.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x).float(), yb) / C.ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % C.ACCUM_STEPS == 0 or i + 1 == len(dl_tr):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        p = predict(model, dl_sel, dev, amp=use_amp)
        acc = float(accuracy_score(p["y_true"], p["y_pred"]))
        hist.append({"epoch": epoch, "sel_accuracy": acc})
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  replicación {protocol} s{seed} ep {epoch:2d} | accuracy de selección {acc:.4f} (mejor {best_acc:.4f})")
    model.load_state_dict(best_state)
    from xgboost import XGBClassifier
    f_tr = predict(model, dl_tr_eval, dev, amp=use_amp)
    xgb = XGBClassifier(random_state=R["xgb_random_state"], n_jobs=-1, tree_method="hist")
    xgb.fit(f_tr["embedding"], f_tr["y_true"])
    pt = predict(model, dl_test, dev, amp=use_amp, xgb=xgb)
    yt, yp = pt["y_true"], pt["y_pred"]
    summary = {"protocol": protocol, "seed": seed, "best_epoch": best_epoch, "selection_accuracy": best_acc,
               "accuracy": float(accuracy_score(yt, yp)),
               "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
               "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "accuracy_head": float(accuracy_score(yt, pt["probs_head"].argmax(1))),
               "train_time_s": time.perf_counter() - t_start}
    rng = np.random.default_rng(seed)
    boots = [accuracy_score(yt[ix], yp[ix]) for ix in rng.integers(0, len(yt), (C.N_BOOT // 5, len(yt)))]
    summary["accuracy_ic_boot"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"summary": summary, "history": hist, "test_predictions": pt, "metrics": compute_metrics(yt, pt["probs"])}


def run_replication(log=print) -> pd.DataFrame:
    out_root = C.RESULTS / "replication"
    out_root.mkdir(parents=True, exist_ok=True)
    det = configure_determinism()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, df = build_full_cache()
    save_pickle(env_info(det), out_root / "env.pkl")
    save_pickle(model_info(build_model("Celik-HC", dropout=0.0), "Celik-HC"), out_root / "model_info.pkl")
    rows = []
    for seed in C.REPLICATION["seeds"]:
        split = replication_split(df, seed)
        if seed == C.REPLICATION["seeds"][0]:
            save_json({"replication": {k: v for k, v in C.REPLICATION.items()}, "n_images": len(df),
                       "leakage_audit": leakage_audit(df, split)}, out_root / "protocol.json")
            save_pickle({k: v for k, v in split.items()}, out_root / "split.pkl")
        for protocol in PROTOCOLS:
            run = out_root / protocol / f"seed_{seed:02d}"
            if not is_done(run):
                res = _train(seed, protocol, str(cache), df, split, dev, log)
                run.mkdir(parents=True, exist_ok=True)
                for k, v in res.items():
                    save_pickle(v, run / f"{k}.pkl")
                touch_done(run)
            from .persist import load_pickle
            rows.append(load_pickle(run / "summary.pkl"))
    tab = pd.DataFrame(rows)
    rep = C.REPLICATION["reported"]
    tab["accuracy_reportada"] = rep["accuracy"]
    tab["discrepancia_accuracy"] = tab["accuracy"] - rep["accuracy"]
    tab["reportado_en_ic"] = tab["accuracy_ic_boot"].apply(lambda ic: ic[0] <= rep["accuracy"] <= ic[1])
    save_csv(tab.drop(columns=["accuracy_ic_boot"]).assign(
        ic_lo=tab["accuracy_ic_boot"].str[0], ic_hi=tab["accuracy_ic_boot"].str[1]), out_root / "replication_summary.csv")
    backup_tree(out_root, C.BACKUP_DIR, C.RESULTS)
    return tab
