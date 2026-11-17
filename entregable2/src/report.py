"""Tablas de resumen y pruebas estadísticas a partir de las corridas terminadas (results/summary)."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .design import hpo_table
from .evaluate import collect_runs
from .persist import is_done, load_pickle, save_csv, save_json, save_text
from .stats import (friedman_permutation, mcnemar_all, mean_ci, minimum_detectable_effect, overfitting_gap, pairwise,
                    posthoc, variance_decomposition, wide)

SUMMARY_METRICS = ["f1_macro", "accuracy", "balanced_accuracy", "f1_weighted", "precision_macro", "recall_macro",
                   "auc_roc_macro", "auc_pr_macro", "mcc", "ece", "nll", "brier"]


def _final_runs(model: str, root: Path):
    mdir = root / model
    return [d for d in sorted(mdir.glob("seed_*")) if is_done(d)] if mdir.exists() else []


def _fmt(ci: dict) -> str:
    return f"{ci['media']:.3f} ± {ci['de']:.3f} [{ci['ic_t'][0]:.3f}, {ci['ic_t'][1]:.3f}]"


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("model", sort=False):
        row = {"model": m, "n_semillas": len(g)}
        for k in SUMMARY_METRICS:
            if k in g and g[k].notna().all() and len(g) > 1:
                row[k] = _fmt(mean_ci(g[k]))
        for k in ("params", "gmacs", "size_mb", "train_time_s", "epochs_run", "s_per_epoch", "ms_per_image"):
            if k in g:
                row[k] = float(g[k].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _per_run_tables(models: list[str], root: Path) -> dict[str, pd.DataFrame]:
    per_class, calib, unc, rob, cam, over, head = [], [], [], [], [], [], []
    for m in models:
        for d in _final_runs(m, root):
            s = int(d.name.split("_")[1])
            met = load_pickle(d / "metrics_test.pkl")
            per_class += [{"model": m, "seed": s, "clase": c, "f1": v, "precision": met["precision_per_class"][c],
                           "recall": met["recall_per_class"][c]} for c, v in met["f1_per_class"].items()]
            if (d / "metrics_test_head.pkl").exists():
                mh = load_pickle(d / "metrics_test_head.pkl")
                head.append({"model": m, "seed": s, "f1_macro_xgb": met["f1_macro"], "f1_macro_softmax": mh["f1_macro"],
                             "accuracy_xgb": met["accuracy"], "accuracy_softmax": mh["accuracy"]})
            if (d / "calibration.pkl").exists():
                cal = load_pickle(d / "calibration.pkl")
                if cal.get("aplicable"):
                    calib.append({"model": m, "seed": s, "temperature": cal["temperature"],
                                  **{f"{k}_antes": v for k, v in cal["antes"].items()},
                                  **{f"{k}_despues": v for k, v in cal["despues"].items()}})
            if (d / "uncertainty_mc_dropout.pkl").exists():
                u = load_pickle(d / "uncertainty_mc_dropout.pkl")
                if u.get("aplicable"):
                    pt = load_pickle(d / "test_predictions.pkl")
                    ok = u["probs_mean"].argmax(1) == u["y_true"]
                    unc.append({"model": m, "seed": s, **{f"mc_{k}": v for k, v in u["metrics"].items()},
                                "mi_media_aciertos": float(u["mutual_information"][ok].mean()) if ok.any() else np.nan,
                                "mi_media_errores": float(u["mutual_information"][~ok].mean()) if (~ok).any() else np.nan,
                                "entropia_media": float(u["entropy_total"].mean()),
                                "f1_macro_determinista": float(load_pickle(d / "metrics_test.pkl")["f1_macro"]),
                                "n_test": int(len(pt["y_true"]))})
            if (d / "robustness.pkl").exists():
                r = load_pickle(d / "robustness.pkl")
                clean = r["results"]["limpio"]["f1_macro"]
                rob += [{"model": m, "seed": s, "perturbacion": k, "f1_macro": v["f1_macro"], "accuracy": v["accuracy"],
                         "caida_f1": clean - v["f1_macro"]} for k, v in r["results"].items()]
            if (d / "gradcam.npz").exists():
                z = np.load(d / "gradcam.npz")
                cam.append({"model": m, "seed": s, "frac_energia_fondo": float(z["frac_energia_fondo"].mean()),
                            "frac_area_fondo": float(z["frac_area_fondo"].mean()),
                            "exceso_fondo": float((z["frac_energia_fondo"] - z["frac_area_fondo"]).mean())})
            if (d / "history.pkl").exists():
                over.append({"model": m, "seed": s, **overfitting_gap(load_pickle(d / "history.pkl"))})
    return {"f1_por_clase": pd.DataFrame(per_class), "calibracion": pd.DataFrame(calib),
            "incertidumbre_mc_dropout": pd.DataFrame(unc), "robustez": pd.DataFrame(rob),
            "gradcam_fondo": pd.DataFrame(cam), "sobreajuste": pd.DataFrame(over), "celik_xgb_vs_softmax": pd.DataFrame(head)}


def _model_table(models: list[str], root: Path) -> pd.DataFrame:
    rows = []
    for m in models:
        runs = _final_runs(m, root)
        if runs:
            inf = load_pickle(runs[0] / "model_info.pkl")
            rows.append({k: (", ".join(v) if isinstance(v, list) and k == "deviations" else v) for k, v in inf.items()
                         if k != "input"})
    return pd.DataFrame(rows)


def _test_preds(models: list[str], roots: dict[str, Path]) -> dict:
    out = {}
    for m in models:
        runs = _final_runs(m, roots[m])
        if runs:
            out[m] = {int(d.name.split("_")[1]): load_pickle(d / "test_predictions.pkl") for d in runs}
    return out


def build_summary(log=print) -> dict:
    C.SUMMARY.mkdir(parents=True, exist_ok=True)
    new = [m for m in C.MODELS if _final_runs(m, C.RESULTS)]
    base = [m for m in C.BASE_MODELS if C.E1_RESULTS.exists() and _final_runs(m, C.E1_RESULTS)]
    roots = {**{m: C.RESULTS for m in new}, **{m: C.E1_RESULTS for m in base}}
    df_new = collect_runs(new, C.RESULTS)
    df_base = collect_runs(base, C.E1_RESULTS) if base else pd.DataFrame()
    if len(df_base):
        df_base["origen"] = "Entregable 1"
    if len(df_new):
        df_new["origen"] = "Entregable 2"
    df = pd.concat([df_new, df_base], ignore_index=True)
    if df.empty:
        log("No hay corridas finales terminadas.")
        return {}
    save_csv(df, C.SUMMARY / "metricas_por_corrida.csv")
    save_csv(benchmark_table(df), C.SUMMARY / "tabla_benchmark.csv")
    save_csv(_model_table(new, C.RESULTS), C.SUMMARY / "tabla_modelos.csv")
    for name, tab in _per_run_tables(new, C.RESULTS).items():
        if len(tab):
            save_csv(tab, C.SUMMARY / f"{name}.csv")
    if base:
        rb = _per_run_tables(base, C.E1_RESULTS)["robustez"]
        if len(rb):
            save_csv(rb, C.SUMMARY / "robustez_modelos_base.csv")
    hpo = hpo_table([m for m in C.MODELS if (C.RESULTS / m / "hpo").exists()])
    if len(hpo):
        save_csv(hpo, C.SUMMARY / "busqueda_hiperparametros.csv")

    stats = {}
    order = new + base
    w = wide(df, C.PRIMARY_METRIC, order)
    if w.shape[1] >= 2 and w.shape[0] >= 3:
        save_csv(w.reset_index(), C.SUMMARY / "f1_macro_por_semilla.csv")
        if w.shape[1] >= 3:
            stats["friedman"] = friedman_permutation(w)
            ph = posthoc(w)
            if ph.get("disponible"):
                ph["nemenyi"].to_csv(C.SUMMARY / "posthoc_nemenyi.csv")
                ph["conover"].to_csv(C.SUMMARY / "posthoc_conover_holm.csv")
        save_csv(pairwise(w), C.SUMMARY / "pruebas_pares.csv")
        save_csv(minimum_detectable_effect(w), C.SUMMARY / "diferencia_minima_detectable.csv")
        stats["descomposicion_varianza"] = variance_decomposition(df[df["model"].isin(w.columns)])
        stats["semillas_usadas"] = [int(s) for s in w.index]
        stats["modelos"] = list(w.columns)
        preds = _test_preds(list(w.columns), roots)
        if len(preds) >= 2:
            save_csv(mcnemar_all(preds), C.SUMMARY / "mcnemar_por_semilla.csv")
    save_json(stats, C.SUMMARY / "estadistica_global.json")

    abl = ablation_summary()
    if abl is not None:
        save_csv(abl, C.SUMMARY / "ablacion.csv")
    rep = C.RESULTS / "replication" / "replication_summary.csv"
    if rep.exists():
        save_csv(pd.read_csv(rep), C.SUMMARY / "replicacion_paper_guia.csv")
    save_text(f"Resumen generado el {time.strftime('%Y-%m-%d %H:%M')} con {len(df)} corridas finales.\n\n"
              "Regla de decisión por pares: una diferencia es relevante y significativa si |diferencia media| >= "
              f"{C.DELTA} y el p de la t pareada corregido por Holm es < {C.ALPHA}. La columna "
              "relevante_y_significativa_wilcoxon aplica la misma regla con Wilcoxon exacto; con n semillas su p "
              "mínimo es 2/2^n y, tras Holm sobre todos los pares, queda acotado por p_wilcoxon_holm_minimo_posible, "
              "de modo que con 8 semillas y 45 pares no puede ser < 0,05. shapiro_p_dif permite revisar la normalidad "
              "de las diferencias que supone la t. En ablacion.csv, dif_media = modelo completo - variante: un valor "
              "positivo indica que el componente retirado aporta.\n", C.SUMMARY / "LEEME.txt")
    log(f"Resumen escrito en {C.SUMMARY}")
    return stats


def ablation_summary() -> pd.DataFrame | None:
    root = C.RESULTS / "ablation"
    if not root.exists():
        return None
    rows = []
    for vdir in sorted(p for p in root.rglob("*") if p.is_dir() and p.name.startswith("seed_")):
        if not is_done(vdir):
            continue
        variant = vdir.parent.relative_to(root).as_posix()
        met = load_pickle(vdir / "metrics_test.pkl")
        rows.append({"variante": variant, "modelo_base": variant.split("/")[0], "seed": int(vdir.name.split("_")[1]),
                     "f1_macro": met["f1_macro"], "accuracy": met["accuracy"], "ece": met["ece"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = []
    for base, g in df.groupby("modelo_base"):
        full = collect_runs([base], C.RESULTS)
        if full.empty:
            continue
        w = pd.concat([full[["seed", "f1_macro"]].assign(variante=base), g[["seed", "f1_macro", "variante"]]])
        w = w.pivot(index="seed", columns="variante", values="f1_macro").dropna()
        if w.shape[0] >= 3:
            pairs = [(base, v) for v in w.columns if v != base]
            out.append(pairwise(w, pairs=pairs).assign(modelo_base=base))
    detail = df.groupby("variante").agg(n=("seed", "count"), f1_media=("f1_macro", "mean"), f1_de=("f1_macro", "std"),
                                         accuracy_media=("accuracy", "mean"), ece_media=("ece", "mean")).reset_index()
    if out:
        detail = detail.merge(pd.concat(out)[["B", "dif_media", "ic_t_lo", "ic_t_hi", "p_t_holm", "p_wilcoxon_holm",
                                              "relevante_y_significativa", "relevante_y_significativa_wilcoxon"]]
                              .rename(columns={"B": "variante"}),
                              on="variante", how="left")
    return detail


def package_results(dest: Path | None = None, include_weights: bool = False, log=print) -> Path:
    """Comprime results/ (sin checkpoints de reanudación) para descargar o adjuntar al informe."""
    dest = Path(dest or (Path(C.BACKUP_DIR) if C.BACKUP_DIR else C.RESULTS.parent))
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"resultados_entrega2_{time.strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in C.RESULTS.rglob("*"):
            if not f.is_file() or f.suffix in (".tmp", ".lock") or f.name.startswith("checkpoint_"):
                continue
            if f.relative_to(C.RESULTS).parts[0].startswith("_") or f.suffix == ".zip":
                continue
            if f.suffix == ".pt" and not include_weights and "teacher_effnetb4" not in f.parts:
                continue
            z.write(f, Path("results") / f.relative_to(C.RESULTS))
    log(f"Resultados comprimidos en {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return out
