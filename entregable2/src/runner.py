"""Ejecutor del experimento.

    python -m src.runner status
    python -m src.runner estimate [--models ...]
    python -m src.runner all [--models ...]
    python -m src.runner prepare | teacher | hpo | final | ablation | replicate | gradcam | summary | package

Si el proceso se interrumpe, basta con repetir el mismo comando: las corridas con DONE se
saltan y las parciales se reanudan desde la última época guardada.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["status", "prepare", "estimate", "teacher", "hpo", "final", "ablation", "replicate", "gradcam", "summary",
          "package", "all"]


def _parse():
    ap = argparse.ArgumentParser(description="Benchmark del Entregable 2 (TrashNet)")
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--models", nargs="+", default=None, help="subconjunto de modelos (por defecto, los siete)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="subconjunto de semillas de la etapa final")
    ap.add_argument("--variants", nargs="+", default=None, help="variantes de ablación (por defecto, las del paper guía)")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida con un subconjunto en results_smoke/")
    ap.add_argument("--workers", type=int, default=None, help="num_workers del DataLoader (0 = sin procesos)")
    return ap.parse_args()


def _keep_awake(on: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def main() -> int:
    args = _parse()
    if args.smoke:
        os.environ["TRASHNET_SMOKE"] = "1"
        os.environ.setdefault("TRASHNET_RESULTS", str(ROOT / "results_smoke"))
    if args.workers is not None:
        os.environ["TRASHNET_WORKERS"] = str(args.workers)

    from . import config as C
    C.ensure_dirs()
    models = args.models or C.MODELS
    unknown = [m for m in models if m not in C.MODELS]
    if unknown:
        print(f"Modelos desconocidos: {unknown}. Opciones: {C.MODELS}")
        return 1
    subset = {"train": 48, "val": 24, "test": 24} if args.smoke else None
    log_file = C.LOGS / f"runner_{time.strftime('%Y-%m-%d')}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    import torch
    from . import train as T
    from .data import prepare_common
    from .design import select_hpo, trial_hp, write_preregistro, write_requirements_lock
    from .persist import backup_run, backup_tree, is_done, load_pickle, save_csv
    from .report import build_summary, package_results

    log(f"=== {args.stage} | resultados: {C.RESULTS} | respaldo: {C.BACKUP_DIR or 'no'} | dispositivo: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | workers={C.NUM_WORKERS} | "
        f"accum={C.ACCUM_STEPS}")
    if not torch.cuda.is_available() and not args.smoke and args.stage not in ("status", "prepare", "summary", "package"):
        log("AVISO: no se detecta GPU.")

    errors: list[str] = []

    def run_one(model, seed, hp, run, stage, extra=None):
        for attempt in (1, 2):
            try:
                t0 = time.time()
                r = T.train_run(model, seed, hp, run, common, stage=stage, subset=subset, log=log, extra_config=extra)
                if r == "done":
                    log(f"OK {model} {run.relative_to(C.RESULTS)} ({(time.time() - t0) / 60:.1f} min)")
                return r
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                run.mkdir(parents=True, exist_ok=True)
                (run / f"ERROR_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(tb, encoding="utf-8")
                if attempt == 1 and C.NUM_WORKERS > 0 and any(s in tb for s in ("DataLoader worker", "BrokenPipe", "spawn")):
                    log(f"Fallo de los workers en {model} {run.name}; se reintenta con num_workers=0.")
                    C.NUM_WORKERS = 0
                    continue
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    log("Memoria de GPU insuficiente. Repite con TRASHNET_ACCUM=2 (o 4): el lote efectivo se mantiene en 32. "
                        "Borra antes las carpetas de corridas sin DONE de este modelo.")
                    raise
                log(f"ERROR en {model} {run.name}: {e!r} (detalle en {run}/ERROR_*.txt)")
                errors.append(f"{model} {run.name}")
                return "error"

    def sync_common():
        for d in (C.COMMON, C.SUMMARY, C.LOGS):
            backup_tree(d, C.BACKUP_DIR, C.RESULTS)

    def status():
        import pandas as pd
        rows = []
        sel = load_pickle(C.COMMON / "hpo_selected.pkl") if (C.COMMON / "hpo_selected.pkl").exists() else {}
        for m in models:
            done_hpo = sum(is_done(C.hpo_dir(m, t)) for t in C.HPO_TRIALS_USED)
            seeds = [s for s in C.SEEDS if is_done(C.run_dir(m, s))]
            part = []
            for s in C.SEEDS:
                d = C.run_dir(m, s)
                if not is_done(d) and (d / "history.pkl").exists():
                    part.append(f"{d.name}@ep{load_pickle(d / 'history.pkl')['epoch'][-1]}")
            rows.append({"modelo": m, "HPO": f"{done_hpo}/{len(C.HPO_TRIALS_USED)}",
                         "trial elegido": sel.get(m, {}).get("trial", "-"), "semillas": f"{len(seeds)}/{len(C.SEEDS)}",
                         "parciales": ", ".join(part) or "-"})
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"Profesor de FocusRCNet: {'listo' if T.teacher_path().exists() else 'pendiente'}")

    def stage_teacher():
        if "FocusRCNet" not in models:
            return
        run = T.teacher_path().parent
        run_one("EffNetB4-Teacher", C.HPO_SEED, dict(C.TEACHER_HP), run, "teacher")
        if is_done(run):
            backup_run(run, C.BACKUP_DIR, C.RESULTS, include_weights=True)

    def stage_hpo():
        if "FocusRCNet" in models and not T.teacher_path().exists():
            stage_teacher()
        for t in C.HPO_TRIALS_USED:
            for m in models:
                run_one(m, C.HPO_SEED, trial_hp(m, t), C.hpo_dir(m, t), "hpo", extra={"trial": t})
        sel = select_hpo(models)
        for m in models:
            if m in sel:
                v = sel[m]
                log(f"HPO {m}: trial {v['trial']} (F1 val {v['val_f1_macro']:.4f}) -> lr={v['hp']['lr']:g}, "
                    f"wd={v['hp']['weight_decay']:g}, dropout={v['hp']['dropout']}")
        sync_common()

    def selected() -> dict | None:
        f = C.COMMON / "hpo_selected.pkl"
        sel = load_pickle(f) if f.exists() else {}
        missing = [m for m in models if m not in sel]
        if missing:
            log(f"Falta la búsqueda de hiperparámetros de {missing}: ejecuta antes la etapa hpo.")
            return None
        return sel

    def stage_final():
        sel = selected()
        if sel is None:
            return 1
        if "FocusRCNet" in models and not T.teacher_path().exists():
            stage_teacher()
        for s in args.seeds or C.SEEDS:
            for m in models:
                run_one(m, s, sel[m]["hp"], C.run_dir(m, s), "final", extra={"hpo_trial": sel[m]["trial"]})
        sync_common()
        return 0

    def stage_ablation():
        from .models import ABLATIONS
        variants = args.variants or [v for v in ABLATIONS if v.startswith(C.GUIDE_MODEL + "/")]
        f = C.COMMON / "hpo_selected.pkl"
        sel = load_pickle(f) if f.exists() else {}
        for v in variants:
            base = ABLATIONS[v][0]
            if base not in sel:
                log(f"Falta la búsqueda de hiperparámetros de {base}; se omite {v}.")
                continue
            if base == "FocusRCNet" and not T.teacher_path().exists():
                stage_teacher()
            for s in args.seeds or C.SEEDS:
                run_one(v, s, sel[base]["hp"], C.ablation_dir(v, s), "final",
                        extra={"ablation_of": base, "hpo_trial": sel[base]["trial"]})
        sync_common()

    def stage_gradcam():
        import numpy as np
        for m in models:
            for s in args.seeds or C.SEEDS:
                run = C.run_dir(m, s)
                f = run / "gradcam.npz"
                if not is_done(run):
                    continue
                if f.exists():
                    with np.load(f) as z:
                        if float(np.asarray(z["cam_pred"], dtype=np.float32).max()) > 0:
                            continue
                t0 = time.time()
                try:
                    o = T.regenerate_gradcam(m, run, common, C.RESULTS / "_gradcam", subset=subset, log=log)
                except Exception as e:
                    (run / f"ERROR_gradcam_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(traceback.format_exc(),
                                                                                         encoding="utf-8")
                    log(f"ERROR Grad-CAM {m} {run.name}: {e!r}")
                    errors.append(f"gradcam {m} {run.name}")
                    continue
                extra = (f" | reentrenado: concordancia {o['concordancia_predicciones_prueba']:.3f}, F1 prueba "
                         f"{o['f1_macro_prueba_reentrenado']:.4f} (original {o['f1_macro_prueba_original']:.4f})"
                         if o["reentrenado"] else "")
                log(f"OK Grad-CAM {m} {run.name} capa {o['capa']} ({(time.time() - t0) / 60:.1f} min){extra}")

    def stage_estimate():
        import shutil
        import pandas as pd
        est_root = C.RESULTS / "_estimate"
        shutil.rmtree(est_root, ignore_errors=True)
        rows = []
        quick = {**C.DEFAULT_HP, "max_epochs": 3, "warmup_epochs": 1}
        if "FocusRCNet" in models and not T.teacher_path().exists():
            tdir = est_root / "teacher"
            T.train_run("EffNetB4-Teacher", 999, quick, tdir, common, stage="teacher", subset=subset, log=log)
            h = load_pickle(tdir / "history.pkl")["epoch_time_s"]
            rows.append({"modelo": "EffNetB4-Teacher", "s_epoca": float(sum(h[1:]) / max(1, len(h) - 1)), "veces": 1})
            T.teacher_path = lambda: tdir / "best_model.pt"
        for m in models:
            d = est_root / m
            T.train_run(m, 999, {**trial_hp(m, 0, hpo=False), "max_epochs": 3, "warmup_epochs": 1}, d, common,
                        stage="hpo", subset=subset, log=log)
            h = load_pickle(d / "history.pkl")["epoch_time_s"]
            rows.append({"modelo": m, "s_epoca": float(sum(h[1:]) / max(1, len(h) - 1)), "veces": None})
        shutil.rmtree(est_root, ignore_errors=True)
        tab = pd.DataFrame(rows)
        n_hpo, n_seed = len(C.HPO_TRIALS_USED), len(C.SEEDS)
        for frac, name in ((1.0, "h_peor_caso"), (0.7, "h_tipico")):
            tab[name] = [
                r.s_epoca * C.TEACHER_HP["max_epochs"] * frac / 3600 if r.modelo == "EffNetB4-Teacher"
                else r.s_epoca * (n_hpo * C.HPO_MAX_EPOCHS + n_seed * C.DEFAULT_HP["max_epochs"]) * frac / 3600
                for r in tab.itertuples()]
        tab = tab.drop(columns="veces")
        print(tab.round(2).to_string(index=False))
        log(f"Total: peor caso {tab.h_peor_caso.sum():.1f} h; típico (parada temprana ~70 %) {tab.h_tipico.sum():.1f} h "
            f"de GPU para {', '.join(models)} (sin contar robustez, Grad-CAM ni ablación)")
        save_csv(tab, C.SUMMARY / f"estimacion_tiempo_{time.strftime('%Y%m%d_%H%M')}.csv")

    _keep_awake(True)
    try:
        if args.stage == "status":
            status()
            return 0
        if args.stage == "summary":
            build_summary(log)
            sync_common()
            return 0
        if args.stage == "package":
            build_summary(log)
            package_results(log=log)
            return 0
        common = prepare_common(log=log)
        sp = common["splits"]
        log(f"Partición v2 verificada: train {len(sp['train']['y'])}, val {len(sp['val']['y'])}, test {len(sp['test']['y'])}")
        write_preregistro()
        write_requirements_lock(C.COMMON / "requirements-lock.txt")
        if args.stage == "prepare":
            sync_common()
            return 0
        if args.stage == "estimate":
            stage_estimate()
            return 0
        if args.stage == "replicate":
            from .replication import run_replication
            print(run_replication(log).to_string(index=False))
            return 0
        if args.stage == "teacher":
            stage_teacher()
        elif args.stage == "hpo":
            stage_hpo()
        elif args.stage == "final":
            if stage_final():
                return 1
        elif args.stage == "ablation":
            stage_ablation()
        elif args.stage == "gradcam":
            stage_gradcam()
        elif args.stage == "all":
            stage_hpo()
            if stage_final():
                return 1
            build_summary(log)
            sync_common()
            package_results(log=log)
        status()
        if errors:
            log(f"Terminó con {len(errors)} corrida(s) con error: {errors}. Repite el comando para reintentarlas.")
            return 2
        log("Etapa completada.")
        return 0
    except KeyboardInterrupt:
        log("Interrumpido. El progreso hasta la última época está guardado: repite el mismo comando.")
        return 130
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
