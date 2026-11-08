"""Ejecutor del experimento desde la terminal (recomendado frente a Jupyter).

Uso (desde la raíz del repositorio, con el entorno activado):

    python -m src.runner status          # tabla de corridas completas, parciales y pendientes
    python -m src.runner estimate        # mide s/época (2 épocas por modelo) y estima horas de GPU
    python -m src.runner all             # prepare -> hpo -> pilot -> power -> final
    python -m src.runner hpo | pilot | power | final | robustness

Si el proceso se interrumpe (corte de luz, cierre de la terminal, Ctrl+C), basta
con volver a ejecutar el MISMO comando: las corridas con DONE se saltan y las
parciales se reanudan desde la última época guardada.

--smoke ejecuta todo con un subconjunto pequeño y 2 épocas en results_smoke/
(no toca results/). Sirve para comprobar la instalación antes de entrenar.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse():
    ap = argparse.ArgumentParser(description="Entrenamiento de los modelos base (TrashNet)")
    ap.add_argument("stage", choices=["status", "prepare", "estimate", "hpo", "pilot", "power", "final",
                                      "robustness", "all"])
    ap.add_argument("--models", nargs="+", default=None, help="subconjunto de modelos (por defecto, los tres)")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida en results_smoke/")
    ap.add_argument("--workers", type=int, default=None, help="num_workers del DataLoader (0 = sin procesos)")
    return ap.parse_args()


def _keep_awake(on: bool) -> None:
    """Impide que Windows suspenda el equipo mientras el ejecutor está activo."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def main() -> int:
    args = _parse()
    if args.smoke:
        os.environ["TRASHNET_SMOKE"] = "1"
        os.environ.setdefault("TRASHNET_RESULTS", str(ROOT / "results_smoke"))
    if args.workers is not None:
        os.environ["TRASHNET_WORKERS"] = str(args.workers)

    from . import config as C                    # después de fijar las variables de entorno
    C.ensure_dirs()
    models = args.models or C.MODELS
    subset = {"train": 48, "val": 24, "test": 24} if args.smoke else None

    log_file = C.LOGS / f"runner_{time.strftime('%Y-%m-%d')}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    import torch
    from .data import prepare_common
    from .design import pilot_and_power, select_hpo, write_preregistro
    from .persist import is_done, load_pickle, load_torch
    from .train import train_run

    log(f"=== {args.stage} | resultados en {C.RESULTS} | dispositivo: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | workers={C.NUM_WORKERS}")
    if not torch.cuda.is_available() and not args.smoke and args.stage not in ("status", "prepare", "power"):
        log("AVISO: no se detecta GPU. Revisa la instalación de PyTorch con CUDA antes de entrenar.")

    errors: list[str] = []

    def run_one(model, seed, hp, run, stage, eval_test, extra=None):
        for attempt in (1, 2):
            try:
                t0 = time.time()
                r = train_run(model, seed, hp, run, common, stage=stage, eval_test=eval_test,
                              subset=subset, log=log, extra_config=extra)
                if r == "done":
                    log(f"OK {model} {run.name} ({(time.time() - t0) / 60:.1f} min)")
                return r
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                (run / f"ERROR_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(tb, encoding="utf-8")
                worker_issue = any(s in tb for s in ("DataLoader worker", "BrokenPipe", "pickle", "spawn"))
                if attempt == 1 and worker_issue and C.NUM_WORKERS > 0:
                    log(f"Fallo de los workers del DataLoader en {model} {run.name}; se reintenta con num_workers=0.")
                    C.NUM_WORKERS = 0
                    continue
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    log("Memoria de GPU insuficiente. Baja batch_size a 16 en config.DEFAULT_HP para LOS TRES "
                        "modelos y borra las carpetas de corridas que no tengan DONE.")
                    raise
                log(f"ERROR en {model} {run.name}: {e!r} (detalle en {run.name}/ERROR_*.txt)")
                errors.append(f"{model} {run.name}")
                return "error"

    def status() -> None:
        import pandas as pd
        rows = []
        for m in models:
            hp_done = sum(is_done(C.hpo_dir(m, t)) for t in C.HPO_TRIALS_USED)
            seeds = sorted(C.RESULTS.joinpath(m).glob("seed_*")) if C.RESULTS.joinpath(m).exists() else []
            done = [d.name for d in seeds if is_done(d)]
            part = []
            for d in seeds:
                if not is_done(d) and (d / "history.pkl").exists():
                    h = load_pickle(d / "history.pkl")
                    part.append(f"{d.name}@ep{h['epoch'][-1] if h['epoch'] else 0}")
            rows.append({"modelo": m, "HPO completos": f"{hp_done}/{len(C.HPO_TRIALS_USED)}",
                         "semillas completas": len(done), "parciales": ", ".join(part) or "-"})
        print(pd.DataFrame(rows).to_string(index=False))
        pa = C.COMMON / "power_analysis.pkl"
        if pa.exists():
            p = load_pickle(pa)
            print(f"R_final = {p['R_final']} (R calculado = {p['R_calculado']}, MDE = {p['MDE']:.4f}, delta = {p['delta']})")

    _keep_awake(True)
    try:
        if args.stage == "status":
            status()
            return 0
        common = prepare_common()
        log(f"Partición v2 verificada: train {len(common['splits']['train']['y'])}, val {len(common['splits']['val']['y'])}, "
            f"test {len(common['splits']['test']['y'])}")
        if args.stage == "prepare":
            return 0
        write_preregistro()

        if args.stage == "estimate":
            import shutil
            est = {}
            for m in models:
                d = C.RESULTS / "_estimate" / m
                shutil.rmtree(d, ignore_errors=True)
                hp = {**C.DEFAULT_HP, "max_epochs": 3, "warmup_epochs": 1}
                train_run(m, 999, hp, d, common, stage="estimate", eval_test=False, subset=subset, log=log)
                h = load_pickle(d / "history.pkl")
                est[m] = float(sum(h["epoch_time_s"][1:]) / max(1, len(h["epoch_time_s"]) - 1))
                shutil.rmtree(d, ignore_errors=True)
            n_hpo = len(C.HPO_TRIALS_USED)
            for frac, name in ((1.0, "peor caso (todas las épocas)"), (0.6, "típico (early stopping ~60 %)")):
                hpo_h = sum(est[m] * C.HPO_MAX_EPOCHS * frac * n_hpo for m in est) / 3600
                run_h = sum(est[m] * C.DEFAULT_HP["max_epochs"] * frac for m in est) / 3600
                log(f"{name}: HPO {hpo_h:.1f} h + {run_h:.2f} h por semilla de los 3 modelos "
                    f"-> con R=6: {hpo_h + 6 * run_h:.1f} h; con R=10: {hpo_h + 10 * run_h:.1f} h")
            for m, s in est.items():
                log(f"  {m}: {s:.1f} s/época")
            return 0

        stages = ["hpo", "pilot", "power", "final"] if args.stage == "all" else [args.stage]
        for stg in stages:
            if stg == "hpo":
                for t in C.HPO_TRIALS_USED:                  # intercalado: el trial t de los 3 modelos
                    for m in models:
                        hp = {**C.DEFAULT_HP, **C.HPO_GRID[t], "max_epochs": C.HPO_MAX_EPOCHS}
                        run_one(m, C.HPO_SEED, hp, C.hpo_dir(m, t), "hpo", eval_test=False, extra={"trial": t})
                sel = select_hpo(C.MODELS)
                for m, v in sel.items():
                    log(f"HPO {m}: trial {v['trial']} (F1 val {v['val_f1_macro']:.4f}) -> "
                        f"lr={v['hp']['lr']}, wd={v['hp']['weight_decay']}, dropout={v['hp']['dropout']}")
            elif stg in ("pilot", "final"):
                f_sel = C.COMMON / "hpo_selected.pkl"
                if not f_sel.exists():
                    log("Falta hpo_selected.pkl: ejecuta primero la etapa hpo.")
                    return 1
                sel = load_pickle(f_sel)
                if stg == "pilot":
                    seeds = list(C.PILOT_SEEDS)
                else:
                    f_pa = C.COMMON / "power_analysis.pkl"
                    if not f_pa.exists():
                        log("Falta power_analysis.pkl: ejecuta primero pilot y power.")
                        return 1
                    seeds = list(range(max(load_pickle(f_pa)["R_final"], len(C.PILOT_SEEDS))))
                for s in seeds:                              # intercalado: semilla s de los 3 modelos
                    for m in models:
                        run_one(m, s, sel[m]["hp"], C.run_dir(m, s), "final", eval_test=True,
                                extra={"pilot": s in C.PILOT_SEEDS, "hpo_trial": sel[m]["trial"]})
            elif stg == "power":
                out = pilot_and_power(C.MODELS)
                pa = out["power"]
                log(f"Piloto: R calculado = {pa['R_calculado']}, R_final = {pa['R_final']}, MDE = {pa['MDE']:.4f} "
                    f"(delta = {pa['delta']})" + (" -> MDE > delta, se declara como limitación" if pa["MDE_supera_delta"] else ""))
            elif stg == "robustness":
                from .analysis import PERTURBATIONS, perturbed_test_set, robustness_run
                from .data import load_cache
                cache = load_cache()
                rows = common["splits"]["test"]["rows"]
                sets = {k: perturbed_test_set(cache, rows, k) for k in PERTURBATIONS}
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for m in models:
                    for d in sorted((C.RESULTS / m).glob("seed_*")):
                        if is_done(d) and not (d / "robustness.pkl").exists():
                            robustness_run(m, d, common, cache, dev, sets)
                            log(f"robustez {m} {d.name} lista")
        status()
        if errors:
            log(f"Terminó con {len(errors)} corrida(s) con error: {errors}. Vuelve a ejecutar el comando para reintentarlas.")
            return 2
        log("Etapa completada.")
        return 0
    except KeyboardInterrupt:
        log("Interrumpido. El progreso hasta la última época está guardado: vuelve a ejecutar el mismo comando.")
        return 130
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
