"""Orquestador del Entregable 3 (Colab Pro).

    python -m e3.orchestrator setup                 # rutas, enlaces y restauración desde Drive
    python -m e3.orchestrator prepare               # datos, s_lo y conjuntos adicionales
    python -m e3.orchestrator run --worker 0 [--hours 20]
    python -m e3.orchestrator status
    python -m e3.orchestrator extra-pass            # evalúa lo que quedó pendiente en los conjuntos adicionales
    python -m e3.orchestrator smoke                 # prueba de humo completa en carpetas _smoke
    python -m e3.orchestrator piloto                # opción B: resumen del piloto en validación y regla de elección
La cola la elige la variable E3_COLA (a | b_piloto | b); sin ella se usa la cola congelada de v1-v3.

Cada tarea corre en un proceso aparte (e3.worker). Si la sesión se corta, basta con volver a ejecutar
setup y run: lo terminado se salta (marcas .done en Drive) y lo parcial se reanuda desde el último
checkpoint reflejado en Drive.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import cfg
from . import sync as S
from .worker import EXIT_BLOCKED, EXIT_OK, EXIT_OOM, run_path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True) if src.suffix == "" and not src.exists() else None
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(src, dst, target_is_directory=src.is_dir())


def setup() -> None:
    """Prepara las rutas que esperan los códigos congelados del E1 y del E2 y restaura desde Drive."""
    for d in (cfg.WORK, cfg.DATA, cfg.EXTRA_DIR, *cfg.RESULTS.values(), cfg.DRIVE, cfg.DRIVE_CLAIMS,
              *cfg.DRIVE_RESULTS.values(), cfg.DRIVE_LOGS):
        d.mkdir(parents=True, exist_ok=True)
    # el E2 busca la partición y las huellas en entregable2/data (en el repositorio están en data/)
    _link(cfg.REPO / "data" / "splits", cfg.E2_ROOT / "data" / "splits")
    _link(cfg.REPO / "data" / "SHA256SUMS.txt", cfg.E2_ROOT / "data" / "SHA256SUMS.txt")
    # el E1 busca las imágenes en data/raw/trashnet (el E2 las descarga en DATA/raw)
    _link(cfg.DATA / "raw", cfg.REPO / "data" / "raw" / "trashnet")
    # common/ congelado de cada entregable (partición, normalización, pesos de clase, hiperparámetros elegidos)
    for fam, src in (("e2", cfg.E2_ROOT / "results" / "common"), ("e1", cfg.REPO / "results" / "common")):
        dst = cfg.RESULTS[fam] / "common"
        if not dst.exists():
            shutil.copytree(src, dst)
    n = {fam: S.restore_results(fam) for fam in cfg.RESULTS}
    for name in cfg.EXTRA_DATASETS:
        src = cfg.DRIVE_EXTRA / f"{name}.npz"
        if src.exists() and not (cfg.EXTRA_DIR / f"{name}.npz").exists():
            shutil.copy2(src, cfg.EXTRA_DIR / src.name)
            shutil.copy2(cfg.DRIVE_EXTRA / f"{name}.json", cfg.EXTRA_DIR / f"{name}.json")
    S.log_line(f"setup listo | trabajo {cfg.WORK} | Drive {cfg.DRIVE} | archivos restaurados {n}")


def _worker_cmd(t: dict, worker: str) -> list[str]:
    return [sys.executable, "-m", "e3.worker", json.dumps(t), worker]


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cfg.E3_ROOT)
    env.setdefault("TRASHNET_WORKERS", "4")
    return env


def run_task(t: dict, worker: str) -> int:
    p = subprocess.Popen(_worker_cmd(t, worker), cwd=str(cfg.E3_ROOT), env=_env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def prepare(worker: str = "0") -> int:
    rc = run_task({"fam": "e2", "kind": "prep", "model": "-", "id": "prep-e2"}, worker)
    if rc != EXIT_OK:
        S.log_line("La preparación del E2 falló; revisa el registro antes de entrenar.")
        return rc
    rc = run_task({"fam": "e1", "kind": "prep", "model": "-", "id": "prep-e1"}, worker)
    if rc != EXIT_OK:
        S.log_line("La preparación del E1 falló; revisa el registro antes de entrenar.")
    return rc


def _is_finished(t: dict) -> bool:
    if S.task_done(t["id"]):
        return True
    run = run_path(t)
    return (run / "DONE").exists() and (t["kind"] == "hpo" or (run / "E3_HOOK.json").exists())


def run(worker: str = "0", hours: float = 1e9, queue: list[dict] | None = None, only: str | None = None) -> int:
    queue = queue or cfg.build_queue()
    if only:
        queue = [t for t in queue if t["fam"] == only]
    t_end = time.time() + hours * 3600
    fails: dict[str, int] = {}
    durations: list[tuple[str, float]] = []
    S.log_line(f"Inicio de la cola: {len(queue)} tareas, {sum(_is_finished(t) for t in queue)} ya terminadas, "
               f"worker {worker}, límite {hours:g} h")
    idle_rounds = 0
    while True:
        if time.time() > t_end:
            S.log_line("Se alcanzó el límite de horas de esta sesión: no se lanzan tareas nuevas.")
            break
        launched, blocked_any = False, False
        for t in queue:
            if _is_finished(t) or fails.get(t["id"], 0) >= 2:
                continue
            if not S.try_claim(t["id"], worker):
                continue
            S.restore_checkpoint(run_path(t), "e1" if t["fam"] == "e1" else "e2")
            S.log_line(f">>> {t['id']}")
            t0 = time.time()
            rc = run_task(t, worker)
            dt = (time.time() - t0) / 60
            if rc == EXIT_OK:
                S.mark_done(t["id"], {"worker": worker, "minutos": round(dt, 2)})
                durations.append((t["id"], dt))
                S.log_line(f"<<< {t['id']} terminada en {dt:.1f} min | {_progress(queue)}")
                launched = True
                break                       # vuelve a recorrer la cola desde la prioridad más alta
            S.release_claim(t["id"])
            if rc == EXIT_BLOCKED:
                blocked_any = True
                continue
            fails[t["id"]] = fails.get(t["id"], 0) + 1
            S.log_line(f"!!! {t['id']} falló (código {rc}, intento {fails[t['id']]}/2)")
            if rc == EXIT_OOM:
                S.log_line("Memoria de GPU insuficiente: usa una GPU con más memoria (L4 o A100) o fija "
                           "TRASHNET_ACCUM=2 en la celda de parámetros (el lote efectivo sigue siendo 32).")
            launched = True
            break
        if launched:
            idle_rounds = 0
            continue
        pending = [t for t in queue if not _is_finished(t) and fails.get(t["id"], 0) < 2]
        if not pending:
            S.log_line("Cola completa.")
            break
        if blocked_any or any((cfg.DRIVE_CLAIMS / f"{t['id']}.json").exists() for t in pending):
            idle_rounds += 1
            if idle_rounds > 90:            # 90 min esperando a otra sesión
                S.log_line("Sin avances en 90 min; se detiene este worker.")
                break
            time.sleep(60)
            continue
        S.log_line(f"Quedan {len(pending)} tareas que fallaron dos veces; revisa Drive/E3_TrashNet/logs/ERROR_*.")
        break
    failed = [k for k, v in fails.items() if v >= 2]
    if failed:
        S.log_line(f"Tareas con error persistente: {failed}")
    return 0


def _progress(queue: list[dict]) -> str:
    done = sum(_is_finished(t) for t in queue)
    return f"{done}/{len(queue)} tareas terminadas"


def status(queue: list[dict] | None = None) -> None:
    queue = queue or cfg.build_queue()
    rows: dict[str, list[int]] = {}
    mins: dict[str, list[float]] = {}
    for t in queue:
        key = f"{t['fam']}:{t['kind']}:{t['model']}"
        r = rows.setdefault(key, [0, 0])
        r[1] += 1
        if _is_finished(t):
            r[0] += 1
            f = cfg.DRIVE_CLAIMS / f"{t['id']}.done"
            if f.exists():
                try:
                    mins.setdefault(key, []).append(float(json.loads(f.read_text())["minutos"]))
                except (ValueError, KeyError, OSError):
                    pass
    print(f"{'grupo':48s} {'hechas':>8s} {'min/tarea':>10s}")
    total_left = 0.0
    for k, (d, n) in rows.items():
        m = sum(mins.get(k, [])) / len(mins[k]) if mins.get(k) else float("nan")
        print(f"{k:48s} {d:>3d}/{n:<4d} {m:>10.1f}")
        if m == m:
            total_left += m * (n - d)
    active = [p.stem for p in cfg.DRIVE_CLAIMS.glob("*.json")]
    print(f"\nEn curso: {active or '-'}")
    print(f"Tiempo restante estimado (solo grupos con tareas ya medidas): {total_left / 60:.1f} h de GPU")
    pend = list((cfg.DRIVE_WEIGHTS / "pendientes").rglob("pendiente.json")) if cfg.DRIVE_WEIGHTS.exists() else []
    if pend:
        print(f"Corridas con evaluación ampliada pendiente: {len(pend)} (ejecuta extra-pass cuando estén los datos)")


def extra_pass(worker: str = "0") -> None:
    """Reejecuta el gancho de evaluación de las corridas con conjuntos adicionales pendientes."""
    rc = run_task({"fam": "e2", "kind": "prep", "model": "-", "id": "prep-e2"}, worker)
    if rc != EXIT_OK:
        return
    todo = []
    for f in (cfg.DRIVE_WEIGHTS).rglob("pendiente.json") if cfg.DRIVE_WEIGHTS.exists() else []:
        todo.append(json.loads(f.read_text(encoding="utf-8"))["tarea"])
    for fam in cfg.RESULTS:
        for h in cfg.RESULTS[fam].rglob("E3_HOOK.json"):
            info = json.loads(h.read_text(encoding="utf-8"))
            if info.get("faltan_extra") and info["tarea"] not in todo:
                todo.append(info["tarea"])
    S.log_line(f"extra-pass: {len(todo)} corridas con evaluación pendiente")
    for t in todo:
        run_task(t, worker)


PARAMS_M = {"NAGFuse-B1": 76.6, "NAGFuse-B2": 52.8, "NAGFuse-B3": 113.6}   # millones aprox. (cabezas de 6 clases), para desempatar


def piloto() -> int:
    """Resume el piloto de la opción B (solo validación) y aplica la regla del protocolo v4."""
    import pickle
    filas = []
    for m in cfg.CANDIDATOS_B:
        t = {"fam": "nag", "kind": "hpo", "model": m, "trial": cfg.PILOTO_TRIAL}
        run = run_path(t)
        f = run / "result.pkl"
        if not f.exists():
            f = cfg.DRIVE_RESULTS["e2"] / run.relative_to(cfg.RESULTS["e2"]) / "result.pkl"
        if not f.exists():
            print(f"{m}: sin resultado todavía ({f})")
            continue
        r = pickle.load(open(f, "rb"))
        filas.append({"candidato": m, "f1_val": float(r["selection_val_f1_macro"]), "mejor_epoca": int(r["best_epoch"]),
                      "epocas": int(r["epochs_run"]), "minutos": round(r["train_time_s"] / 60, 1),
                      "params_M": PARAMS_M[m]})
    for r in filas:
        print(f"{r['candidato']:12s} F1 val {r['f1_val']:.4f} | época {r['mejor_epoca']}/{r['epocas']} | "
              f"{r['minutos']} min | {r['params_M']} M parámetros")
    print(f"Referencia: versión A en el mismo intento = {cfg.REF_PILOTO_A:.4f}")
    if len(filas) < len(cfg.CANDIDATOS_B):
        print("Piloto incompleto: no se aplica la regla todavía.")
        return 1
    mejor = max(r["f1_val"] for r in filas)
    empatados = [r for r in filas if mejor - r["f1_val"] < 0.002]          # regla: empate dentro de 0.002 -> menos parámetros
    elegido = min(empatados, key=lambda r: r["params_M"])
    lanzar = elegido["f1_val"] > cfg.REF_PILOTO_A
    res = {"filas": filas, "referencia_A": cfg.REF_PILOTO_A, "elegido": elegido["candidato"], "lanzar": lanzar,
           "regla": "máximo F1 de validación; empate < 0.002 -> menos parámetros; se lanza solo si supera a A",
           "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
    S.write_json_atomic(res, cfg.DRIVE / "piloto_opcion_B.json")
    print(f"\nELEGIDO: {elegido['candidato']} | ¿se lanza la fase confirmatoria? {'SÍ' if lanzar else 'NO'}")
    print(f"Guardado en {cfg.DRIVE / 'piloto_opcion_B.json'}. Siguiente paso: fijar ELEGIDO_B = "
          f"\"{elegido['candidato']}\" en e3/cfg.py, registrar en design_log.md y crear el tag e3-protocolo-v4.")
    return 0


def smoke() -> int:
    os.environ["E3_SMOKE"] = "1"
    import importlib
    importlib.reload(cfg)
    importlib.reload(S)
    setup()
    if prepare() != EXIT_OK:
        return 1
    q = cfg.smoke_queue()
    run("smoke", hours=2, queue=q)
    ok = all(_is_finished(t) for t in q)
    S.log_line("PRUEBA DE HUMO " + ("SUPERADA" if ok else "CON FALLAS: revisa los errores antes de lanzar"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "prepare", "run", "status", "extra-pass", "smoke", "piloto"])
    ap.add_argument("--worker", default="0")
    ap.add_argument("--hours", type=float, default=1e9)
    ap.add_argument("--only", choices=["nag", "e2", "e1"], default=None)
    a = ap.parse_args()
    if a.cmd == "smoke":
        return smoke()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "prepare":
        return prepare(a.worker)
    elif a.cmd == "run":
        return run(a.worker, a.hours, only=a.only)
    elif a.cmd == "status":
        status()
    elif a.cmd == "extra-pass":
        extra_pass(a.worker)
    elif a.cmd == "piloto":
        return piloto()
    return 0


if __name__ == "__main__":
    sys.exit(main())
