"""Persistencia en Google Drive: espejo de checkpoints, restauración tras un corte y reservas de tareas.

Principios:
- Todo archivo se escribe primero como .tmp y luego se renombra (os.replace): o está completo o no existe.
- El checkpoint de la época en curso se guarda en disco local cada época. Si la corrida dura más de
  MIRROR_EVERY_S segundos, se copia también a Drive con esa periodicidad. Un corte de la máquina virtual
  pierde como máximo la corrida en curso (o MIRROR_EVERY_S en las largas); nunca una corrida terminada.
- Las corridas terminadas (DONE) se copian a Drive sin pesos; los pesos solo se conservan según KEEP_WEIGHTS.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
from pathlib import Path

from . import cfg

SKIP_SUFFIX = (".tmp", ".lock", ".part")


def log_line(msg: str, name: str = "orquestador") -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    for d in (cfg.WORK / "logs", cfg.DRIVE_LOGS):
        try:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{name}_{time.strftime('%Y-%m-%d')}.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def write_json_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ resultados terminados
def sync_run_files(run: Path, fam: str, include_weights: bool = False) -> None:
    """Copia a Drive los archivos de una corrida (sin checkpoints ni, por defecto, pesos)."""
    rel = run.relative_to(cfg.RESULTS[fam])
    dst = cfg.DRIVE_RESULTS[fam] / rel
    for f in run.iterdir():
        if not f.is_file() or f.name.endswith(SKIP_SUFFIX) or f.name.startswith("checkpoint_"):
            continue
        if f.suffix == ".pt" and not include_weights:
            continue
        try:
            if not (dst / f.name).exists() or (dst / f.name).stat().st_size != f.stat().st_size \
                    or f.stat().st_mtime > (dst / f.name).stat().st_mtime:
                copy_atomic(f, dst / f.name)
        except OSError as e:
            log_line(f"[drive] no se pudo copiar {f}: {e}")


def sync_tree(src: Path, fam: str) -> None:
    if not src.exists():
        return
    for f in src.rglob("*"):
        if f.is_file() and not f.name.endswith(SKIP_SUFFIX) and not f.name.startswith("checkpoint_") \
                and f.suffix != ".pt":
            dst = cfg.DRIVE_RESULTS[fam] / f.relative_to(cfg.RESULTS[fam])
            try:
                if not dst.exists() or dst.stat().st_size != f.stat().st_size:
                    copy_atomic(f, dst)
            except OSError as e:
                log_line(f"[drive] no se pudo copiar {f}: {e}")


def restore_results(fam: str) -> int:
    """Drive -> disco local (al iniciar una sesión). Devuelve el número de archivos restaurados."""
    src, dst = cfg.DRIVE_RESULTS[fam], cfg.RESULTS[fam]
    n = 0
    if not src.exists():
        return 0
    for f in src.rglob("*"):
        if f.is_file() and not f.name.endswith(SKIP_SUFFIX):
            t = dst / f.relative_to(src)
            if not t.exists() or t.stat().st_size != f.stat().st_size:
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += 1
    return n


# ------------------------------------------------------------------ espejo del checkpoint en curso
def mirror_dir(run: Path, fam: str) -> Path:
    return cfg.DRIVE_CKPT / fam / run.relative_to(cfg.RESULTS[fam])


_last_mirror: dict[str, float] = {}


def mirror_checkpoint(run: Path, fam: str, ckpt: Path, force: bool = False) -> bool:
    key = str(run)
    now = time.time()
    if key not in _last_mirror:               # el reloj empieza con la primera época de este proceso
        _last_mirror[key] = now
        if not force:
            return False
    if not force and now - _last_mirror[key] < cfg.MIRROR_EVERY_S:
        return False
    md = mirror_dir(run, fam)
    try:
        for name in ("config.pkl", "config.json", "env.pkl", "model_info.pkl", "history.pkl"):
            if (run / name).exists():
                copy_atomic(run / name, md / name)
        copy_atomic(ckpt, md / "checkpoint_last.pt")
        write_json_atomic({"copiado": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()},
                          md / "espejo.json")
        _last_mirror[key] = now
        return True
    except OSError as e:
        log_line(f"[drive] no se pudo reflejar el checkpoint de {run}: {e}")
        return False


def restore_checkpoint(run: Path, fam: str) -> bool:
    """Si la corrida no tiene DONE en local y Drive tiene un espejo, lo trae para que train_run reanude."""
    if (run / "DONE").exists():
        return False
    md = mirror_dir(run, fam)
    ck = md / "checkpoint_last.pt"
    if not ck.exists():
        return False
    local = run / "checkpoint_last.pt"
    if local.exists() and local.stat().st_mtime >= ck.stat().st_mtime:
        return False
    run.mkdir(parents=True, exist_ok=True)
    for f in md.iterdir():
        if f.is_file() and f.name != "espejo.json" and not f.name.endswith(".tmp"):
            shutil.copy2(f, run / f.name)
    return True


def drop_mirror(run: Path, fam: str) -> None:
    shutil.rmtree(mirror_dir(run, fam), ignore_errors=True)


# ------------------------------------------------------------------ reservas (varias sesiones a la vez)
def _claim_path(tid: str) -> Path:
    return cfg.DRIVE_CLAIMS / f"{tid}.json"


def task_done(tid: str) -> bool:
    return (cfg.DRIVE_CLAIMS / f"{tid}.done").exists()


def mark_done(tid: str, info: dict) -> None:
    write_json_atomic({**info, "terminado": time.strftime("%Y-%m-%d %H:%M:%S")}, cfg.DRIVE_CLAIMS / f"{tid}.done")
    _claim_path(tid).unlink(missing_ok=True)


def try_claim(tid: str, worker: str) -> bool:
    p = _claim_path(tid)
    if p.exists():
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            fresh = time.time() - float(c.get("latido", 0)) < cfg.CLAIM_STALE_S
            if fresh and c.get("worker") != worker:
                return False
        except (ValueError, OSError):
            pass
    write_json_atomic({"worker": worker, "host": socket.gethostname(), "latido": time.time(),
                       "desde": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
    return True


def heartbeat(tid: str, worker: str) -> None:
    try:
        write_json_atomic({"worker": worker, "host": socket.gethostname(), "latido": time.time(),
                           "ultimo": time.strftime("%Y-%m-%d %H:%M:%S")}, _claim_path(tid))
    except OSError:
        pass


def release_claim(tid: str) -> None:
    _claim_path(tid).unlink(missing_ok=True)
