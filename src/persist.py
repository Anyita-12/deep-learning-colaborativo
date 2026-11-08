"""Persistencia atómica y verificada.

Regla: un archivo de resultados o está completo o no existe. Se escribe en
`<archivo>.tmp`, se fuerza al disco (flush + fsync) y se renombra con
os.replace, que es atómico en Windows y en Linux. Un corte de luz deja, como
mucho, un .tmp huérfano que se ignora.

En Windows, OneDrive o el antivirus pueden bloquear un archivo un instante;
por eso os.replace se reintenta.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NEUTRAL_TYPES = (dict, list, tuple, str, int, float, bool, type(None), np.ndarray, np.generic, pd.DataFrame, pd.Series)


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":            # Windows no permite abrir directorios para fsync
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _replace_with_retry(src: Path, dst: Path, tries: int = 10, wait: float = 0.5) -> None:
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def _check_neutral(obj, where: str = "obj") -> None:
    """Rechaza tensores, módulos u objetos propios dentro de un .pkl de resultados."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_neutral(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_neutral(v, f"{where}[{i}]")
    elif not isinstance(obj, NEUTRAL_TYPES):
        raise TypeError(f"{where} es {type(obj).__name__}: convierte a tipos neutros (.detach().cpu().numpy()).")


def _same_structure(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b)
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.shape == b.shape
    return True


def save_pickle(obj, path, verify: bool = True) -> Path:
    path = Path(path)
    _check_neutral(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)
    _fsync_dir(path.parent)
    if verify:
        back = load_pickle(path)
        if not _same_structure(obj, back):
            raise IOError(f"Verificación fallida al recargar {path}")
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)
    return path


def save_text(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)
    return path


def save_torch(obj, path, keep_previous: bool = False, verify_load: bool = False) -> Path:
    """torch.save atómico. Con keep_previous=True, la versión anterior queda como
    `<nombre>_prev.pt`: si el archivo nuevo se dañara, se reanuda desde la anterior."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as z:            # el formato de torch.save es zip: valida el índice
        if not z.namelist():
            raise IOError(f"Checkpoint vacío: {tmp}")
    if verify_load:
        torch.load(tmp, map_location="cpu", weights_only=False)
    if keep_previous and path.exists():
        _replace_with_retry(path, path.with_name(path.stem + "_prev" + path.suffix))
    _replace_with_retry(tmp, path)
    _fsync_dir(path.parent)
    return path


def load_torch(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def touch_done(run_dir) -> Path:
    p = Path(run_dir) / "DONE"
    with open(p, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        f.flush()
        os.fsync(f.fileno())
    return p


def is_done(run_dir) -> bool:
    return (Path(run_dir) / "DONE").exists()


def backup_run(run_dir, backup_root, results_root, include_checkpoint: bool = False) -> Path | None:
    """Copia una corrida terminada a la carpeta sincronizada. Nunca detiene el entrenamiento."""
    if not backup_root:
        return None
    run_dir, results_root = Path(run_dir), Path(results_root)
    dst = Path(backup_root) / run_dir.relative_to(results_root)
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for f in run_dir.iterdir():
            if f.suffix in (".tmp", ".lock") or (not include_checkpoint and f.name.startswith("checkpoint_")):
                continue
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        return dst
    except OSError as e:
        print(f"[respaldo] No se pudo copiar {run_dir.name}: {e}")
        return None


# ------------------------------------------------------------------ bloqueo por corrida
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259            # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(run_dir) -> bool:
    """Evita que dos procesos (dos terminales, o terminal y cuaderno) entrenen la misma
    corrida a la vez. Un bloqueo de un proceso que ya no existe (corte, cierre) se retira."""
    import socket
    p = Path(run_dir) / "RUNNING.lock"
    if p.exists():
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            if info.get("host") == socket.gethostname() and _pid_alive(int(info.get("pid", -1))) \
                    and int(info.get("pid", -1)) != os.getpid():
                return False
        except (ValueError, OSError):
            pass
    save_json({"pid": os.getpid(), "host": socket.gethostname(), "since": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
    return True


def release_lock(run_dir) -> None:
    try:
        (Path(run_dir) / "RUNNING.lock").unlink(missing_ok=True)
    except OSError:
        pass
