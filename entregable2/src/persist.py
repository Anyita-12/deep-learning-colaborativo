"""Escritura atómica de resultados: cada archivo queda completo o no existe."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import socket
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NEUTRAL_TYPES = (dict, list, tuple, str, int, float, bool, type(None), np.ndarray, np.generic, pd.DataFrame, pd.Series)


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
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_neutral(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_neutral(v, f"{where}[{i}]")
    elif not isinstance(obj, NEUTRAL_TYPES):
        raise TypeError(f"{where} es {type(obj).__name__}; se esperan tipos neutros (numpy, pandas, nativos).")


def _write_atomic(path: Path, writer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)
    return path


def save_pickle(obj, path) -> Path:
    _check_neutral(obj)
    return _write_atomic(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


SAFE_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "str", "bytes", "bytearray", "int", "float", "complex",
                 "bool", "slice", "range", "object"}
SAFE_PREFIXES = ("numpy", "pandas", "collections", "datetime", "copyreg", "_codecs")
SAFE_CLASSES = {("torch.torch_version", "TorchVersion")}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if ((module == "builtins" and name in SAFE_BUILTINS) or module.split(".")[0] in SAFE_PREFIXES
                or (module, name) in SAFE_CLASSES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Tipo no permitido en el pickle: {module}.{name}")


def load_pickle(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def save_json(obj, path) -> Path:
    data = json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    return _write_atomic(path, lambda f: f.write(data))


def save_text(text: str, path) -> Path:
    return _write_atomic(path, lambda f: f.write(text.encode("utf-8")))


def save_csv(df: pd.DataFrame, path) -> Path:
    data = df.to_csv(index=False).encode("utf-8")
    return _write_atomic(path, lambda f: f.write(data))


def save_npz(path, **arrays) -> Path:
    return _write_atomic(path, lambda f: np.savez_compressed(f, **arrays))


def save_torch(obj, path, keep_previous: bool = False) -> Path:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as z:
        if not z.namelist():
            raise IOError(f"Checkpoint vacío: {tmp}")
    if keep_previous and path.exists():
        _replace_with_retry(path, path.with_name(path.stem + "_prev" + path.suffix))
    _replace_with_retry(tmp, path)
    return path


def load_torch(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def touch_done(run_dir) -> Path:
    return save_text(time.strftime("%Y-%m-%d %H:%M:%S"), Path(run_dir) / "DONE")


def is_done(run_dir) -> bool:
    return (Path(run_dir) / "DONE").exists()


def backup_run(run_dir, backup_root, results_root, include_weights: bool = False) -> Path | None:
    if not backup_root:
        return None
    run_dir, results_root = Path(run_dir), Path(results_root)
    dst = Path(backup_root) / run_dir.relative_to(results_root)
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for f in run_dir.iterdir():
            if not f.is_file() or f.suffix in (".tmp", ".lock") or f.name.startswith("checkpoint_"):
                continue
            if f.suffix == ".pt" and not include_weights:
                continue
            shutil.copy2(f, dst / f.name)
        return dst
    except OSError as e:
        print(f"[respaldo] No se pudo copiar {run_dir}: {e}")
        return None


def backup_tree(src, backup_root, results_root) -> None:
    if not backup_root:
        return
    src, results_root = Path(src), Path(results_root)
    if not src.exists():
        return
    for f in src.rglob("*"):
        if f.is_file() and f.suffix not in (".tmp", ".lock") and not f.name.startswith("checkpoint_"):
            dst = Path(backup_root) / f.relative_to(results_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(f, dst)
            except OSError as e:
                print(f"[respaldo] No se pudo copiar {f}: {e}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(run_dir) -> bool:
    p = Path(run_dir) / "RUNNING.lock"
    if p.exists():
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            pid = int(info.get("pid", -1))
            if info.get("host") == socket.gethostname() and pid != os.getpid() and _pid_alive(pid):
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
