"""Conjuntos adicionales (evaluación sin reentrenar): descarga, mapeo de clases, casi duplicados y caché 224x224.

Cada conjunto queda en EXTRA_DIR/<nombre>.npz con x (N, 224, 224, 3) uint8, y (N,) en el orden de CLASSES,
y un <nombre>.json con la procedencia, los conteos, la regla de mapeo y los casi duplicados retirados.
El redimensionado es el mismo de TrashNet en el E2: PIL bicúbico directo a 224x224.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import cfg

IMG = 224
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def warp_label(leaf: str) -> str | None:
    """Regla declarada en el protocolo para WaRP-C (28 categorías -> materiales de TrashNet)."""
    s = leaf.lower()
    if s.startswith("glass"):
        return "glass"
    if s.startswith("cans"):
        return "metal"
    if "cardboard" in s or s == "detergent-box":
        return "cardboard"
    if s.startswith(("bottle", "detergent", "canister")):
        return "plastic"
    return None


def _download(url: str, dst: Path, log=print) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"Descargando {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    os.replace(tmp, dst)
    return dst


def _kaggle(slug: str, log=print) -> Path:
    import kagglehub
    log(f"Descargando {slug} con kagglehub")
    return Path(kagglehub.dataset_download(slug))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_images(name: str, root: Path) -> list[tuple[Path, str]]:
    spec = cfg.EXTRA_DATASETS[name]
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in EXTS or "__MACOSX" in p.parts:
            continue
        if name == "warp_c":
            if "test_crops" not in [q.lower() for q in p.parts]:
                continue
            lab = warp_label(p.parent.name)
        else:
            lab = spec["mapa"].get(p.parent.name)
            if lab is None:                      # GD v2 usa minúsculas; RealWaste, mayúscula inicial
                lab = {k.lower(): v for k, v in spec["mapa"].items()}.get(p.parent.name.lower())
        if lab is not None:
            out.append((p, lab))
    return out


def dhash(arr224: np.ndarray) -> np.ndarray:
    """dHash de 64 bits (Krawetz) sobre imágenes (N, 224, 224, 3) uint8 -> (N,) uint64."""
    out = np.empty(len(arr224), dtype=np.uint64)
    w = (1 << np.arange(64, dtype=np.uint64)).astype(np.uint64)
    for i, a in enumerate(arr224):
        g = np.asarray(Image.fromarray(a).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).ravel().astype(np.uint64)
        out[i] = np.bitwise_or.reduce(bits * w)
    return out


def hamming_min(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distancia de Hamming mínima de cada hash de a a cualquiera de b."""
    res = np.empty(len(a), dtype=np.int16)
    bb = b[None, :]
    for k in range(0, len(a), chunk):
        x = np.bitwise_xor(a[k:k + chunk, None], bb)
        cnt = np.zeros(x.shape, dtype=np.int16)
        for s in range(64):
            cnt += ((x >> np.uint64(s)) & np.uint64(1)).astype(np.int16)
        res[k:k + chunk] = cnt.min(1)
    return res


def build(name: str, trashnet_cache: np.ndarray, log=print) -> Path:
    """Descarga (o reutiliza la copia de Drive), mapea, retira casi duplicados de TrashNet y guarda la caché."""
    spec = cfg.EXTRA_DATASETS[name]
    out = cfg.EXTRA_DIR / f"{name}.npz"
    meta_p = cfg.EXTRA_DIR / f"{name}.json"
    drive_npz, drive_meta = cfg.DRIVE_EXTRA / out.name, cfg.DRIVE_EXTRA / meta_p.name
    if out.exists() and meta_p.exists():
        return out
    if drive_npz.exists() and drive_meta.exists():
        cfg.EXTRA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_npz, out)
        shutil.copy2(drive_meta, meta_p)
        return out
    raw = cfg.WORK_BASE / "extra_raw" / name
    manual = cfg.DRIVE_BASE / "manual" / f"{name}.zip"         # alternativa: zip subido a mano a Drive
    meta = {"nombre": name, **{k: v for k, v in spec.items() if k != "mapa"},
            "regla_mapeo": spec["mapa"], "fecha_descarga": time.strftime("%Y-%m-%d %H:%M")}
    if manual.exists():
        root = raw / "manual"
        if not root.exists():
            with zipfile.ZipFile(manual) as z:
                z.extractall(root)
        meta["origen_archivo"] = str(manual.name)
        meta["sha256_zip"] = _sha256(manual)
    elif "url" in spec:
        z = _download(spec["url"], cfg.WORK_BASE / "extra_raw" / f"{name}.zip", log)
        meta["sha256_zip"] = _sha256(z)
        root = raw
        if not root.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)
        for inner in list(root.rglob("*.zip")):          # RealWaste trae un zip dentro del zip
            dst = inner.with_suffix("")
            if not dst.exists():
                with zipfile.ZipFile(inner) as zf:
                    zf.extractall(dst)
    else:
        root = _kaggle(spec["kaggle"], log)
        meta["kaggle_path"] = str(root)
    items = list_images(name, root)
    if not items:
        raise RuntimeError(f"{name}: no se encontraron imágenes con la regla de mapeo bajo {root}")
    labels = np.array([cfg.CLASSES.index(l) for _, l in items])
    meta["conteo_mapeado"] = {c: int((labels == i).sum()) for i, c in enumerate(cfg.CLASSES)}
    arr = np.empty((len(items), IMG, IMG, 3), dtype=np.uint8)
    keep = np.ones(len(items), bool)
    for i, (p, _) in enumerate(items):
        try:
            with Image.open(p) as im:
                arr[i] = np.asarray(im.convert("RGB").resize((IMG, IMG), Image.BICUBIC))
        except Exception as e:                           # imagen dañada: se descarta y se registra
            keep[i] = False
            log(f"  {name}: no se pudo leer {p.name} ({e})")
    meta["ilegibles"] = int((~keep).sum())
    # casi duplicados con cualquier imagen de TrashNet (train, val o test)
    hd = hamming_min(dhash(arr), dhash(np.asarray(trashnet_cache)))
    dup = (hd <= cfg.DEDUP_HAMMING) & keep
    meta["casi_duplicados_trashnet"] = int(dup.sum())
    keep &= ~dup
    # submuestreo estratificado con semilla fija
    rng = np.random.default_rng(cfg.EXTRA_SEED)
    sel = []
    for k in range(len(cfg.CLASSES)):
        ids = np.flatnonzero(keep & (labels == k))
        if len(ids) > cfg.EXTRA_MAX_PER_CLASS:
            ids = np.sort(rng.choice(ids, cfg.EXTRA_MAX_PER_CLASS, replace=False))
        sel.extend(ids.tolist())
    sel = np.array(sorted(sel))
    meta["conteo_final"] = {c: int((labels[sel] == i).sum()) for i, c in enumerate(cfg.CLASSES)}
    meta["n_final"] = int(len(sel))
    meta["clases_presentes"] = [c for c in cfg.CLASSES if meta["conteo_final"][c] > 0]
    cfg.EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    rel = [str(items[i][0].relative_to(root)) for i in sel]
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, x=arr[sel], y=labels[sel].astype(np.int64), relpath=np.array(rel))
    os.replace(tmp, out)
    meta["sha256_cache"] = _sha256(out)
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg.DRIVE_EXTRA.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, drive_npz)
    shutil.copy2(meta_p, drive_meta)
    log(f"{name}: {meta['n_final']} imágenes, {meta['casi_duplicados_trashnet']} casi duplicados retirados, "
        f"clases {meta['conteo_final']}")
    return out


def available() -> dict[str, Path]:
    return {n: cfg.EXTRA_DIR / f"{n}.npz" for n in cfg.EXTRA_DATASETS if (cfg.EXTRA_DIR / f"{n}.npz").exists()}


def load(name: str) -> dict:
    z = np.load(cfg.EXTRA_DIR / f"{name}.npz")
    meta = json.loads((cfg.EXTRA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return {"x": z["x"], "y": z["y"], "meta": meta}
