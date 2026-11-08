"""Datos: partición v2 del EDA, caché 224x224, normalización y DataLoaders.

Decisiones heredadas del EDA (Sección 2):
- La partición es `data/splits/trashnet_split_v2_s1234.csv` (1800/360/361),
  estratificada y agrupada por casi-duplicados, sin las 6 imágenes con etiqueta
  contradictoria. Aquí NO se vuelve a partir: se carga y se verifica la huella del test.
- Todas las imágenes miden 512x384; se redimensionan a 224x224 (misma deformación).

Para que el entrenamiento no dependa del disco ni de decodificar JPEG en cada
época, las 2521 imágenes se guardan una vez como un arreglo uint8 (N, 224, 224, 3)
en `data/interim/` (~380 MB) y se leen con memmap.

Reproducibilidad de la aumentación: cada imagen recibe en cada época una semilla
propia (EpochSampler). El resultado es idéntico con 0 o con N workers y al
reanudar tras un corte.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2

from . import config as C
from .seeds import seed_worker

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ partición del EDA
def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def split_fingerprint(df: pd.DataFrame, which: str = "test") -> str:
    """Misma huella que el EDA: SHA-256 de (ruta, hash) ordenados por ruta."""
    sub = df[df["split"] == which].sort_values("relpath")
    payload = "".join(f"{r}\t{h}\n" for r, h in zip(sub["relpath"], sub["sha256"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registered_test_fingerprint() -> str | None:
    if not C.SUMS.exists():
        return None
    key = f"test_set:data/splits/{C.SPLIT_CSV.name}"
    for line in C.SUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == key:
            return parts[0]
    return None


def load_split(verify: bool = True) -> pd.DataFrame:
    if not C.SPLIT_CSV.exists():
        raise FileNotFoundError(f"No existe {C.SPLIT_CSV}. Ejecuta primero el EDA (Sección 2).")
    df = pd.read_csv(C.SPLIT_CSV)
    need = {"relpath", "label", "label_id", "sha256", "group", "split"}
    if not need <= set(df.columns):
        raise ValueError(f"Faltan columnas en la partición: {need - set(df.columns)}")
    if verify:
        reg = registered_test_fingerprint()
        cur = split_fingerprint(df, "test")
        if reg is None:
            raise RuntimeError("No hay huella del test en data/SHA256SUMS.txt. Ejecuta el EDA.")
        if reg != cur:
            raise RuntimeError("La huella del test no coincide con la del EDA. No continuar.")
        groups_per_split = df.groupby("group")["split"].nunique()
        if (groups_per_split > 1).any():
            raise RuntimeError("Hay grupos de casi-duplicados repartidos entre particiones.")
        for sa, sb in [("train", "val"), ("train", "test"), ("val", "test")]:
            if set(df.loc[df.split == sa, "sha256"]) & set(df.loc[df.split == sb, "sha256"]):
                raise RuntimeError(f"Hay archivos idénticos en {sa} y {sb}.")
    labels = sorted(df["label"].unique())
    if labels != C.CLASSES:
        raise ValueError(f"Clases inesperadas: {labels}")
    return df.reset_index(drop=True)


def find_class_root(base: Path = C.RAW_DIR) -> Path:
    """Carpeta que contiene las 6 subcarpetas de clase (la descompresión puede anidar)."""
    base = Path(base)
    for p in [base, *sorted(base.rglob("*"))]:
        if p.is_dir() and all((p / c).is_dir() for c in C.CLASSES):
            return p
    raise FileNotFoundError(f"No encuentro las carpetas de clase bajo {base}")


# ------------------------------------------------------------------ caché 224x224
def cache_paths(size: int = C.IMG_SIZE) -> tuple[Path, Path]:
    stem = f"trashnet_v2_{size}_uint8"
    return C.INTERIM / f"{stem}.npy", C.INTERIM / f"{stem}.json"


def build_cache(df: pd.DataFrame, size: int = C.IMG_SIZE, force: bool = False, verify_hash: bool = True) -> Path:
    """Redimensiona cada imagen a size x size (bicúbica, con antialias de PIL) y la guarda
    en el orden de las filas del CSV. Verifica el SHA-256 de cada archivo contra la partición."""
    npy, meta = cache_paths(size)
    csv_hash = sha256_file(C.SPLIT_CSV)
    if npy.exists() and meta.exists() and not force:
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("split_csv_sha256") == csv_hash and m.get("n") == len(df) and m.get("size") == size:
            return npy
    root = find_class_root()
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    for i, (rel, h) in enumerate(zip(df["relpath"], df["sha256"])):
        p = root / rel
        if verify_hash and sha256_file(p) != h:
            raise RuntimeError(f"{rel}: el archivo no coincide con el hash de la partición.")
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC))
    arr.flush()
    del arr
    import os
    os.replace(tmp, npy)
    meta.write_text(json.dumps({"split_csv": C.SPLIT_CSV.name, "split_csv_sha256": csv_hash, "n": len(df),
                                "size": size, "interpolation": "PIL bicubic", "method": C.RESIZE_METHOD},
                               indent=2), encoding="utf-8")
    return npy


def load_cache(size: int = C.IMG_SIZE, mmap: bool = True) -> np.ndarray:
    npy, _ = cache_paths(size)
    return np.load(npy, mmap_mode="r" if mmap else None)


def channel_stats(cache: np.ndarray, idx: np.ndarray, chunk: int = 256) -> dict:
    """Media y DE por canal (escala [0,1]) solo con las filas idx (train)."""
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for k in range(0, len(idx), chunk):
        x = cache[np.sort(idx[k:k + chunk])].astype(np.float64) / 255.0
        s += x.sum(axis=(0, 1, 2))
        s2 += (x ** 2).sum(axis=(0, 1, 2))
        n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return {"mean": mean.round(5).tolist(), "std": std.round(5).tolist(), "n_pixels": int(n)}


def class_weights(y_train: np.ndarray, k: int = C.NUM_CLASSES) -> np.ndarray:
    """Inversa de la frecuencia, normalizada para que el promedio sea 1: w_k = N / (K n_k)."""
    counts = np.bincount(y_train, minlength=k).astype(float)
    return (len(y_train) / (k * counts)).astype(np.float32)


# ------------------------------------------------------------------ transformaciones
def train_transform(mean, std):
    a = C.AUGMENTATION
    return v2.Compose([
        v2.RandomResizedCrop(a["RandomResizedCrop"]["size"], scale=a["RandomResizedCrop"]["scale"], antialias=True),
        v2.RandomHorizontalFlip(a["RandomHorizontalFlip"]["p"]),
        v2.RandomVerticalFlip(a["RandomVerticalFlip"]["p"]),
        v2.RandomRotation(a["RandomRotation"]["degrees"]),
        v2.ColorJitter(**a["ColorJitter"]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])


def eval_transform(mean, std):
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))])


# ------------------------------------------------------------------ Dataset y Sampler
class CachedImages(Dataset):
    """Lee del arreglo uint8 en memmap. Recibe un índice entero (evaluación) o una
    tupla (posición, semilla) desde EpochSampler (entrenamiento)."""

    def __init__(self, cache_file, rows, labels, transform):
        self.cache_file = str(cache_file)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self._arr = None                   # se abre de forma perezosa en cada proceso

    def __len__(self):
        return len(self.rows)

    def _array(self):
        if self._arr is None:
            self._arr = np.load(self.cache_file, mmap_mode="r")
        return self._arr

    def __getstate__(self):               # no se serializa el memmap al crear workers
        st = self.__dict__.copy()
        st["_arr"] = None
        return st

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            pos, aug_seed = int(item[0]), int(item[1])
        else:
            pos, aug_seed = int(item), None
        img = torch.from_numpy(np.array(self._array()[self.rows[pos]])).permute(2, 0, 1)   # uint8 (3,H,W)
        if aug_seed is None:
            x = self.transform(img)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aug_seed)
                x = self.transform(img)
        return x, int(self.labels[pos]), pos


class EpochSampler(Sampler):
    """Orden aleatorio y semillas de aumentación determinados solo por (semilla, época)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = n, int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(self.n, generator=g).tolist()
        aug = torch.randint(0, 2**62, (self.n,), generator=g).tolist()
        return iter([(i, aug[i]) for i in order])


def make_loader(ds, batch_size, sampler=None, num_workers=None, seed=0):
    nw = C.NUM_WORKERS if num_workers is None else num_workers
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
        worker_init_fn=seed_worker if nw > 0 else None,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )


# ------------------------------------------------------------------ archivos comunes
def prepare_common(force: bool = False, verify_hash: bool = True) -> dict:
    """Crea results/common/: splits.pkl, norm_stats.pkl, class_weights.pkl, class_names.pkl.
    Si ya existen y coinciden con la partición, solo los carga."""
    from .persist import load_pickle, save_pickle
    C.ensure_dirs()
    df = load_split(verify=True)
    cache = build_cache(df, force=force, verify_hash=verify_hash)
    f_splits = C.COMMON / "splits.pkl"
    if f_splits.exists() and not force:
        sp = load_pickle(f_splits)
        if sp.get("test_fingerprint") == split_fingerprint(df, "test"):
            return {"splits": sp, "norm": load_pickle(C.COMMON / "norm_stats.pkl"),
                    "class_weights": load_pickle(C.COMMON / "class_weights.pkl"), "cache": str(cache)}
    sp = {"split_csv": C.SPLIT_CSV.name, "test_fingerprint": split_fingerprint(df, "test"),
          "split_seed": 1234, "classes": C.CLASSES, "removed_contradictory": 6,
          "n_groups": int(df["group"].nunique())}
    for s in SPLITS:
        sub = df[df.split == s]
        sp[s] = {"rows": sub.index.to_numpy(np.int64), "relpath": sub["relpath"].tolist(),
                 "y": sub["label_id"].to_numpy(np.int64), "sha256": sub["sha256"].tolist(),
                 "group": sub["group"].to_numpy(np.int64)}
    arr = load_cache()
    norm = channel_stats(arr, sp["train"]["rows"])
    norm.update({"computed_on": "train", "size": C.IMG_SIZE})
    cw = {"weights": class_weights(sp["train"]["y"]), "classes": C.CLASSES,
          "counts_train": np.bincount(sp["train"]["y"], minlength=C.NUM_CLASSES)}
    save_pickle(sp, f_splits)
    save_pickle(norm, C.COMMON / "norm_stats.pkl")
    save_pickle(cw, C.COMMON / "class_weights.pkl")
    save_pickle(C.CLASSES, C.COMMON / "class_names.pkl")
    return {"splits": sp, "norm": norm, "class_weights": cw, "cache": str(cache)}


def subset_rows(sp_split: dict, n: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Submuestra estratificada aproximada para la prueba de humo (n=None devuelve todo)."""
    rows, y = sp_split["rows"], sp_split["y"]
    if n is None or n >= len(rows):
        return rows, y
    rng = np.random.default_rng(seed)
    pick = []
    per = max(1, n // C.NUM_CLASSES)
    for k in range(C.NUM_CLASSES):
        ids = np.flatnonzero(y == k)
        pick.extend(rng.choice(ids, size=min(per, len(ids)), replace=False))
    pick = np.sort(np.array(pick))
    return rows[pick], y[pick]
