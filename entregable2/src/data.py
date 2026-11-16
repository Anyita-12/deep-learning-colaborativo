"""Datos: descarga verificada de TrashNet, partición v2 del Entregable 1, caché 224x224 y DataLoaders."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
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


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download_dataset(log=print) -> Path:
    """Descarga dataset-resized.zip del repositorio oficial y lo descomprime en data/raw."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not C.DATASET_ZIP.exists():
        log(f"Descargando TrashNet desde {C.DATASET_URL}")
        tmp = C.DATASET_ZIP.with_suffix(".tmp")
        with urllib.request.urlopen(C.DATASET_URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, C.DATASET_ZIP)
    target = C.RAW_DIR / "dataset-resized"
    if not target.exists():
        with zipfile.ZipFile(C.DATASET_ZIP) as z:
            members = [m for m in z.namelist() if m.startswith("dataset-resized/") and not m.endswith(".DS_Store")]
            z.extractall(C.RAW_DIR, members=members)
    return target


def find_class_root(base: Path = C.RAW_DIR) -> Path:
    base = Path(base)
    for p in [base, *sorted(base.rglob("*"))]:
        if p.is_dir() and "__MACOSX" not in p.parts and all((p / c).is_dir() for c in C.CLASSES):
            return p
    raise FileNotFoundError(f"No se encuentran las carpetas de clase bajo {base}")


def split_fingerprint(df: pd.DataFrame, which: str = "test") -> str:
    sub = df[df["split"] == which].sort_values("relpath")
    payload = "".join(f"{r}\t{h}\n" for r, h in zip(sub["relpath"], sub["sha256"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registered_test_fingerprint() -> str | None:
    key = f"test_set:data/splits/{C.SPLIT_CSV.name}"
    for line in C.SUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == key:
            return parts[0]
    return None


def load_split() -> pd.DataFrame:
    df = pd.read_csv(C.SPLIT_CSV)
    need = {"relpath", "label", "label_id", "sha256", "group", "split"}
    if not need <= set(df.columns):
        raise ValueError(f"Faltan columnas en la partición: {need - set(df.columns)}")
    if registered_test_fingerprint() != split_fingerprint(df, "test"):
        raise RuntimeError("La huella del conjunto de prueba no coincide con la registrada en el Entregable 1.")
    if (df.groupby("group")["split"].nunique() > 1).any():
        raise RuntimeError("Hay grupos de casi duplicados repartidos entre particiones.")
    for sa, sb in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(df.loc[df.split == sa, "sha256"]) & set(df.loc[df.split == sb, "sha256"]):
            raise RuntimeError(f"Hay archivos idénticos en {sa} y {sb}.")
    if sorted(df["label"].unique()) != C.CLASSES:
        raise ValueError("Las clases de la partición no coinciden con config.CLASSES.")
    return df.reset_index(drop=True)


def _cache_paths(name: str) -> tuple[Path, Path]:
    return C.INTERIM / f"{name}.npy", C.INTERIM / f"{name}.json"


def _build_array(paths: list[Path], hashes: list[str] | None, name: str, meta: dict, force: bool = False) -> Path:
    npy, fmeta = _cache_paths(name)
    if npy.exists() and fmeta.exists() and not force:
        if json.loads(fmeta.read_text(encoding="utf-8")).get("key") == meta["key"]:
            return npy
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), C.IMG_SIZE, C.IMG_SIZE, 3))
    for i, p in enumerate(paths):
        if hashes is not None and sha256_file(p) != hashes[i]:
            raise RuntimeError(f"{p}: el archivo no coincide con el hash de la partición.")
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BICUBIC))
    arr.flush()
    del arr
    os.replace(tmp, npy)
    fmeta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npy


def build_cache(df: pd.DataFrame, force: bool = False) -> Path:
    root = find_class_root()
    meta = {"key": sha256_file(C.SPLIT_CSV), "split_csv": C.SPLIT_CSV.name, "n": len(df), "size": C.IMG_SIZE,
            "interpolation": "PIL bicubic", "method": C.RESIZE_METHOD}
    return _build_array([root / r for r in df["relpath"]], df["sha256"].tolist(), f"trashnet_v2_{C.IMG_SIZE}_uint8",
                        meta, force)


def full_dataset_table() -> pd.DataFrame:
    """Las 2527 imágenes originales (incluidas las 6 excluidas en la partición v2), en orden de clase y nombre."""
    root = find_class_root()
    rows = []
    for k, c in enumerate(C.CLASSES):
        for p in sorted((root / c).glob("*.jpg"), key=lambda q: (len(q.stem), q.stem)):
            rows.append({"relpath": f"{c}/{p.name}", "label": c, "label_id": k})
    return pd.DataFrame(rows)


def build_full_cache(force: bool = False) -> tuple[Path, pd.DataFrame]:
    root = find_class_root()
    df = full_dataset_table()
    key = hashlib.sha256("".join(df["relpath"]).encode()).hexdigest()
    npy = _build_array([root / r for r in df["relpath"]], None, f"trashnet_full_{C.IMG_SIZE}_uint8",
                       {"key": key, "n": len(df), "size": C.IMG_SIZE}, force)
    return npy, df


def load_cache(name: str | None = None, mmap: bool = True) -> np.ndarray:
    npy, _ = _cache_paths(name or f"trashnet_v2_{C.IMG_SIZE}_uint8")
    return np.load(npy, mmap_mode="r" if mmap else None)


def channel_stats(cache: np.ndarray, idx: np.ndarray, chunk: int = 256) -> dict:
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for k in range(0, len(idx), chunk):
        x = cache[np.sort(idx[k:k + chunk])].astype(np.float64) / 255.0
        s += x.sum(axis=(0, 1, 2))
        s2 += (x ** 2).sum(axis=(0, 1, 2))
        n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s / n
    return {"mean": mean.round(5).tolist(), "std": np.sqrt(s2 / n - mean ** 2).round(5).tolist(), "n_pixels": int(n)}


def class_weights(y_train: np.ndarray, k: int = C.NUM_CLASSES) -> np.ndarray:
    counts = np.bincount(y_train, minlength=k).astype(float)
    return (len(y_train) / (k * counts)).astype(np.float32)


def normalization(common: dict, kind: str) -> tuple[list, list]:
    if kind == "imagenet":
        return C.IMAGENET_MEAN, C.IMAGENET_STD
    return common["norm"]["mean"], common["norm"]["std"]


def train_transform(mean, std, augment: bool = True):
    a = C.AUGMENTATION
    ops = []
    if augment:
        ops = [
            v2.RandomResizedCrop(a["RandomResizedCrop"]["size"], scale=a["RandomResizedCrop"]["scale"], antialias=True),
            v2.RandomHorizontalFlip(a["RandomHorizontalFlip"]["p"]),
            v2.RandomVerticalFlip(a["RandomVerticalFlip"]["p"]),
            v2.RandomRotation(a["RandomRotation"]["degrees"]),
            v2.ColorJitter(**a["ColorJitter"]),
        ]
    return v2.Compose(ops + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))])


def eval_transform(mean, std):
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))])


class CachedImages(Dataset):
    def __init__(self, cache_file, rows, labels, transform):
        self.cache_file = str(cache_file)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self._arr = None

    def __len__(self):
        return len(self.rows)

    def _array(self):
        if self._arr is None:
            self._arr = np.load(self.cache_file, mmap_mode="r")
        return self._arr

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_arr"] = None
        return st

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            pos, aug_seed = int(item[0]), int(item[1])
        else:
            pos, aug_seed = int(item), None
        img = torch.from_numpy(np.array(self._array()[self.rows[pos]])).permute(2, 0, 1)
        if aug_seed is None:
            x = self.transform(img)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(aug_seed)
                x = self.transform(img)
        return x, int(self.labels[pos]), pos


class EpochSampler(Sampler):
    """Orden y semillas de aumentación determinados solo por (semilla, época)."""

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
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
                      worker_init_fn=seed_worker if nw > 0 else None,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def prepare_common(force: bool = False, log=print) -> dict:
    from .persist import load_pickle, save_pickle
    C.ensure_dirs()
    download_dataset(log)
    df = load_split()
    cache = build_cache(df, force=force)
    f_splits = C.COMMON / "splits.pkl"
    if f_splits.exists() and not force:
        sp = load_pickle(f_splits)
        if sp.get("test_fingerprint") == split_fingerprint(df, "test"):
            return {"splits": sp, "norm": load_pickle(C.COMMON / "norm_stats.pkl"),
                    "class_weights": load_pickle(C.COMMON / "class_weights.pkl"), "cache": str(cache)}
    sp = {"split_csv": C.SPLIT_CSV.name, "test_fingerprint": split_fingerprint(df, "test"), "split_seed": 1234,
          "classes": C.CLASSES, "removed_contradictory": 6, "n_groups": int(df["group"].nunique()),
          "dataset_zip_sha256": sha256_file(C.DATASET_ZIP)}
    for s in SPLITS:
        sub = df[df.split == s]
        sp[s] = {"rows": sub.index.to_numpy(np.int64), "relpath": sub["relpath"].tolist(),
                 "y": sub["label_id"].to_numpy(np.int64), "sha256": sub["sha256"].tolist(),
                 "group": sub["group"].to_numpy(np.int64)}
    norm = channel_stats(load_cache(), sp["train"]["rows"])
    norm.update({"computed_on": "train", "size": C.IMG_SIZE})
    cw = {"weights": class_weights(sp["train"]["y"]), "classes": C.CLASSES,
          "counts_train": np.bincount(sp["train"]["y"], minlength=C.NUM_CLASSES)}
    save_pickle(sp, f_splits)
    save_pickle(norm, C.COMMON / "norm_stats.pkl")
    save_pickle(cw, C.COMMON / "class_weights.pkl")
    save_pickle(C.CLASSES, C.COMMON / "class_names.pkl")
    return {"splits": sp, "norm": norm, "class_weights": cw, "cache": str(cache)}


def subset_rows(sp_split: dict, n: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rows, y = sp_split["rows"], sp_split["y"]
    if n is None or n >= len(rows):
        return rows, y
    rng = np.random.default_rng(seed)
    per = max(1, n // C.NUM_CLASSES)
    pick = []
    for k in range(C.NUM_CLASSES):
        ids = np.flatnonzero(y == k)
        pick.extend(rng.choice(ids, size=min(per, len(ids)), replace=False))
    pick = np.sort(np.array(pick))
    return rows[pick], y[pick]
