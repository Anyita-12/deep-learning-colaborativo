"""Datos de TrashNet: descarga, verificación, índice, duplicados, partición
estratificada por grupos y huella (hash) del conjunto de test.

No depende de TensorFlow: se puede ejecutar y probar sin GPU ni TF.
Fuente: https://huggingface.co/datasets/garythung/trashnet (MIT).
Citar: https://github.com/garythung/trashnet
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

URL_RESIZED = ("https://huggingface.co/datasets/garythung/trashnet/"
               "resolve/main/dataset-resized.zip")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # índice = label_id
EXPECTED_COUNTS = {"cardboard": 403, "glass": 501, "metal": 410,
                   "paper": 594, "plastic": 482, "trash": 137}  # README del repo original
IMG_EXT = {".jpg", ".jpeg", ".png"}
_STR_COLS = {"sha256": str, "dhash": str, "relpath": str, "label": str}


# ----------------------------------------------------------------- hashes
def sha256_file(path, chunk: int = 1 << 20) -> str:
    """SHA-256 de un archivo (huella: si cambia un solo byte, cambia el hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def register_checksum(sums_path, name: str, digest: str) -> str:
    """Registra 'digest  name' en SHA256SUMS.txt.
    Devuelve 'nuevo', 'coincide' o 'DIFERENTE' (si ya había otro hash para ese nombre)."""
    sums_path = Path(sums_path)
    entries = {}
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d, n = line.split(maxsplit=1)
                entries[n.strip()] = d
    if name in entries:
        return "coincide" if entries[name] == digest else "DIFERENTE"
    entries[name] = digest
    sums_path.parent.mkdir(parents=True, exist_ok=True)
    sums_path.write_text("".join(f"{d}  {n}\n" for n, d in entries.items()), encoding="utf-8")
    return "nuevo"


# ------------------------------------------------------ descarga y extracción
def download(dest, url: str = URL_RESIZED, min_mb: float = 30.0) -> Path:
    """Descarga el zip a `dest` (vía archivo .part; solo se renombra si terminó bien)."""
    dest = Path(dest)
    if dest.exists():
        print(f"Ya existe: {dest} ({dest.stat().st_size / 1e6:.1f} MB). No se descarga.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = [-10]

    def hook(n_blocks, block_size, total):
        if total > 0:
            pct = min(100, int(100 * n_blocks * block_size / total))
            if pct >= last[0] + 10:
                last[0] = pct
                print(f"  {pct:3d} % de {total / 1e6:.1f} MB")

    t0 = time.perf_counter()
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo descargar ({e}).\nDescarga manual: abre {url} en el navegador "
            f"y guarda el archivo exactamente como:\n{dest}\nLuego vuelve a ejecutar la celda.") from e
    size_mb = tmp.stat().st_size / 1e6
    if size_mb < min_mb:
        tmp.unlink()
        raise RuntimeError(f"Archivo de {size_mb:.1f} MB (< {min_mb} MB): descarga incompleta "
                           "o página de error. Intenta la descarga manual.")
    os.replace(tmp, dest)
    print(f"Descargado: {dest} ({size_mb:.1f} MB) en {time.perf_counter() - t0:.0f} s")
    return dest


def find_class_root(base) -> Path:
    """Primera carpeta bajo `base` que contiene las 6 subcarpetas de clase."""
    for dirpath, dirnames, _ in os.walk(base):
        if "__MACOSX" in Path(dirpath).parts:
            continue
        if set(CLASSES) <= set(dirnames):
            return Path(dirpath)
    raise FileNotFoundError(f"No hay una carpeta con las clases {CLASSES} dentro de {base}")


def extract(zip_path, out_dir) -> Path:
    """Descomprime con verificación de integridad y protección contra rutas maliciosas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()  # verifica el CRC de cada archivo
        if bad:
            raise RuntimeError(f"Zip corrupto en: {bad}. Borra el zip y descárgalo de nuevo.")
        for m in zf.infolist():
            parts = Path(m.filename).parts
            if not parts or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
                continue  # basura de macOS / archivos ocultos
            target = (out_dir / m.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Ruta sospechosa dentro del zip: {m.filename}")
            zf.extract(m, out_dir)
    return find_class_root(out_dir)


# ------------------------------------------------------------------ índice
def _dhash_hex(img: Image.Image, hash_size: int = 8) -> str:
    """Difference hash: 64 bits que comparan brillo de píxeles vecinos en una
    miniatura 9x8. Imágenes casi iguales -> hashes a poca distancia Hamming."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def build_index(class_root) -> pd.DataFrame:
    """Una fila por archivo: ruta relativa, etiqueta, tamaño, hashes y estadísticas simples."""
    class_root = Path(class_root)
    rows = []
    for label in CLASSES:
        files = sorted((class_root / label).iterdir(), key=lambda p: p.name)
        for p in files:
            if p.suffix.lower() not in IMG_EXT or p.name.startswith("."):
                continue
            row = {"relpath": f"{label}/{p.name}", "label": label,
                   "label_id": CLASSES.index(label), "bytes": p.stat().st_size,
                   "sha256": sha256_file(p), "ok": False, "error": ""}
            try:
                with Image.open(p) as im:
                    im.load()  # decodifica completo: detecta archivos truncados
                    rgb = im.convert("RGB")
                    th = np.asarray(rgb.resize((64, 48)), dtype=np.float32)  # miniatura (48, 64, 3)
                    row.update(width=im.width, height=im.height, mode=im.mode, ok=True,
                               dhash=_dhash_hex(rgb),
                               mean_r=float(th[..., 0].mean()), mean_g=float(th[..., 1].mean()),
                               mean_b=float(th[..., 2].mean()),
                               bright_frac=float((th.min(axis=2) > 200).mean()))  # fracción ~blanca
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    return pd.DataFrame(rows)


def read_csv(path) -> pd.DataFrame:
    """Lee índices/particiones conservando hashes como texto (evita que '00ab…' se vuelva número)."""
    return pd.read_csv(path, dtype=_STR_COLS)


# -------------------------------------------------------------- duplicados
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)  # bits en 1 por byte


def near_duplicate_pairs(hex_hashes, max_dist: int = 2, chunk: int = 256):
    """Pares (i, j, distancia) con distancia Hamming <= max_dist (i < j). O(n^2) por bloques."""
    H = np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)
    n, out = len(H), []
    for i0 in range(0, n, chunk):
        blk = H[i0:i0 + chunk]
        x = np.bitwise_xor(blk[:, None], H[None, :])                     # (c, n) uint64
        d = _POP[x.view(np.uint8)].reshape(len(blk), n, 8).sum(axis=2)   # (c, n) bits distintos
        ii, jj = np.nonzero(d <= max_dist)
        keep = jj > ii + i0
        for a, b in zip(ii[keep], jj[keep]):
            out.append((int(a + i0), int(b), int(d[a, b])))
    return out


def assign_groups(df: pd.DataFrame, pairs) -> np.ndarray:
    """Une en un mismo grupo duplicados exactos (SHA-256) y casi-duplicados (union-find).
    `df` debe tener índice 0..n-1 en el mismo orden que se usó para `pairs`."""
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idx in df.groupby("sha256").indices.values():
        for k in idx[1:]:
            union(int(idx[0]), int(k))
    for i, j, _ in pairs:
        union(i, j)
    return pd.factorize(pd.Series([find(i) for i in range(len(df))]))[0]


def duplicate_report(df: pd.DataFrame, pairs) -> pd.DataFrame:
    cols = ["relpath_a", "relpath_b", "label_a", "label_b", "dist", "exact", "cross_class"]
    if not pairs:
        return pd.DataFrame(columns=cols)
    i, j, d = map(np.array, zip(*pairs))
    a, b = df.iloc[i].reset_index(drop=True), df.iloc[j].reset_index(drop=True)
    rep = pd.DataFrame({"relpath_a": a.relpath, "relpath_b": b.relpath, "label_a": a.label,
                        "label_b": b.label, "dist": d, "exact": a.sha256.values == b.sha256.values})
    rep["cross_class"] = rep.label_a != rep.label_b
    return rep.sort_values(["cross_class", "dist"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- partición
def make_split(df: pd.DataFrame, seed: int, frac_train: float = 0.70,
               frac_val: float = 0.15) -> pd.DataFrame:
    """Partición estratificada por clase a nivel de GRUPO: un grupo de duplicados
    nunca queda repartido entre train/val/test (evita fuga de información)."""
    g = (df.groupby("group")["label"].agg(lambda s: s.mode().iat[0])
           .rename("label").reset_index())
    g_tr, g_rest = train_test_split(g, train_size=frac_train, stratify=g["label"], random_state=seed)
    rel_val = frac_val / (1.0 - frac_train)
    g_va, g_te = train_test_split(g_rest, train_size=rel_val, stratify=g_rest["label"], random_state=seed)
    part = {**dict.fromkeys(g_tr.group, "train"), **dict.fromkeys(g_va.group, "val"),
            **dict.fromkeys(g_te.group, "test")}
    out = df[["relpath", "label", "label_id", "group", "sha256"]].copy()
    out["split"] = out["group"].map(part)
    return out.reset_index(drop=True)


def split_fingerprint(split_df: pd.DataFrame, which: str = "test") -> str:
    """Huella del contenido de una partición: SHA-256 de (ruta, hash de archivo) ordenados."""
    sub = split_df[split_df["split"] == which].sort_values("relpath")
    payload = "".join(f"{r}\t{h}\n" for r, h in zip(sub["relpath"], sub["sha256"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_create_split(df: pd.DataFrame, split_path, seed: int, **kw):
    """Si el CSV existe, lo carga (partición congelada) y verifica que coincida con los
    archivos actuales; si no existe, lo crea. Devuelve (split_df, creado: bool)."""
    split_path = Path(split_path)
    if split_path.exists():
        sp = read_csv(split_path)
        cur = dict(zip(df["relpath"], df["sha256"]))
        old = dict(zip(sp["relpath"], sp["sha256"]))
        if cur != old:
            missing, extra = set(old) - set(cur), set(cur) - set(old)
            changed = [k for k in set(old) & set(cur) if old[k] != cur[k]]
            raise RuntimeError(f"La partición guardada no coincide con los archivos: faltan {len(missing)}, "
                               f"sobran {len(extra)}, cambiados {len(changed)}. ¿Otra versión del dataset?")
        return sp, False
    sp = make_split(df, seed, **kw)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    sp.to_csv(split_path, index=False)
    return sp, True
