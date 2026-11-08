"""
eda_tests.py  (v2)
Pruebas de hipotesis y auditorias del EDA para TrashNet (tipo 7: clasificacion de imagenes, 6 clases).

Convenciones de forma:
  F_* : (N, F) float  -> descriptores por imagen (brillo, contraste, nitidez, ...)
  y_* : (N,)   int    -> etiqueta en {0..5}
  H   : (N, B) uint8  -> dHash (bits) por imagen
Las funciones solo CALCULAN. Los umbrales de decision se declaran a priori (preregistro).
Regla anti-fuga: todo lo que use la etiqueta (Kruskal, shortcut_probe) se corre SOLO con train.
La comparacion entre particiones (KS, TOST, adversarial, duplicados) usa solo entradas X.

Cambios v2 respecto a v1 (motivados por correr el modulo sobre TrashNet real):
  * descriptores constantes (p. ej. aspecto/tamano en la version redimensionada) se omiten sin romper Kruskal/KS
  * border_descriptors: estadisticos del marco de la imagen (proxy de fondo/adquisicion)
  * shortcut_probe usa balanced accuracy (azar = 1/K) y reporta la ganancia sobre el azar
  * n_for_noise_test: tamano de muestra para PROBAR (no solo estimar) que el ruido de etiqueta es < p_max
  * equivalence_table: TOST por descriptor entre particiones ("no significativo != equivalente")
  * group_stratified_split y partition_hash: particion agrupada y estratificada + hash para congelar el test
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import ttost_ind

SEED = 0
DESC_NAMES = ["brillo", "contraste_rms", "nitidez_lap", "blanco_sat", "fondo_claro", "negro_sat",
              "sesgo_R_menos_B", "aspecto_w_h", "tamano_sqrt_hw", "dispersion_cromatica"]
BORDER_NAMES = ["borde_R", "borde_G", "borde_B", "borde_std", "borde_claro"]


# ------------------------------------------------------------------ 0. Descriptores de pixeles
def pixel_descriptors(img: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) uint8 RGB -> (10,) float. Ver DESC_NAMES."""
    x = img.astype(np.float32) / 255.0                                   # (H, W, 3)
    lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]       # (H, W) luminancia
    lap = (-4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1]
           + lum[1:-1, :-2] + lum[1:-1, 2:])                              # Laplaciano 4-vecinos
    h, w = lum.shape
    mn = img.min(-1)                                                      # (H, W) minimo por pixel
    return np.array([
        lum.mean(),                          # brillo medio
        lum.std(),                           # contraste RMS
        lap.var(),                           # nitidez (varianza del Laplaciano)
        (mn == 255).mean(),                  # fraccion de blanco saturado
        (mn >= 200).mean(),                  # fraccion "fondo claro" (proxy de area NO ocupada por el objeto)
        (img.max(-1) == 0).mean(),           # fraccion de negro saturado
        x[..., 0].mean() - x[..., 2].mean(), # sesgo cromatico R-B (balance de blancos)
        w / h,                               # relacion de aspecto
        np.sqrt(h * w),                      # tamano efectivo
        x.reshape(-1, 3).std(0).mean(),      # dispersion cromatica media
    ], dtype=np.float64)


def border_descriptors(img: np.ndarray, b: int = 16) -> np.ndarray:
    """Estadisticos del marco de b pixeles (donde casi nunca esta el objeto) -> (5,) float. Ver BORDER_NAMES."""
    m = np.ones(img.shape[:2], dtype=bool)
    m[b:-b, b:-b] = False                                                 # (H, W) True solo en el marco
    px = img[m]                                                           # (n_marco, 3) uint8
    pf = px.astype(np.float32) / 255.0
    return np.array([*pf.mean(0), pf.std(), (px.min(-1) >= 200).mean()], dtype=np.float64)


# ------------------------------------------------------------------ 1. Balance de clases
def class_balance(y: np.ndarray) -> dict:
    counts = np.bincount(y)
    p = counts / counts.sum()
    h = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(counts))        # entropia normalizada [0,1]
    return {"counts": counts, "imbalance_ratio": counts.max() / counts.min(), "entropy_norm": h}


def chi2_partitions(y_parts: dict[str, np.ndarray]) -> dict:
    """H0: la distribucion de clases es la misma en todas las particiones (homogeneidad)."""
    k = max(int(v.max()) for v in y_parts.values()) + 1
    table = np.vstack([np.bincount(v, minlength=k) for v in y_parts.values()])   # (P, K)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))                   # Cramer's V
    return {"chi2": chi2, "dof": dof, "p": p, "cramers_v": v}


# ------------------------------------------------------------------ 2. Desplazamiento de covariables
def _holm(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["p"].notna()
    df["p_holm"] = np.nan
    if ok.any():
        df.loc[ok, "p_holm"] = multipletests(df.loc[ok, "p"], method="holm")[1]
    return df


def ks_partitions(F_ref: np.ndarray, F_other: np.ndarray, names=DESC_NAMES) -> pd.DataFrame:
    """KS de dos muestras por descriptor + Holm. D es el tamano de efecto. Constantes -> NaN."""
    rows = []
    for j, n in enumerate(names):
        a, b = F_ref[:, j], F_other[:, j]
        if np.ptp(np.r_[a, b]) == 0:
            rows.append((n, np.nan, np.nan))
        else:
            rows.append((n, *stats.ks_2samp(a, b)))
    return _holm(pd.DataFrame(rows, columns=["feature", "D", "p"]))


def ks_critical_D(n1: int, n2: int, alpha: float = 0.05) -> float:
    """D minimo detectable (aprox. asintotica) por KS bilateral, sin correccion multiple."""
    c = np.sqrt(-0.5 * np.log(alpha / 2))
    return c * np.sqrt((n1 + n2) / (n1 * n2))


def adversarial_validation(F_a: np.ndarray, F_b: np.ndarray, n_perm: int = 100) -> dict:
    """Clasificador que intenta distinguir A de B. H0: AUC = 0.5 (no distinguibles). p por permutacion."""
    X = np.vstack([F_a, F_b])                                       # (Na+Nb, F)
    z = np.r_[np.zeros(len(F_a)), np.ones(len(F_b))].astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    auc, _, p = permutation_test_score(clf, X, z, scoring="roc_auc", cv=cv,
                                       n_permutations=n_perm, random_state=SEED)
    return {"auc": auc, "p_perm": p}


def tost_means(a: np.ndarray, b: np.ndarray, margin: float) -> float:
    """TOST (Welch). H0: |mu_a - mu_b| >= margin. p pequeno => EQUIVALENTES dentro de +-margin."""
    return ttost_ind(a, b, -margin, margin, usevar="unequal")[0]


def equivalence_table(F_ref: np.ndarray, F_other: np.ndarray, margin_sd: float = 0.2,
                      names=DESC_NAMES) -> pd.DataFrame:
    """TOST por descriptor con margen = margin_sd * DE(ref). p_tost pequeno => equivalentes. Holm sobre la familia."""
    rows = []
    for j, n in enumerate(names):
        a, b = F_ref[:, j], F_other[:, j]
        sd = a.std(ddof=1)
        if np.ptp(np.r_[a, b]) == 0:
            rows.append((n, np.nan, np.nan))
            continue
        rows.append((n, (b.mean() - a.mean()) / sd, tost_means(a, b, margin_sd * sd)))
    df = pd.DataFrame(rows, columns=["feature", "diff_en_DE", "p"])
    return _holm(df).rename(columns={"p": "p_tost", "p_holm": "p_tost_holm"})


# ------------------------------------------------------------------ 3. Duplicados, agrupamiento y fuga
def dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    """img: (H, W, 3) uint8 -> (size*size,) bits."""
    from PIL import Image
    g = Image.fromarray(img).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)                               # (size, size+1)
    return (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()


def hamming_matrix(H: np.ndarray) -> np.ndarray:
    """(N, B) bits -> (N, N) distancias de Hamming, por algebra lineal (sin tensor N*N*B)."""
    Hf = H.astype(np.float32)
    s = Hf.sum(1)
    return s[:, None] + s[None, :] - 2.0 * (Hf @ Hf.T)


def near_duplicate_audit(H: np.ndarray, part: np.ndarray, max_dist: int = 4) -> dict:
    """Cuenta pares casi duplicados y cuantos cruzan particiones (part: (N,) id de particion)."""
    D = hamming_matrix(H)
    i, j = np.triu_indices(len(H), k=1)
    close = D[i, j] <= max_dist
    i, j = i[close], j[close]
    cross = part[i] != part[j]
    props = np.bincount(part) / len(part)
    return {"n_pairs": int(close.sum()), "n_cross": int(cross.sum()),
            "frac_cross": float(cross.mean()) if close.any() else float("nan"),
            "expected_frac_if_split_ignores_duplicates": float(1 - (props ** 2).sum())}


def duplicate_groups(H: np.ndarray, max_dist: int = 4) -> np.ndarray:
    """Id de grupo por imagen (componentes conexas del grafo de casi duplicados) -> particion agrupada."""
    D = hamming_matrix(H)
    i, j = np.where(np.triu(D <= max_dist, k=1))
    n = len(H)
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(A, directed=False)[1]


def group_stratified_split(y: np.ndarray, groups: np.ndarray, seed: int = SEED, n_splits: int = 7):
    """Particion estratificada por clase y AGRUPADA (un grupo nunca cruza particiones).
    n_splits=7 -> ~71/14/14 (train/val/test). Devuelve indices ordenados (train, val, test)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(np.zeros(len(y)), y, groups)]
    return np.sort(np.concatenate(folds[2:])), np.sort(folds[1]), np.sort(folds[0])


def partition_hash(names, file_sha, idx: np.ndarray) -> str:
    """SHA-256 de la lista ordenada 'nombre:sha_archivo' de una particion: sirve para congelar el test."""
    items = sorted(f"{names[i]}:{file_sha[i]}" for i in idx)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


# ------------------------------------------------------------------ 4. Atajos (shortcut learning)
def kruskal_by_class(F_train: np.ndarray, y_train: np.ndarray, names=DESC_NAMES) -> pd.DataFrame:
    """H0: el descriptor tiene la misma distribucion en las 6 clases. epsilon^2 = H/(n-1). Constantes -> NaN."""
    n = len(y_train)
    rows = []
    for j, nm in enumerate(names):
        col = F_train[:, j]
        if np.ptp(col) == 0:
            rows.append((nm, np.nan, np.nan, np.nan))
            continue
        h, p = stats.kruskal(*[col[y_train == c] for c in np.unique(y_train)])
        rows.append((nm, h, h / (n - 1), p))
    return _holm(pd.DataFrame(rows, columns=["feature", "H", "epsilon2", "p"]))


def shortcut_probe(F_nuisance: np.ndarray, y_train: np.ndarray, n_perm: int = 100) -> dict:
    """Clasificador SOLO con variables de adquisicion/fondo. H0: balanced accuracy = azar (1/K), por permutacion.
    Un p pequeno prueba que HAY senal; la magnitud ('gain') dice si importa en la practica."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    bacc, _, p = permutation_test_score(clf, F_nuisance, y_train, scoring="balanced_accuracy", cv=cv,
                                        n_permutations=n_perm, random_state=SEED)
    chance = 1.0 / len(np.unique(y_train))
    return {"bal_acc": bacc, "chance": chance, "gain": bacc - chance, "p_perm": p,
            "p_min_posible": 1.0 / (n_perm + 1)}


# ------------------------------------------------------------------ 5. Ruido de etiquetas
def n_for_proportion(p: float = 0.05, e: float = 0.03, z: float = 1.96) -> int:
    """Tamano de muestra para ESTIMAR una proporcion con margen e (ancho de IC). No da potencia para un test."""
    return int(np.ceil(z ** 2 * p * (1 - p) / e ** 2))


def n_for_noise_test(p_true: float = 0.02, p_max: float = 0.05, alpha: float = 0.05,
                     power: float = 0.80, n_max: int = 5000) -> dict:
    """Menor n para que el test binomial exacto (H1: p < p_max) tenga la potencia pedida si la tasa real es p_true."""
    for n in range(20, n_max):
        ks = np.arange(0, n + 1)
        ok = ks[stats.binom.cdf(ks, n, p_max) <= alpha]
        if len(ok) == 0:
            continue
        k_star = int(ok.max())                                            # rechaza H0 si k_malas <= k_star
        pw = float(stats.binom.cdf(k_star, n, p_true))
        if pw >= power:
            return {"n": n, "k_critico": k_star, "potencia": pw}
    return {"n": None, "k_critico": None, "potencia": None}


def label_noise_test(k_wrong: int, n_reviewed: int, p0: float = 0.05, alpha: float = 0.05) -> dict:
    """Revision manual de n imagenes, k mal etiquetadas. H0: p >= p0 ; H1: p < p0 (binomial exacta)."""
    lo, hi = proportion_confint(k_wrong, n_reviewed, alpha=alpha, method="wilson")
    p = stats.binomtest(k_wrong, n_reviewed, p0, alternative="less").pvalue
    return {"p_hat": k_wrong / n_reviewed, "wilson_ci": (lo, hi), "p_value": p}


# ------------------------------------------------------------------ Demo con datos SINTETICOS
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(SEED)
    counts = np.array([403, 501, 410, 594, 482, 137])
    y = np.repeat(np.arange(6), counts)                          # (2527,)
    N = len(y)
    F = rng.normal(size=(N, len(DESC_NAMES)))                    # (2527, 10) sin senal de clase ni de particion
    F[:, 0] += 0.4 * (y == 1)                                    # atajo SINTETICO: la clase 1 sale mas brillante
    F[:, 7] = 1.333                                              # constante a proposito (como en la version redimensionada)

    idx = np.arange(N)
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    print("kruskal con constante:\n", kruskal_by_class(F[tr], y[tr]).round(4).to_string(index=False))
    print("ks con constante (fila aspecto):\n", ks_partitions(F[tr], F[te]).iloc[7].to_dict())
    print("shortcut probe:", shortcut_probe(F[tr][:, :1], y[tr]))
    print("n para PROBAR ruido<5% con tasa real 2%:", n_for_noise_test(), "| n para ESTIMAR (+-3%):", n_for_proportion())
    g = np.arange(N) // 2                                        # grupos de 2 imagenes consecutivas
    a, b, c = group_stratified_split(y, g, seed=SEED)
    print("split agrupado:", len(a), len(b), len(c), "| grupos compartidos:",
          len(set(g[a]) & set(g[c])), len(set(g[a]) & set(g[b])))
    print("hash test:", partition_hash([str(i) for i in idx], ["x"] * N, c)[:16])
