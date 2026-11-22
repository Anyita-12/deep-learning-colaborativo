"""Propiedades formales de NAGFuse y su verificación numérica (solo numpy y scipy, sin torch).

Teorema 1 (estimador MAD bajo contaminación). Sean d_1..d_n los coeficientes HH de nivel 1 de x + e, con
e_ij ~ N(0, sigma^2) iid y Haar ortonormal. Supongamos que a lo sumo una fracción eps < 1/2 de los coeficientes
está "contaminada" (contenido de la imagen o recorte a [0, 1]) y el resto es ruido puro, d_i ~ N(0, sigma^2).
Con G(u) = 2 Phi(u) - 1, la mediana poblacional m de |d| cumple
    G^{-1}((1 - 2 eps) / (2 (1 - eps))) <= m / sigma <= G^{-1}(1 / (2 (1 - eps))),
y para la mediana muestral, con probabilidad >= 1 - 2 exp(-2 n t^2) (DKW), la misma cota vale con
    (1/2 - t - eps) / (1 - eps)  y  (1/2 + t) / (1 - eps)  en lugar de los argumentos anteriores.
Por tanto sigma_hat = mediana / Phi^{-1}(3/4) queda entre c_lo(eps, t) sigma y c_hi(eps, t) sigma.

Teorema 2 (mezcla y compuerta). Para p = pi p_V + (1 - pi) p_C con pi en [0, 1] y cualquier clase y,
    -log p_y <= pi (-log p_V,y) + (1 - pi)(-log p_C,y) <= max(-log p_V,y, -log p_C,y)      (convexidad de -log),
y si sigma_hat >= rho s_lo entonces pi = 1 y p = p_V exactamente. Con el Teorema 1: si el ruido real cumple
sigma >= rho s_lo / c_lo(eps, t), la predicción de NAGFuse coincide con la de la rama ViT con probabilidad
>= 1 - 2 exp(-2 n t^2).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

MAD_CONST = float(norm.ppf(0.75))


def G_inv(p: float) -> float:
    """Inversa de G(u) = P(|Z| <= u) = 2 Phi(u) - 1."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p fuera de [0, 1)")
    return float(norm.ppf((1.0 + p) / 2.0))


def mad_bounds(eps: float, t: float = 0.0) -> tuple[float, float]:
    """Cotas (c_lo, c_hi) de sigma_hat / sigma del Teorema 1. t = 0 da la versión poblacional."""
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps debe estar en [0, 1/2)")
    lo_arg = (0.5 - t - eps) / (1.0 - eps)
    hi_arg = (0.5 + t) / (1.0 - eps)
    if lo_arg <= 0 or hi_arg >= 1:
        raise ValueError("t demasiado grande para este eps")
    return G_inv(lo_arg) / MAD_CONST, G_inv(hi_arg) / MAD_CONST


def dkw_t(n: int, delta: float) -> float:
    """t tal que 2 exp(-2 n t^2) = delta."""
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# ------------------------------------------------------------------ numpy espejo de nagfuse.py
def haar_dwt2_np(x: np.ndarray):
    x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((x00 + x01 + x10 + x11) / 2, (x00 - x01 + x10 - x11) / 2,
            (x00 + x01 - x10 - x11) / 2, (x00 - x01 - x10 + x11) / 2)


def haar_idwt2_np(a, h, v, d) -> np.ndarray:
    out = np.empty(a.shape[:-2] + (2 * a.shape[-2], 2 * a.shape[-1]), dtype=np.result_type(a, np.float64))
    out[..., 0::2, 0::2] = (a + h + v + d) / 2
    out[..., 0::2, 1::2] = (a - h + v - d) / 2
    out[..., 1::2, 0::2] = (a + h - v - d) / 2
    out[..., 1::2, 1::2] = (a - h - v + d) / 2
    return out


def mad_sigma_np(x01: np.ndarray) -> float:
    """x01: (C, H, W). Misma definición que nagfuse.mad_sigma (mediana inferior)."""
    d = np.abs(haar_dwt2_np(x01.astype(np.float64))[3]).ravel()
    return float(np.sort(d)[(d.size - 1) // 2] / MAD_CONST)


def gate_np(s, s_lo: float, rho: float):
    return np.clip((np.asarray(s, float) - s_lo) / ((rho - 1.0) * s_lo), 0.0, 1.0)


# ------------------------------------------------------------------ verificaciones numéricas
def verify_mad_bounds(eps_grid=(0.0, 0.1, 0.2, 0.3, 0.4), sigmas=(0.01, 0.05, 0.1), n_side: int = 224,
                      reps: int = 200, delta: float = 0.01, seed: int = 0) -> list[dict]:
    """Simula imágenes donde una fracción eps de los coeficientes HH lleva contenido arbitrario (adversarial:
    magnitud grande) y el resto es ruido gaussiano puro, y compara sigma_hat / sigma con las cotas."""
    rng = np.random.default_rng(seed)
    n = 3 * (n_side // 2) ** 2
    t = dkw_t(n, delta)
    rows = []
    for eps in eps_grid:
        lo, hi = mad_bounds(eps, t)
        lo_pop, hi_pop = mad_bounds(eps, 0.0)
        for s in sigmas:
            for modo in ("hacia_arriba", "hacia_cero"):     # los dos peores casos de la contaminación
                r = np.empty(reps)
                for k in range(reps):
                    d = rng.normal(0.0, s, n)
                    m = int(round(eps * n))
                    if m:
                        d[rng.choice(n, m, replace=False)] = 1e3 * s if modo == "hacia_arriba" else 0.0
                    r[k] = np.sort(np.abs(d))[(n - 1) // 2] / MAD_CONST / s
                rows.append({"eps": eps, "sigma": s, "contaminacion": modo, "cota_inf": lo, "cota_sup": hi,
                             "cota_inf_pob": lo_pop, "cota_sup_pob": hi_pop, "min_obs": float(r.min()),
                             "max_obs": float(r.max()),
                             "dentro": bool((r >= lo).all() and (r <= hi).all())})
    return rows


def verify_mixture_bound(n: int = 100_000, k: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pv = rng.dirichlet(np.ones(k) * 0.3, n)
    pc = rng.dirichlet(np.ones(k) * 0.3, n)
    pi = rng.uniform(0, 1, n)
    y = rng.integers(0, k, n)
    p = pi[:, None] * pv + (1 - pi[:, None]) * pc
    lhs = -np.log(p[np.arange(n), y])
    mid = pi * -np.log(pv[np.arange(n), y]) + (1 - pi) * -np.log(pc[np.arange(n), y])
    top = np.maximum(-np.log(pv[np.arange(n), y]), -np.log(pc[np.arange(n), y]))
    return {"n": n, "viol_jensen": int((lhs > mid + 1e-12).sum()), "viol_max": int((mid > top + 1e-12).sum()),
            "holgura_media": float((mid - lhs).mean())}


def verify_haar(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (3, 224, 224))
    coeffs = haar_dwt2_np(x)
    rec = haar_idwt2_np(*coeffs)
    energy = sum(float((c ** 2).sum()) for c in coeffs)
    return {"error_reconstruccion": float(np.abs(rec - x).max()),
            "error_energia_rel": abs(energy - float((x ** 2).sum())) / float((x ** 2).sum())}
