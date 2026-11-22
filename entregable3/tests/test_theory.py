"""Pruebas de las propiedades formales (Teoremas 1 y 2). Solo numpy y scipy: corren en CI sin GPU."""
import numpy as np
import pytest

from e3 import theory as T


def test_haar_reconstruccion_y_energia():
    r = T.verify_haar()
    assert r["error_reconstruccion"] < 1e-12
    assert r["error_energia_rel"] < 1e-12


def test_ruido_gaussiano_en_hh_conserva_varianza():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, (3, 224, 224))
    d = T.haar_dwt2_np(x)[3]
    assert abs(d.std() - 0.05) < 0.002


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.25, 0.4])
def test_cotas_mad_contienen_lo_observado(eps):
    rows = T.verify_mad_bounds(eps_grid=(eps,), sigmas=(0.03, 0.08), reps=15, seed=2)
    assert all(r["dentro"] for r in rows)


def test_cotas_mad_ordenadas_y_exactas_sin_contaminacion():
    lo, hi = T.mad_bounds(0.0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    prev = (1.0, 1.0)
    for eps in (0.1, 0.2, 0.3, 0.4):
        lo, hi = T.mad_bounds(eps)
        assert lo < prev[0] and hi > prev[1]      # más contaminación, cota más ancha
        prev = (lo, hi)


def test_mezcla_respeta_jensen_y_maximo():
    r = T.verify_mixture_bound(n=20_000)
    assert r["viol_jensen"] == 0 and r["viol_max"] == 0


def test_compuerta_monotona_y_limites():
    s = np.linspace(0, 0.1, 501)
    g = T.gate_np(s, s_lo=0.01, rho=2.0)
    assert np.all(np.diff(g) >= 0)
    assert np.all(g[s <= 0.01] == 0) and np.all(g[s >= 0.02] == 1)
