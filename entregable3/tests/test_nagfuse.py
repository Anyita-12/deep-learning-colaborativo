"""Pruebas del modelo propio NAGFuse (CPU, sin pesos preentrenados para no descargar nada)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from e3 import nagfuse as N  # noqa: E402
from e3 import theory as T  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = N.NAGFuse(num_classes=6, dropout=0.1, pretrained=False, s_lo=0.01, rho=2.0)
    return m.eval()


def _norm(x01):
    mean = torch.tensor(N.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(N.IMAGENET_STD).view(1, 3, 1, 1)
    return (x01 - mean) / std


def test_haar_torch_igual_a_numpy_y_reconstruye():
    x = torch.rand(2, 3, 224, 224, dtype=torch.float64)
    a, h, v, d = N.haar_dwt2(x)
    ref = T.haar_dwt2_np(x.numpy())
    for c, r in zip((a, h, v, d), ref):
        assert np.allclose(c.numpy(), r)
    assert torch.allclose(N.haar_idwt2(a, h, v, d), x, atol=1e-12)


def test_mad_sigma_estima_ruido_puro_dentro_de_la_cota():
    torch.manual_seed(1)
    for s in (0.02, 0.05, 0.1):
        x = 0.5 + s * torch.randn(4, 3, 224, 224)
        est = N.mad_sigma(x) / s
        lo, hi = T.mad_bounds(0.0, T.dkw_t(3 * 112 * 112, 1e-3))
        assert torch.all(est > lo) and torch.all(est < hi)


def test_mad_sigma_igual_a_numpy():
    x = torch.rand(1, 3, 224, 224)
    assert abs(float(N.mad_sigma(x)[0]) - T.mad_sigma_np(x[0].numpy())) < 1e-6


def test_contraccion_reduce_el_ruido():
    torch.manual_seed(2)
    clean = torch.zeros(1, 3, 224, 224)
    clean[..., 40:180, 60:160] = 0.8                      # imagen constante por tramos
    noisy = (clean + 0.05 * torch.randn_like(clean)).clamp(0, 1)
    den = N.wavelet_shrink(noisy, N.mad_sigma(noisy))
    assert float(((den - clean) ** 2).mean()) < 0.5 * float(((noisy - clean) ** 2).mean())


def test_salidas_y_formas(model):
    x = _norm(torch.rand(2, 3, 224, 224))
    with torch.no_grad():
        logp = model(x)
    assert logp.shape == (2, 6)
    assert torch.allclose(logp.exp().sum(1), torch.ones(2), atol=1e-5)
    model.train()
    out = model(x)
    model.eval()
    assert isinstance(out, tuple) and len(out) == 3 and out[1].shape == (2, 6)


def test_compuerta_abierta_da_exactamente_la_rama_vit(model):
    torch.manual_seed(3)
    x = _norm((0.5 + 0.2 * torch.randn(2, 3, 224, 224)).clamp(0, 1))   # ruido fuerte: sigma_hat >> 2 s_lo
    with torch.no_grad():
        d = model.details(x)
        logp = model(x)
    assert torch.all(d["pi"] == 1.0)
    assert torch.allclose(logp, torch.log_softmax(model.vit(x).float(), 1), atol=1e-6)


def test_compuerta_apagada_usa_pi0(model):
    x = _norm((0.5 + 0.2 * torch.randn(1, 3, 224, 224)).clamp(0, 1))
    model.gate_mode = "off"
    with torch.no_grad():
        pi = model.details(x)["pi"]
    model.gate_mode = "on"
    assert torch.allclose(pi, torch.sigmoid(model.pi_logit).expand_as(pi))


def test_gradientes_finitos_con_pi_igual_a_uno():
    lv = torch.log_softmax(torch.randn(3, 6, requires_grad=True), 1)
    lc = torch.log_softmax(torch.randn(3, 6, requires_grad=True), 1)
    w = torch.zeros((), requires_grad=True)
    pi = torch.sigmoid(w) + (1 - torch.sigmoid(w)) * torch.ones(3)      # pi = 1
    out = N.mixture_log_prob(lv, lc, pi)
    out[:, 0].sum().backward()
    assert torch.isfinite(w.grad).all()


def test_cota_de_la_mezcla_en_el_modelo(model):
    x = _norm(torch.rand(4, 3, 224, 224))
    with torch.no_grad():
        d = model.details(x)
    y = torch.tensor([0, 1, 2, 3])
    nll = -torch.log(d["p"][torch.arange(4), y])
    bound = d["pi"] * -torch.log(d["p_v"][torch.arange(4), y]) + (1 - d["pi"]) * -torch.log(d["p_c"][torch.arange(4), y])
    assert torch.all(nll <= bound + 1e-5)


def test_state_dict_guarda_s_lo_y_rho(model):
    sd = model.state_dict()
    assert math.isclose(float(sd["s_lo"]), 0.01) and math.isclose(float(sd["rho"]), 2.0)


@pytest.mark.parametrize("name", list(N.VARIANT_KW))
def test_variantes(name):
    m = N.build_nagfuse(name, pretrained=False).eval()
    with torch.no_grad():
        o = m(_norm(torch.rand(1, 3, 224, 224)))
    assert o.shape == (1, 6)
