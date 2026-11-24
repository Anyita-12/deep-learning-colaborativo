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
    assert math.isclose(float(sd["s_lo"]), 0.01, rel_tol=1e-6) and math.isclose(float(sd["rho"]), 2.0, rel_tol=1e-6)  # float32


@pytest.mark.parametrize("name", list(N.VARIANT_KW))
def test_variantes(name):
    m = N.build_nagfuse(name, pretrained=False).eval()
    with torch.no_grad():
        o = m(_norm(torch.rand(1, 3, 224, 224)))
    assert o.shape == (1, 6)


def test_finite_logits_reintenta_en_fp32_y_no_altera_salidas_finitas():
    x = torch.zeros(2, 3)
    llamadas = []

    def fn_inestable(inp):
        llamadas.append(inp.dtype)
        return torch.full((2, 6), float("nan")) if len(llamadas) == 1 else torch.ones(2, 6)

    z = N.finite_logits(fn_inestable, x, training=False)
    assert torch.isfinite(z).all() and len(llamadas) == 2

    n = []
    z2 = N.finite_logits(lambda inp: (n.append(1), torch.ones(2, 6))[1], x, training=False)
    assert len(n) == 1 and torch.equal(z2, torch.ones(2, 6))      # salida finita: una sola evaluación


def test_finite_logits_falla_si_fp32_tambien_es_no_finita():
    with pytest.raises(FloatingPointError):
        N.finite_logits(lambda inp: torch.full((1, 6), float("inf")), torch.zeros(1, 3), training=False)


def test_variante_sin_denoising_finita_con_ruido_fuerte_y_compuerta_apagada():
    torch.manual_seed(1)
    m = N.NAGFuse(num_classes=6, pretrained=False, s_lo=0.01, rho=2.0, denoise=False).eval()
    x01 = (torch.rand(2, 3, 224, 224) + 0.5 * torch.randn(2, 3, 224, 224)).clamp(0, 1)
    for mode in ("on", "off"):
        m.gate_mode = mode
        assert torch.isfinite(m(_norm(x01))).all()


# ------------------------------------------------------------------ opción B (protocolo v4)
def test_version_a_conserva_nombres_de_pesos():
    """La refactorización de las ramas no cambia los nombres del state_dict de A (sus pesos siguen cargando)."""
    m = N.NAGFuse(num_classes=6, pretrained=False, s_lo=0.01)
    k = m.state_dict().keys()
    assert "vit.heads.1.weight" in k and "cnn_head.3.weight" in k and "pi_logit" in k
    assert m.vit.heads[1].in_features == 768 and m.cnn_head[3].in_features == 1280


@pytest.mark.parametrize("nombre", ["NAGFuse-B1", "NAGFuse-B2", "NAGFuse-B3"])
def test_candidatos_b_formas_y_compuerta(nombre):
    torch.manual_seed(0)
    m = N.NAGFuse(num_classes=6, dropout=0.1, pretrained=False, s_lo=0.01, rho=2.0, **N.VARIANT_KW[nombre]).eval()
    x = _norm(torch.rand(2, 3, 224, 224))
    with torch.no_grad():
        logp = m(x)
    assert logp.shape == (2, 6) and torch.allclose(logp.exp().sum(1), torch.ones(2), atol=1e-5)
    m.train()
    out = m(x)
    assert isinstance(out, tuple) and out[1].shape == (2, 6) and out[2].shape == (2, 6)
    m.eval()
    xr = _norm((0.5 + 0.2 * torch.randn(2, 3, 224, 224)).clamp(0, 1))      # ruido fuerte: pi = 1
    with torch.no_grad():
        d = m.details(xr)
        assert torch.all(d["pi"] == 1.0)
        assert torch.allclose(m(xr), torch.log_softmax(m.vit(xr).float(), 1), atol=1e-6)   # Teorema 2 en B


def test_nagfuse_b_exige_candidato_elegido(monkeypatch):
    from e3 import cfg
    monkeypatch.setattr(cfg, "ELEGIDO_B", None)
    with pytest.raises(ValueError):
        N.build_nagfuse("NAGFuse-B", pretrained=False)
