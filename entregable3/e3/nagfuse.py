"""NAGFuse: fusión CNN-ViT con compuerta adaptativa al nivel de ruido (Noise-Adaptive Gated Fusion).

Implementación propia a partir de la formulación de experiments/procedencia.md y del capítulo de diseño.
Solo se importan de torchvision las dos redes preentrenadas (misma fuente de pesos que los rivales del E2);
el estimador de ruido, la transformada de Haar, la contracción, la compuerta y la mezcla están escritos aquí.

Notación (por imagen):
    x01 in [0,1]^{3xHxW}                          imagen antes de normalizar
    d = HH_1(x01)                                 coeficientes diagonales de Haar ortonormal, nivel 1
    sigma_hat = mediana(|d|) / Phi^{-1}(3/4)       estimador MAD de Donoho y Johnstone (1994)
    x_c = W^{-1} S_tau(W x01),  tau = sigma_hat * sqrt(2 ln n)   contracción suave (solo rama CNN)
    p_V = softmax(f_V(x)),  p_C = softmax(f_C(x_c))
    pi(x) = pi0 + (1 - pi0) g(sigma_hat),  g(s) = clip((s - s_lo) / ((rho - 1) s_lo), 0, 1)
    p(x) = pi(x) p_V + (1 - pi(x)) p_C            mezcla en el espacio de probabilidades
La compuerta g solo actúa en inferencia; en entrenamiento pi = pi0 (aprendido).
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAD_CONST = 0.6744897501960817          # Phi^{-1}(3/4): mediana de |Z| con Z ~ N(0, 1)


# ------------------------------------------------------------------ transformada de Haar ortonormal 2D
def haar_dwt2(x: torch.Tensor):
    """Un nivel de Haar 2D ortonormal. x: (B, C, H, W) con H y W pares.
    Devuelve (a, h, v, d), cada uno (B, C, H/2, W/2). Conserva la energía: ||x||^2 = ||a||^2+||h||^2+||v||^2+||d||^2."""
    x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    a = (x00 + x01 + x10 + x11) / 2
    h = (x00 - x01 + x10 - x11) / 2
    v = (x00 + x01 - x10 - x11) / 2
    d = (x00 - x01 - x10 + x11) / 2
    return a, h, v, d


def haar_idwt2(a, h, v, d) -> torch.Tensor:
    """Inversa exacta de haar_dwt2."""
    x00 = (a + h + v + d) / 2
    x01 = (a - h + v - d) / 2
    x10 = (a + h - v - d) / 2
    x11 = (a - h - v + d) / 2
    top = torch.stack([x00, x01], dim=-1).flatten(-2)       # (B, C, H/2, W)
    bot = torch.stack([x10, x11], dim=-1).flatten(-2)
    return torch.stack([top, bot], dim=-2).flatten(-3, -2)   # (B, C, H, W)


def mad_sigma(x01: torch.Tensor) -> torch.Tensor:
    """Estimador MAD de la desviación estándar del ruido aditivo. x01: (B, C, H, W) -> (B,).
    Usa los coeficientes HH de nivel 1 de los C canales juntos (mediana inferior si el número es par)."""
    _, _, _, d = haar_dwt2(x01.float())
    return d.abs().flatten(1).median(dim=1).values / MAD_CONST


def soft_threshold(c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.sign(c) * torch.clamp(c.abs() - tau, min=0.0)


def wavelet_shrink(x01: torch.Tensor, sigma: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Contracción suave de los detalles de Haar con el umbral universal tau = sigma sqrt(2 ln n),
    n = H * W (Donoho y Johnstone, 1994). sigma: (B,). Devuelve la imagen reconstruida en [0, 1]."""
    n = x01.shape[-1] * x01.shape[-2]
    tau = (sigma.float() * math.sqrt(2.0 * math.log(n))).view(-1, 1, 1, 1)
    a, details = x01.float(), []
    for _ in range(levels):
        a, h, v, d = haar_dwt2(a)
        details.append((soft_threshold(h, tau), soft_threshold(v, tau), soft_threshold(d, tau)))
    for h, v, d in reversed(details):
        a = haar_idwt2(a, h, v, d)
    return a.clamp(0.0, 1.0)


def gate_value(sigma: torch.Tensor, s_lo: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """g(s) = clip((s - s_lo) / ((rho - 1) s_lo), 0, 1): no decreciente, 0 hasta s_lo y 1 desde rho * s_lo."""
    return ((sigma - s_lo) / ((rho - 1.0) * s_lo)).clamp(0.0, 1.0)


def mixture_log_prob(lv: torch.Tensor, lc: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """log(pi * exp(lv) + (1 - pi) * exp(lc)) estable, con pi: (B,). Si pi == 1 devuelve lv exactamente.
    Se usa el doble where para que pi = 1 no produzca gradientes NaN."""
    pi = pi.float().view(-1, 1)
    full = pi >= 1.0
    pi_safe = torch.where(full, torch.full_like(pi, 0.5), pi)
    mix = torch.logaddexp(torch.log(pi_safe) + lv, torch.log1p(-pi_safe) + lc)
    return torch.where(full, lv, mix)


def load_noise_stats(path: str | None = None) -> dict | None:
    path = path or os.environ.get("E3_NOISE_STATS")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ modelo
class NAGFuse(nn.Module):
    """use_vit / use_cnn permiten las variantes de una sola rama; denoise activa la contracción de la rama CNN.
    gate_mode: 'on' (compuerta en inferencia) u 'off' (pi = pi0 siempre), para la ablación sin reentrenar."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True, use_vit: bool = True,
                 use_cnn: bool = True, denoise: bool = True, s_lo: float | None = None, rho: float = 2.0,
                 shrink_levels: int = 2):
        super().__init__()
        if not (use_vit or use_cnn):
            raise ValueError("NAGFuse necesita al menos una rama")
        from torchvision import models as tvm
        self.use_vit, self.use_cnn, self.denoise, self.shrink_levels = use_vit, use_cnn, denoise, shrink_levels
        if use_vit:
            self.vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.vit.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.vit.hidden_dim, num_classes))
        if use_cnn:
            eff = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.cnn = eff.features
            self.cnn_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                          nn.Linear(1280, num_classes))
        self.pi_logit = nn.Parameter(torch.zeros(()))              # pi0 = 0.5 al inicio
        if s_lo is None:
            st = load_noise_stats()
            s_lo = float(st["s_lo"]) if st else 0.01               # se sobrescribe al cargar el state_dict
        self.register_buffer("s_lo", torch.tensor(float(s_lo)))
        self.register_buffer("rho", torch.tensor(float(rho)))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.gate_mode = "on"
        self.fused = use_vit and use_cnn

    # -- piezas reutilizadas por el análisis
    def to01(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.std + self.mean

    def noise_sigma(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return mad_sigma(self.to01(x))

    def mixture_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        pi0 = torch.sigmoid(self.pi_logit.float()).expand_as(sigma)
        if self.training or self.gate_mode == "off":
            return pi0
        return pi0 + (1.0 - pi0) * gate_value(sigma, self.s_lo, self.rho)

    def _cnn_input(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if not self.denoise:
            return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            xc = wavelet_shrink(self.to01(x), sigma, self.shrink_levels)
            return ((xc - self.mean) / self.std).to(x.dtype)

    def branches(self, x: torch.Tensor) -> dict:
        sigma = self.noise_sigma(x)
        out = {"sigma": sigma}
        if self.use_vit:
            out["z_v"] = self.vit(x)
        if self.use_cnn:
            out["z_c"] = self.cnn_head(self.cnn(self._cnn_input(x, sigma)))
        return out

    def forward(self, x: torch.Tensor):
        b = self.branches(x)
        if not self.fused:
            return b["z_v"] if self.use_vit else b["z_c"]
        lv = F.log_softmax(b["z_v"].float(), dim=1)
        lc = F.log_softmax(b["z_c"].float(), dim=1)
        logp = mixture_log_prob(lv, lc, self.mixture_weight(b["sigma"]))
        if self.training:
            return logp, b["z_v"], b["z_c"]
        return logp

    @torch.no_grad()
    def details(self, x: torch.Tensor) -> dict:
        """Salidas intermedias en inferencia (sigma_hat, pi, p_V, p_C, p) para el análisis y el dashboard."""
        b = self.branches(x)
        d = {"sigma": b["sigma"].float()}
        if "z_v" in b:
            d["p_v"] = torch.softmax(b["z_v"].float(), 1)
        if "z_c" in b:
            d["p_c"] = torch.softmax(b["z_c"].float(), 1)
        if self.fused:
            pi = self.mixture_weight(b["sigma"])
            d["pi"] = pi
            d["p"] = torch.exp(mixture_log_prob(F.log_softmax(b["z_v"].float(), 1),
                                                F.log_softmax(b["z_c"].float(), 1), pi))
        return d

    def cam_layer(self):
        return self.vit.encoder.layers[-1].ln_1 if self.use_vit else self.cnn[-1]


VARIANT_KW = {
    "NAGFuse": {},
    "NAGFuse/sin_denoising": {"denoise": False},
    "NAGFuse/solo_vit": {"use_cnn": False, "denoise": False},
    "NAGFuse/solo_cnn": {"use_vit": False},
}


def build_nagfuse(name: str, dropout: float = 0.3, num_classes: int = 6, pretrained: bool = True,
                  rho: float = 2.0) -> NAGFuse:
    return NAGFuse(num_classes, dropout, pretrained=pretrained, rho=rho, **VARIANT_KW[name])
