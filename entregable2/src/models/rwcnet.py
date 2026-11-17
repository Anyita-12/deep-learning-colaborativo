"""Hossen et al. (2024): RWC-Net, fusión de DenseNet201 y MobileNetV2 con dos salidas auxiliares.

Salida auxiliar 1: denseblock2 de DenseNet201 + quinto bloque residual invertido de MobileNetV2 (28x28).
Salida auxiliar 2: denseblock3 de DenseNet201 + último bloque residual invertido de MobileNetV2 (7x7).
Salida final: últimas capas convolucionales de ambos modelos, pooling promedio y clasificador lineal.
Pérdida total: L_final + L_aux2 / 2 + L_aux1 / 4 (ponderación 1 / 2^i de la Ecuación 2).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm

AUX_WEIGHTS = (1.0, 0.5, 0.25)


class AuxHead(nn.Module):
    """Rama auxiliar con la estructura de Inception-v3: pooling a 5x5, compresión 1x1 y convolución 5x5."""

    def __init__(self, cin: int, num_classes: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Conv2d(cin, 128, 1, bias=False), nn.BatchNorm2d(128),
                                  nn.ReLU(), nn.Conv2d(128, 768, 5, bias=False), nn.BatchNorm2d(768), nn.ReLU(),
                                  nn.Flatten(), nn.Dropout(dropout), nn.Linear(768, num_classes))

    def forward(self, x):
        return self.body(x)


class RWCNet(nn.Module):
    """use_densenet / use_mobilenet permiten las ablaciones de una sola rama; con ambas activas el modelo es el original."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True, use_aux: bool = True,
                 use_densenet: bool = True, use_mobilenet: bool = True):
        super().__init__()
        if not (use_densenet or use_mobilenet):
            raise ValueError("RWCNet necesita al menos una rama")
        self.use_d, self.use_m = use_densenet, use_mobilenet
        if use_densenet:
            d = tvm.densenet201(weights=tvm.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None).features
        if use_mobilenet:
            m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None).features
        if use_densenet:
            self.d_stem = nn.Sequential(d.conv0, d.norm0, d.relu0, d.pool0, d.denseblock1, d.transition1, d.denseblock2)
            self.d_mid = nn.Sequential(d.transition2, d.denseblock3)
            self.d_tail = nn.Sequential(d.transition3, d.denseblock4, d.norm5, nn.ReLU())
        if use_mobilenet:
            self.m_stem, self.m_mid, self.m_tail = m[:6], m[6:18], m[18:]
        self.use_aux = use_aux
        c1 = 512 * use_densenet + 32 * use_mobilenet
        c2 = 1792 * use_densenet + 320 * use_mobilenet
        cf = 1920 * use_densenet + 1280 * use_mobilenet
        self.aux1 = AuxHead(c1, num_classes, dropout)
        self.aux2 = AuxHead(c2, num_classes, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cf, num_classes))

    def forward(self, x):
        f1, f2, ff = [], [], []
        if self.use_d:
            d2 = self.d_stem(x); d3 = self.d_mid(d2)
            f1.append(d2); f2.append(d3); ff.append(self.d_tail(d3))
        if self.use_m:
            m5 = self.m_stem(x); m17 = self.m_mid(m5)
            f1.append(m5); f2.append(m17); ff.append(self.m_tail(m17))
        out = self.classifier(self.pool(torch.cat(ff, dim=1)))
        if not (self.training and self.use_aux):
            return out
        a1 = self.aux1(torch.cat(f1, dim=1))
        if self.use_d and self.use_m:
            f2[0] = F.adaptive_avg_pool2d(f2[0], f2[1].shape[-2:])
        a2 = self.aux2(torch.cat(f2, dim=1))
        return out, a2, a1

    def cam_layer(self):
        return self.d_tail if self.use_d else self.m_tail
