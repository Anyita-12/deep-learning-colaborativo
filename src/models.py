"""Los tres modelos base, entrenados desde cero (weights=None).

Los tres terminan en Linear + softmax: una regresión logística multinomial sobre
rasgos aprendidos phi_theta(x). La pérdida (entropía cruzada) recibe logits.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

from . import config as C


class VGGLikeCNN(nn.Module):
    """CNN poco profunda estilo VGG reducida, sin atajos.
    4 bloques [Conv3x3-BN-ReLU-Conv3x3-BN-ReLU-MaxPool2], canales 32-64-128-256,
    GlobalAvgPool, Dropout y Linear(256 -> 6)."""

    def __init__(self, num_classes: int = C.NUM_CLASSES, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, cin = [], 3
        for cout in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            ))
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int = C.NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    if name == "CNN":
        return VGGLikeCNN(num_classes, dropout=dropout)
    if name == "MobileNetV2":
        return tvm.mobilenet_v2(weights=None, num_classes=num_classes, width_mult=1.0, dropout=dropout)
    if name == "ResNet-18":
        m = tvm.resnet18(weights=None, num_classes=num_classes)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))   # misma cabeza con dropout
        return m
    raise ValueError(f"Modelo desconocido: {name}")


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Última capa convolucional (su salida) para Grad-CAM."""
    if name == "CNN":
        return model.features[-1][5]          # ReLU tras la última conv, antes del MaxPool (28x28)
    if name == "MobileNetV2":
        return model.features[-1]             # Conv 1x1 a 1280 canales + BN + ReLU6 (7x7)
    if name == "ResNet-18":
        return model.layer4[-1]               # último bloque residual (7x7)
    raise ValueError(name)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    size_mb += sum(b.numel() * b.element_size() for b in model.buffers()) / 2**20
    return {"params_total": int(total), "params_trainable": int(trainable), "size_mb": float(size_mb)}


@torch.no_grad()
def count_macs(model: nn.Module, size: int = C.IMG_SIZE) -> int:
    """Multiplicaciones-acumulaciones por imagen de las capas Conv2d y Linear (con hooks)."""
    macs = [0]

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        macs[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(1, 3, size, size, device=dev))
    model.train(was)
    for h in hs:
        h.remove()
    return int(macs[0])


def model_info(model: nn.Module, name: str) -> dict:
    info = count_params(model)
    info["macs"] = count_macs(model)
    info["gmacs"] = info["macs"] / 1e9
    info["model"] = name
    info["input"] = [3, C.IMG_SIZE, C.IMG_SIZE]
    return info
