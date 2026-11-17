from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SeparableConv2d(nn.Module):
    """Convolución separable estilo Keras: depthwise k x k seguida de pointwise 1 x 1, sin activación intermedia."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, k, stride=stride, padding=k // 2, groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def conv_bn_act(cin: int, cout: int, k: int = 1, stride: int = 1, groups: int = 1, act: type[nn.Module] | None = nn.ReLU):
    layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    """Woo et al. (2018): atención de canal seguida de atención espacial."""

    def __init__(self, channels: int, reduction: int = 16, k: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(k)

    def forward(self, x):
        return self.spatial(self.channel(x))


class PyramidPooling(nn.Module):
    """Zhao et al. (2017): pooling adaptativo en varias escalas, proyección 1 x 1 y concatenación con la entrada."""

    def __init__(self, channels: int, bins=(1, 2, 3, 6), branch_channels: int | None = None):
        super().__init__()
        bc = branch_channels or channels // len(bins)
        self.bins = bins
        self.branches = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(channels, bc, 1, bias=False), nn.BatchNorm2d(bc), nn.ReLU())
            for b in bins)
        self.out_channels = channels + bc * len(bins)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x] + [F.interpolate(b(x), size=(h, w), mode="bilinear", align_corners=False) for b in self.branches]
        return torch.cat(outs, dim=1)


class SimAM(nn.Module):
    """Yang et al. (2021): atención 3D sin parámetros a partir de la energía mínima de cada neurona."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / max(n, 1)
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)


class Focus(nn.Module):
    """Reordenamiento espacio a profundidad de YOLOv5 seguido de Conv-BN-SiLU."""

    def __init__(self, cin: int, cout: int, k: int = 1):
        super().__init__()
        self.conv = conv_bn_act(cin * 4, cout, k=k, act=nn.SiLU)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))
