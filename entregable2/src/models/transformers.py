"""Huang et al. (2021): ViT-B/16 con ajuste fino. Wang et al. (2025): GFN (ResNet50 + PPM + CBAM + codificador ViT)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm

from .layers import CBAM, PyramidPooling


def _vit_b16(pretrained: bool):
    return tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)


class ViTB16(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.vit = _vit_b16(pretrained)
        self.vit.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.vit.hidden_dim, num_classes))
        self.grid = self.vit.image_size // self.vit.patch_size

    def forward(self, x):
        return self.vit(x)

    def cam_layer(self):
        return self.vit.encoder.layers[-1].ln_1


class GFN(nn.Module):
    """La salida de ResNet50 (2048x7x7) se proyecta a 1000 canales en una malla 4x4, pasa por PPM
    (1000 -> 2000 canales) y CBAM, y se convierte en 16 tokens de 768 dimensiones para los 12
    bloques del codificador de ViT-B/16, con token de clase y embeddings de posición (17 x 768)."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True, grid: int = 4,
                 embed_dropout: float = 0.1, use_ppm: bool = True, use_cbam: bool = True):
        super().__init__()
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                      resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.reduce = nn.Sequential(nn.Conv2d(2048, 1000, 1, bias=False), nn.BatchNorm2d(1000), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(grid))
        self.ppm = PyramidPooling(1000, bins=(1, 2, 3, 6), branch_channels=250) if use_ppm else nn.Identity()
        ch = 2000 if use_ppm else 1000
        self.cbam = CBAM(ch, reduction=16) if use_cbam else nn.Identity()
        vit = _vit_b16(pretrained)
        d = vit.hidden_dim
        self.project = nn.Conv2d(ch, d, 1)
        self.class_token = nn.Parameter(vit.class_token.detach().clone())
        self.pos_embedding = nn.Parameter(self._resize_pos(vit.encoder.pos_embedding.detach(), grid))
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.encoder = vit.encoder.layers
        self.norm = vit.encoder.ln
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    @staticmethod
    def _resize_pos(pos: torch.Tensor, grid: int) -> torch.Tensor:
        cls, patches = pos[:, :1], pos[:, 1:]
        g = int(patches.shape[1] ** 0.5)
        p = patches.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(grid, grid), mode="bicubic", align_corners=False)
        return torch.cat([cls, p.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)], dim=1).clone()

    def forward(self, x):
        f = self.cbam(self.ppm(self.reduce(self.backbone(x))))
        tokens = self.project(f).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.class_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        tokens = self.encoder(self.embed_dropout(tokens + self.pos_embedding))
        return self.head(self.norm(tokens)[:, 0])

    def cam_layer(self):
        return self.backbone[-1]
