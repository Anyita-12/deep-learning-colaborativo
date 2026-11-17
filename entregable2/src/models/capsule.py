"""Huang et al. (2023): ResMsCapsule, red de cápsulas con extractor residual y cápsulas primarias multiescala.

Extractor: ResNet-18 hasta layer3. Cápsulas primarias en tres ramas (núcleos 5, 3 y 1) con
dimensiones (1, 2, 5) (configuración PrimaryCaps-A), concatenadas en cápsulas de 8 dimensiones.
Cápsulas de clase de 16 dimensiones con enrutamiento dinámico (Sabour et al., 2017) y pérdida de margen.
La salida del modelo es log ||v_k||, de modo que softmax(salida) = ||v_k|| / sum_j ||v_j||.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    n2 = (s ** 2).sum(dim=dim, keepdim=True)
    return (n2 / (1 + n2)) * s / torch.sqrt(n2 + eps)


class PrimaryCapsBranch(nn.Module):
    def __init__(self, cin: int, k: int, caps_types: int, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, caps_types * dim, k, stride=2, padding=k // 2)
        self.caps_types, self.dim = caps_types, dim

    def forward(self, x):
        u = self.conv(x)
        b, _, h, w = u.shape
        return u.view(b, self.caps_types, self.dim, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, self.dim)


class ResMsCapsule(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True, caps_dims=(1, 2, 5),
                 kernels=(5, 3, 1), caps_types: int = 32, out_dim: int = 16, routing_iters: int = 3, img_size: int = 224):
        super().__init__()
        r = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.dropout = nn.Dropout2d(dropout)
        self.reduce = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.branches = nn.ModuleList(PrimaryCapsBranch(128, k, caps_types, d) for k, d in zip(kernels, caps_dims))
        self.in_dim = sum(caps_dims)
        side = img_size // 32
        self.n_primary = caps_types * side * side
        self.num_classes, self.out_dim, self.routing_iters = num_classes, out_dim, routing_iters
        self.W = nn.Parameter(0.01 * torch.randn(1, self.n_primary, num_classes, out_dim, self.in_dim))

    def capsule_lengths(self, x):
        f = self.reduce(self.dropout(self.backbone(x)))
        u = squash(torch.cat([b(f) for b in self.branches], dim=-1))
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            u_hat = torch.matmul(self.W, u[:, :, None, :, None]).squeeze(-1)
            logits_b = torch.zeros(u_hat.shape[:3], device=u.device)
            for i in range(self.routing_iters):
                c = F.softmax(logits_b, dim=2)
                v = squash((c.unsqueeze(-1) * u_hat).sum(dim=1))
                if i < self.routing_iters - 1:
                    logits_b = logits_b + (u_hat * v.unsqueeze(1)).sum(-1)
            return v.norm(dim=-1)

    def forward(self, x):
        return torch.log(self.capsule_lengths(x).clamp_min(1e-6))

    def cam_layer(self):
        return self.backbone[-1]


def margin_loss(log_lengths: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None,
                m_pos: float = 0.9, m_neg: float = 0.1, lam: float = 0.5) -> torch.Tensor:
    v = log_lengths.float().exp()
    t = F.one_hot(y, v.shape[1]).float()
    per = (t * F.relu(m_pos - v) ** 2 + lam * (1 - t) * F.relu(v - m_neg) ** 2).sum(1)
    if weight is not None:
        w = weight[y]
        return (w * per).sum() / w.sum()
    return per.mean()
