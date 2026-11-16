"""Shi et al. (2020): Multi-branch Xception (M-b Xception, variante de 896 canales).

Traducción a PyTorch de models/model3.py del repositorio oficial. El flujo medio
sustituye los 8 bloques residuales de Xception por 8 ramas paralelas de tres
convoluciones separables cuyas salidas se suman. Se entrena desde cero, como en el original.
"""
from __future__ import annotations

from torch import nn

from .layers import SeparableConv2d


def _sep_bn(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(SeparableConv2d(cin, cout, 3), nn.BatchNorm2d(cout))


class EntryBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pre_activation: bool):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=2, bias=False), nn.BatchNorm2d(cout))
        self.body = nn.Sequential(
            *([nn.ReLU()] if pre_activation else []),
            _sep_bn(cin, cout), nn.ReLU(), _sep_bn(cout, cout),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.body(x) + self.residual(x)


class MultiBranchBlock(nn.Module):
    def __init__(self, channels: int = 896, branches: int = 8, depth: int = 3):
        super().__init__()
        def branch():
            layers = []
            for _ in range(depth):
                layers += [nn.ReLU(), _sep_bn(channels, channels)]
            return nn.Sequential(*layers)
        self.branches = nn.ModuleList(branch() for _ in range(branches))

    def forward(self, x):
        out = self.branches[0](x)
        for b in self.branches[1:]:
            out = out + b(x)
        return out


class MbXception(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, core_channels: int = 896, branches: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.entry = nn.Sequential(EntryBlock(64, 128, False), EntryBlock(128, 256, True),
                                   EntryBlock(256, core_channels, True))
        self.middle = MultiBranchBlock(core_channels, branches)
        self.exit_residual = nn.Sequential(nn.Conv2d(core_channels, 1024, 1, stride=2, bias=False), nn.BatchNorm2d(1024))
        self.exit_body = nn.Sequential(nn.ReLU(), _sep_bn(core_channels, core_channels), nn.ReLU(),
                                       _sep_bn(core_channels, 1024), nn.MaxPool2d(3, stride=2, padding=1))
        self.exit_head = nn.Sequential(_sep_bn(1024, 1536), nn.ReLU(), _sep_bn(1536, 2048), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2048, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.middle(self.entry(self.stem(x)))
        x = self.exit_body(x) + self.exit_residual(x)
        return self.classifier(self.pool(self.exit_head(x)))

    def cam_layer(self):
        return self.exit_head
