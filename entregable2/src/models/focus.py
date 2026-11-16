"""Zheng et al. (2023): Focus-RCNet (Focus + bloques sandglass + SimAM) con destilación desde EfficientNet-B4.

La arquitectura sigue la Tabla 1 del paper (canales 24-48-96-192-384-512 y repeticiones 3-2-1-1).
"""
from __future__ import annotations

from torch import nn
from torchvision import models as tvm

from .layers import Focus, SimAM, conv_bn_act

STAGES = [(48, 3), (96, 2), (192, 1), (384, 1)]


class Sandglass(nn.Module):
    """Zhou et al. (2020), MobileNeXt: depthwise 3x3, reducción 1x1, expansión 1x1 y depthwise 3x3,
    con atajo entre las representaciones de alta dimensión."""

    def __init__(self, cin: int, cout: int, stride: int, reduction: int):
        super().__init__()
        hidden = max(8, cin // reduction)
        self.body = nn.Sequential(
            conv_bn_act(cin, cin, 3, groups=cin, act=nn.ReLU6),
            conv_bn_act(cin, hidden, 1, act=None),
            conv_bn_act(hidden, cout, 1, act=nn.ReLU6),
            conv_bn_act(cout, cout, 3, stride=stride, groups=cout, act=None),
        )
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.residual else y


class FocusRCNet(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, reduction: int = 2, use_simam: bool = True):
        super().__init__()
        self.focus = Focus(3, 24, k=1)
        stages, cin = [], 24
        for cout, repeat in STAGES:
            blocks = [Sandglass(cin, cout, 2, reduction)] + [Sandglass(cout, cout, 1, reduction) for _ in range(repeat)]
            if use_simam:
                blocks.append(SimAM())
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.conv5 = conv_bn_act(cin, 512, 1, act=nn.SiLU)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.conv5(self.stages(self.focus(x)))))

    def cam_layer(self):
        return self.conv5


class EfficientNetB4Teacher(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.net = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1792, num_classes))

    def forward(self, x):
        return self.net(x)

    def cam_layer(self):
        return self.net.features[-1]
