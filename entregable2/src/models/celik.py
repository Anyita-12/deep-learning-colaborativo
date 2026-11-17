"""Celik (2025): fusión HyperColumn de EfficientNetB0 e InceptionV3 con clasificador XGBoost.

Traducción a PyTorch del repositorio oficial (Keras). Correspondencia de capas
(Tabla 10 del paper): block6d_add -> features[6] y top_activation -> features[8]
de EfficientNet-B0; mixed8 -> Mixed_7a y mixed10 -> Mixed_7c de InceptionV3.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

EFFNET_LAYERS = {"full": (6, 8), "last_only": (8,), "model_1": (4, 5)}
INCEPTION_LAYERS = {"full": ("Mixed_7a", "Mixed_7c"), "last_only": ("Mixed_7c",), "model_1": ("Mixed_6d", "Mixed_6e")}
INCEPTION_ORDER = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
                   "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
                   "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]
EFFNET_CHANNELS = {4: 80, 5: 112, 6: 192, 7: 320, 8: 1280}
INCEPTION_CHANNELS = {"Mixed_6d": 768, "Mixed_6e": 768, "Mixed_7a": 1280, "Mixed_7b": 2048, "Mixed_7c": 2048}


class CelikHC(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True, layers: str = "full",
                 use_effnet: bool = True, use_inception: bool = True):
        super().__init__()
        if not (use_effnet or use_inception):
            raise ValueError("Se necesita al menos un extractor.")
        self.use_effnet, self.use_inception = use_effnet, use_inception
        self.eff_taps = EFFNET_LAYERS[layers] if use_effnet else ()
        self.inc_taps = INCEPTION_LAYERS[layers] if use_inception else ()
        width = 0
        if use_effnet:
            w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.effnet = tvm.efficientnet_b0(weights=w).features[: max(self.eff_taps) + 1]
            width += sum(EFFNET_CHANNELS[i] for i in self.eff_taps)
        if use_inception:
            if pretrained:
                inc = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=True)
            else:
                inc = tvm.inception_v3(weights=None, aux_logits=False, transform_input=False, init_weights=True)
            self.inception_transform = bool(inc.transform_input)
            last = max(INCEPTION_ORDER.index(t) for t in self.inc_taps)
            self.inception = nn.ModuleDict({n: getattr(inc, n) for n in INCEPTION_ORDER[: last + 1]})
            width += sum(INCEPTION_CHANNELS[t] for t in self.inc_taps)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, num_classes)
        self.embedding_dim = 64

    def _inception_input(self, x):
        if not self.inception_transform:
            return x
        ch0 = x[:, 0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        ch1 = x[:, 1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        ch2 = x[:, 2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return torch.cat([ch0, ch1, ch2], dim=1)

    def hypercolumn(self, x):
        feats = []
        if self.use_effnet:
            h = x
            for i, block in enumerate(self.effnet):
                h = block(h)
                if i in self.eff_taps:
                    feats.append(self.pool(h).flatten(1))
        if self.use_inception:
            h = self._inception_input(x)
            for name, block in self.inception.items():
                h = block(h)
                if name in self.inc_taps:
                    feats.append(self.pool(h).flatten(1))
        return torch.cat(feats, dim=1)

    def forward(self, x, return_embedding: bool = False):
        emb = self.embedding(self.dropout(self.hypercolumn(x)))
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

    def cam_layer(self):
        return self.effnet[max(self.eff_taps)] if self.use_effnet else self.inception[self.inc_taps[-1]]
