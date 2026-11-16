from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import config as C
from .models.capsule import margin_loss
from .models.rwcnet import AUX_WEIGHTS


class Criterion(nn.Module):
    """Devuelve (pérdida, logits principales) a partir de la salida del modelo en modo entrenamiento."""

    def __init__(self, kind: str, weight: torch.Tensor | None = None, teacher: nn.Module | None = None,
                 temperature: float = C.KD_TEMPERATURE, alpha: float = C.KD_ALPHA):
        super().__init__()
        self.kind, self.teacher, self.T, self.alpha = kind, teacher, temperature, alpha
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        if kind == "kd" and teacher is None:
            raise ValueError("La destilación necesita un modelo profesor.")

    def ce(self, logits, y):
        return F.cross_entropy(logits.float(), y, weight=self.weight)

    def forward(self, out, x, y):
        if self.kind == "ce":
            return self.ce(out, y), out
        if self.kind == "aux":
            outs = out if isinstance(out, (tuple, list)) else (out,)
            loss = sum(w * self.ce(o, y) for w, o in zip(AUX_WEIGHTS, outs))
            return loss, outs[0]
        if self.kind == "margin":
            return margin_loss(out, y, self.weight), out
        if self.kind == "kd":
            with torch.no_grad():
                t = self.teacher(x).float()
            s = out.float()
            soft = F.kl_div(F.log_softmax(s / self.T, 1), F.softmax(t / self.T, 1), reduction="batchmean") * self.T ** 2
            return self.alpha * soft + (1 - self.alpha) * self.ce(s, y), out
        raise ValueError(self.kind)

    def eval_loss(self, logits, y):
        if self.kind == "margin":
            return margin_loss(logits, y, self.weight)
        return self.ce(logits, y)
