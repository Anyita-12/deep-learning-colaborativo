"""Robustez ante perturbaciones y Grad-CAM (cuaderno 47).

Las perturbaciones se aplican a las imágenes de prueba en [0,1] ANTES de
normalizar, con una semilla fija: los tres modelos y todas las semillas ven
exactamente las mismas imágenes perturbadas (comparación pareada).

`fondo_sustituido` responde al hallazgo del EDA (2.7): reemplaza los píxeles de
fondo claro (mínimo RGB >= 200) por un color aleatorio por imagen. Si un modelo
depende del fondo blanco, su F1 cae más con esta perturbación.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

from . import config as C
from .evaluate import compute_metrics
from .models import build_model, gradcam_layer
from .persist import load_pickle, load_torch, save_pickle

PERTURBATIONS = ["limpio", "ruido_0.05", "ruido_0.10", "desenfoque", "brillo_-30", "brillo_+30",
                 "oclusion", "fondo_sustituido"]
BG_THRESHOLD = 200 / 255


def background_mask(x01: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) en [0,1] -> (N,1,H,W) booleano, píxeles de fondo claro."""
    return (x01.min(dim=1, keepdim=True).values >= BG_THRESHOLD)


def perturb(x01: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    n, _, h, w = x01.shape
    if kind == "limpio":
        return x01
    if kind.startswith("ruido_"):
        s = float(kind.split("_")[1])
        return (x01 + s * torch.randn(x01.shape, generator=gen)).clamp(0, 1)
    if kind == "desenfoque":
        return TF.gaussian_blur(x01, kernel_size=[9, 9], sigma=[2.0, 2.0])
    if kind.startswith("brillo_"):
        f = 1 + float(kind.split("_")[1]) / 100
        return (x01 * f).clamp(0, 1)
    if kind == "oclusion":
        out = x01.clone()
        side = h // 4                                       # cuadro de 56x56 (1/16 del área)
        ys = torch.randint(0, h - side + 1, (n,), generator=gen)
        xs = torch.randint(0, w - side + 1, (n,), generator=gen)
        for i in range(n):
            out[i, :, ys[i]:ys[i] + side, xs[i]:xs[i] + side] = 0.5
        return out
    if kind == "fondo_sustituido":
        m = background_mask(x01)
        col = torch.rand((n, 3, 1, 1), generator=gen)
        return torch.where(m, col.expand_as(x01), x01)
    raise ValueError(kind)


def perturbed_test_set(cache: np.ndarray, rows: np.ndarray, kind: str) -> torch.Tensor:
    """Imágenes de prueba perturbadas en [0,1]. La semilla depende solo de la perturbación."""
    gen = torch.Generator().manual_seed(C.ROBUSTNESS_SEED + PERTURBATIONS.index(kind))
    x = torch.from_numpy(np.array(cache[rows])).permute(0, 3, 1, 2).float() / 255
    return perturb(x, kind, gen)


@torch.no_grad()
def _logits(model, x01: torch.Tensor, mean, std, device, bs: int = 64) -> np.ndarray:
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    out = []
    for k in range(0, len(x01), bs):
        xb = ((x01[k:k + bs] - m) / s).to(device)
        out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def load_best(model_name: str, run: Path, device) -> torch.nn.Module:
    cfg = load_pickle(run / "config.pkl")
    model = build_model(model_name, dropout=cfg["hp"]["dropout"])
    model.load_state_dict(load_torch(run / "best_model.pt"))
    return model.to(device).eval()


def robustness_run(model_name: str, run: Path, common: dict, cache: np.ndarray, device,
                   test_sets: dict | None = None) -> dict:
    """Métricas en prueba bajo cada perturbación. Se guarda en run/robustness.pkl."""
    f = run / "robustness.pkl"
    if f.exists():
        return load_pickle(f)
    sp, norm = common["splits"], common["norm"]
    rows, y = sp["test"]["rows"], sp["test"]["y"]
    model = load_best(model_name, run, device)
    res = {}
    for kind in PERTURBATIONS:
        x = test_sets[kind] if test_sets else perturbed_test_set(cache, rows, kind)
        lg = _logits(model, x, norm["mean"], norm["std"], device)
        p = torch.softmax(torch.from_numpy(lg), 1).numpy().astype(np.float64)
        met = compute_metrics(y, p)
        res[kind] = {"f1_macro": met["f1_macro"], "accuracy": met["accuracy"],
                     "f1_per_class": met["f1_per_class"], "ece": met["ece"]}
    out = {"model": model_name, "seed": int(run.name.split("_")[1]), "perturbations": PERTURBATIONS, "results": res}
    save_pickle(out, f)
    return out


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model, layer, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Grad-CAM (Selvaraju et al., 2017): pesos = promedio espacial del gradiente de la
    clase objetivo respecto a las activaciones; mapa = ReLU(suma ponderada). Salida (N,H,W) en [0,1]."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.gather(1, target.view(-1, 1)).sum().backward()
        a, g = acts["a"].detach(), grads["g"].detach()
    finally:
        h1.remove()
        h2.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).max(1).values.view(-1, 1, 1).clamp_min(1e-12)
    return cam / mx


def gradcam_run(model_name: str, run: Path, common: dict, cache: np.ndarray, device,
                rows: np.ndarray, targets: np.ndarray, bs: int = 16) -> np.ndarray:
    """Mapas Grad-CAM (N,224,224) float16 para las filas dadas, respecto a la clase indicada."""
    sp, norm = common["splits"], common["norm"]
    model = load_best(model_name, run, device)
    for mod in model.modules():                 # ReLU/ReLU6 inplace rompen los hooks de backward
        if hasattr(mod, "inplace"):
            mod.inplace = False
    layer = gradcam_layer(model, model_name)
    m = torch.tensor(norm["mean"]).view(1, 3, 1, 1)
    s = torch.tensor(norm["std"]).view(1, 3, 1, 1)
    maps = []
    for k in range(0, len(rows), bs):
        x01 = torch.from_numpy(np.array(cache[rows[k:k + bs]])).permute(0, 3, 1, 2).float() / 255
        x = ((x01 - m) / s).to(device).requires_grad_(True)
        t = torch.as_tensor(targets[k:k + bs], device=device)
        maps.append(gradcam(model, layer, x, t).cpu().numpy().astype(np.float16))
    return np.concatenate(maps)


def background_attention(maps: np.ndarray, cache: np.ndarray, rows: np.ndarray) -> dict:
    """Fracción de la 'energía' del mapa que cae sobre el fondo claro, frente a la fracción
    del área que es fondo. Si la primera supera a la segunda, el modelo mira más el fondo
    de lo que su área justificaría."""
    x01 = torch.from_numpy(np.array(cache[rows])).permute(0, 3, 1, 2).float() / 255
    bg = background_mask(x01)[:, 0].numpy()
    mp = maps.astype(np.float32)
    energy = (mp * bg).sum((1, 2)) / np.clip(mp.sum((1, 2)), 1e-12, None)
    area = bg.mean((1, 2))
    return {"frac_energia_fondo": energy, "frac_area_fondo": area}
