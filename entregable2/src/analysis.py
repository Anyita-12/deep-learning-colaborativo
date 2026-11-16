"""Robustez ante perturbaciones y Grad-CAM.

Las perturbaciones son las mismas del Entregable 1 y se aplican a las imágenes de prueba
en [0, 1] antes de normalizar, con semilla fija, para comparar de forma pareada con los modelos base.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

from . import config as C
from .evaluate import compute_metrics

PERTURBATIONS = ["limpio", "ruido_0.05", "ruido_0.10", "desenfoque", "brillo_-30", "brillo_+30", "oclusion",
                 "fondo_sustituido"]
BG_THRESHOLD = 200 / 255


def background_mask(x01: torch.Tensor) -> torch.Tensor:
    return x01.min(dim=1, keepdim=True).values >= BG_THRESHOLD


def perturb(x01: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    n, _, h, w = x01.shape
    if kind == "limpio":
        return x01
    if kind.startswith("ruido_"):
        return (x01 + float(kind.split("_")[1]) * torch.randn(x01.shape, generator=gen)).clamp(0, 1)
    if kind == "desenfoque":
        return TF.gaussian_blur(x01, kernel_size=[9, 9], sigma=[2.0, 2.0])
    if kind.startswith("brillo_"):
        return (x01 * (1 + float(kind.split("_")[1]) / 100)).clamp(0, 1)
    if kind == "oclusion":
        out = x01.clone()
        side = h // 4
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
    gen = torch.Generator().manual_seed(C.ROBUSTNESS_SEED + PERTURBATIONS.index(kind))
    x = torch.from_numpy(np.array(cache[rows])).permute(0, 3, 1, 2).float() / 255
    return perturb(x, kind, gen)


@torch.no_grad()
def probs_from_images(model, x01: torch.Tensor, mean, std, device, xgb=None, bs: int = 64, amp: bool = True) -> np.ndarray:
    model.eval()
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    use_amp = amp and device.type == "cuda"
    out = []
    for k in range(0, len(x01), bs):
        xb = ((x01[k:k + bs] - m) / s).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            o = model(xb, return_embedding=True) if xgb is not None else model(xb)
        if xgb is not None:
            out.append(xgb.predict_proba(o[1].float().cpu().numpy()))
        else:
            out.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def robustness_eval(model, common: dict, cache: np.ndarray, mean, std, device, xgb=None) -> dict:
    rows, y = common["splits"]["test"]["rows"], common["splits"]["test"]["y"]
    res = {}
    for kind in PERTURBATIONS:
        p = probs_from_images(model, perturbed_test_set(cache, rows, kind), mean, std, device, xgb)
        met = compute_metrics(y, p)
        res[kind] = {"f1_macro": met["f1_macro"], "accuracy": met["accuracy"], "f1_per_class": met["f1_per_class"],
                     "ece": met["ece"], "y_pred": p.argmax(1).astype(np.int64)}
    return {"perturbations": PERTURBATIONS, "results": res}


def cam_selection(common: dict) -> np.ndarray:
    """Posiciones fijas dentro del conjunto de prueba (CAM_PER_CLASS por clase), iguales para todos los modelos."""
    y = common["splits"]["test"]["y"]
    rng = np.random.default_rng(C.ROBUSTNESS_SEED)
    pick = [rng.choice(np.flatnonzero(y == k), size=min(C.CAM_PER_CLASS, int((y == k).sum())), replace=False)
            for k in range(C.NUM_CLASSES)]
    return np.sort(np.concatenate(pick))


def _to_map(a: torch.Tensor) -> torch.Tensor:
    if a.dim() == 4:
        return a
    tokens = a[:, 1:] if a.shape[1] > 1 and int((a.shape[1] - 1) ** 0.5) ** 2 == a.shape[1] - 1 else a
    g = int(tokens.shape[1] ** 0.5)
    return tokens.reshape(a.shape[0], g, g, -1).permute(0, 3, 1, 2)


def gradcam(model, layer, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Grad-CAM (Selvaraju et al., 2017). Para capas de tokens, se descarta el token de clase y se
    reordena la secuencia en su malla espacial."""
    acts, grads = {}, {}

    def fwd(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        out.gather(1, target.view(-1, 1)).sum().backward()
        a, g = _to_map(acts["a"].detach().float()), _to_map(grads["g"].detach().float())
    finally:
        h.remove()
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(C.CAM_SIZE, C.CAM_SIZE), mode="bilinear", align_corners=False)[:, 0]
    return cam / cam.flatten(1).max(1).values.view(-1, 1, 1).clamp_min(1e-12)


def gradcam_eval(model, common: dict, cache: np.ndarray, mean, std, device, test_pred: np.ndarray, bs: int = 8) -> dict:
    """Mapas para la clase predicha y la clase real de las imágenes seleccionadas, y la fracción de la
    energía del mapa que cae sobre el fondo claro frente a la fracción del área que es fondo."""
    model.eval()
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    layer = model.cam_layer()
    pos = cam_selection(common)
    rows = common["splits"]["test"]["rows"][pos]
    y = common["splits"]["test"]["y"][pos]
    pred = test_pred[pos]
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    maps = {"pred": [], "true": []}
    with torch.enable_grad():
        for k in range(0, len(rows), bs):
            x01 = torch.from_numpy(np.array(cache[rows[k:k + bs]])).permute(0, 3, 1, 2).float() / 255
            x = ((x01 - m) / s).to(device)
            for key, t in (("pred", pred[k:k + bs]), ("true", y[k:k + bs])):
                xi = x.clone().requires_grad_(True)
                maps[key].append(gradcam(model, layer, xi, torch.as_tensor(t, device=device)).cpu().numpy())
    out = {k: np.concatenate(v).astype(np.float16) for k, v in maps.items()}
    x01 = torch.from_numpy(np.array(cache[rows])).permute(0, 3, 1, 2).float() / 255
    bg = F.interpolate(background_mask(x01).float(), size=(C.CAM_SIZE, C.CAM_SIZE), mode="nearest")[:, 0].numpy()
    mp = out["pred"].astype(np.float32)
    energy = (mp * bg).sum((1, 2)) / np.clip(mp.sum((1, 2)), 1e-12, None)
    return {"test_positions": pos, "rows": rows, "y_true": y, "y_pred": pred, "cam_pred": out["pred"],
            "cam_true": out["true"], "frac_energia_fondo": energy.astype(np.float32),
            "frac_area_fondo": bg.mean((1, 2)).astype(np.float32), "layer": type(layer).__name__}
