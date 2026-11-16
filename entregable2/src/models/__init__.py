"""Registro de los modelos del benchmark y de las variantes de ablación."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .. import config as C
from .capsule import ResMsCapsule
from .celik import CelikHC
from .focus import EfficientNetB4Teacher, FocusRCNet
from .mbxception import MbXception
from .rwcnet import RWCNet
from .transformers import GFN, ViTB16

PRETRAINED = os.environ.get("TRASHNET_PRETRAINED", "1") == "1"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    reference: str
    doi: str
    builder: str
    kwargs: dict = field(default_factory=dict)
    pretrained: bool = True
    base_lr: float = 3e-4
    normalization: str = "imagenet"
    loss: str = "ce"
    post_hoc: str | None = None
    defines_architecture: str = "sí"
    deviations: tuple = ()


SPECS: dict[str, ModelSpec] = {
    "Celik-HC": ModelSpec(
        "Celik-HC", "Híbrido DL + ML (fusión multinivel de CNN + XGBoost)",
        "Celik, G. (2025). Multi-layer feature fusion for high-accuracy solid waste classification using a hybrid "
        "deep learning model. The Visual Computer.", "10.1007/s00371-025-04031-3", "celik",
        base_lr=3e-4, post_hoc="xgb", defines_architecture="sí (código oficial)",
        deviations=("PyTorch en lugar de Keras con pesos ImageNet de torchvision",
                    "Dropout antes de la primera capa densa para el espacio de búsqueda común",
                    "Selección de época con validación en lugar del conjunto de prueba")),
    "MbXception": ModelSpec(
        "MbXception", "CNN multirrama (Xception modificada)",
        "Shi, C., Xia, R., & Wang, L. (2020). A novel multi-branch channel expansion network for garbage image "
        "classification. IEEE Access, 8, 154436-154452.", "10.1109/ACCESS.2020.3016116", "mbxception",
        pretrained=False, base_lr=1e-3, normalization="train", defines_architecture="sí (código oficial)",
        deviations=("Entrada 224x224 en lugar de 229x229", "AdamW en lugar de SGD con momento",
                    "Dropout antes del clasificador para el espacio de búsqueda común")),
    "ViT-B16": ModelSpec(
        "ViT-B16", "Transformer (ViT con ajuste fino)",
        "Huang, K., Lei, H., Jiao, Z., & Zhong, Z. (2021). Recycling waste classification using vision transformer "
        "on portable device. Sustainability, 13(21), 11572.", "10.3390/su132111572", "vit",
        base_lr=1e-4, defines_architecture="parcial (ViT estándar; variante no declarada, se asume B/16)",
        deviations=("Variante ViT-B/16 asumida", "Lote 32 en lugar de 512")),
    "GFN": ModelSpec(
        "GFN", "Híbrido CNN-Transformer con atención (ResNet50 + PPM + CBAM + ViT)",
        "Wang, Z., et al. (2025). GFN: A garbage classification fusion network incorporating multiple attention "
        "mechanisms. Electronics, 14(1), 75.", "10.3390/electronics14010075", "gfn",
        base_lr=1e-4, defines_architecture="sí (dimensiones de cada etapa en el texto)",
        deviations=("Proyección 2048x7x7 -> 1000x4x4 mediante conv 1x1 y pooling adaptativo (el texto no la detalla)",
                    "Embeddings de posición inicializados por interpolación de los de ViT-B/16")),
    "FocusRCNet": ModelSpec(
        "FocusRCNet", "CNN ligera con atención y destilación de conocimiento",
        "Zheng, D., Wang, R., Duan, Y., Pang, P. C.-I., & Tan, T. (2023). Focus-RCNet: a lightweight recyclable "
        "waste classification algorithm based on focus and knowledge distillation. Visual Computing for Industry, "
        "Biomedicine, and Art, 6, 19.", "10.1186/s42492-023-00146-3", "focus",
        pretrained=False, base_lr=1e-3, normalization="imagenet", loss="kd",
        defines_architecture="sí (Tabla 1)",
        deviations=("Entrada 224x224 en lugar de 380x380", f"T = {C.KD_TEMPERATURE} y alpha = {C.KD_ALPHA} (no declarados)",
                    "Factor de reducción del bloque sandglass = 2 (no declarado)")),
    "ResMsCapsule": ModelSpec(
        "ResMsCapsule", "Red de cápsulas con extractor residual",
        "Huang, L., Li, M., Xu, T., & Dong, S.-Q. (2023). A waste classification method based on a capsule "
        "network. Environmental Science and Pollution Research, 30, 86454-86462.", "10.1007/s11356-023-27970-7",
        "capsule", base_lr=3e-4, loss="margin", defines_architecture="parcial (estructura y dimensiones de cápsulas)",
        deviations=("ResNet-18 hasta layer3 con pesos ImageNet", "32 tipos de cápsula por rama y núcleos 5/3/1 (no declarados)",
                    "El paper reporta 4 clases; aquí se usan las 6 de TrashNet")),
    "RWCNet": ModelSpec(
        "RWCNet", "Fusión de CNN preentrenadas con salidas auxiliares",
        "Hossen, M. M., et al. (2024). A reliable and robust deep learning model for effective recyclable waste "
        "classification. IEEE Access, 12, 13809-13821.", "10.1109/ACCESS.2024.3354774", "rwcnet",
        base_lr=3e-4, loss="aux", defines_architecture="sí (puntos de fusión y pérdida auxiliar)",
        deviations=("Rama auxiliar con pooling adaptativo a 5x5 para unificar resoluciones",)),
}

TEACHER_SPEC = ModelSpec("EffNetB4-Teacher", "Profesor para destilación", "Tan & Le (2019). EfficientNet.",
                         "10.48550/arXiv.1905.11946", "teacher", base_lr=3e-4)

ABLATIONS: dict[str, tuple[str, dict, dict]] = {
    "Celik-HC/sin_hypercolumn": ("Celik-HC", {"layers": "last_only"}, {}),
    "Celik-HC/solo_efficientnet": ("Celik-HC", {"use_inception": False}, {}),
    "Celik-HC/solo_inception": ("Celik-HC", {"use_effnet": False}, {}),
    "Celik-HC/capas_intermedias": ("Celik-HC", {"layers": "model_1"}, {}),
    "Celik-HC/sin_preentrenamiento": ("Celik-HC", {}, {"pretrained": False, "base_lr": 1e-3, "normalization": "train"}),
    "ViT-B16/sin_preentrenamiento": ("ViT-B16", {}, {"pretrained": False, "base_lr": 3e-4, "normalization": "train"}),
    "GFN/sin_ppm": ("GFN", {"use_ppm": False}, {}),
    "GFN/sin_cbam": ("GFN", {"use_cbam": False}, {}),
    "RWCNet/sin_auxiliares": ("RWCNet", {"use_aux": False}, {"loss": "ce"}),
    "FocusRCNet/sin_destilacion": ("FocusRCNet", {}, {"loss": "ce"}),
    "FocusRCNet/sin_simam": ("FocusRCNet", {"use_simam": False}, {}),
}


def get_spec(name: str) -> ModelSpec:
    if name in SPECS:
        return SPECS[name]
    if name == TEACHER_SPEC.name:
        return TEACHER_SPEC
    if name in ABLATIONS:
        base, kw, over = ABLATIONS[name]
        s = SPECS[base]
        d = asdict(s)
        d.update({"name": name, "kwargs": {**s.kwargs, **kw}, **over})
        d["deviations"] = tuple(d["deviations"])
        return ModelSpec(**d)
    raise ValueError(f"Modelo desconocido: {name}")


def build_model(name: str, dropout: float = 0.3, num_classes: int = C.NUM_CLASSES) -> nn.Module:
    s = get_spec(name)
    pre = s.pretrained and PRETRAINED
    kw = dict(s.kwargs)
    if s.builder == "celik":
        return CelikHC(num_classes, dropout, pretrained=pre, **kw)
    if s.builder == "mbxception":
        return MbXception(num_classes, dropout, **kw)
    if s.builder == "vit":
        return ViTB16(num_classes, dropout, pretrained=pre, **kw)
    if s.builder == "gfn":
        return GFN(num_classes, dropout, pretrained=pre, **kw)
    if s.builder == "focus":
        return FocusRCNet(num_classes, dropout, **kw)
    if s.builder == "capsule":
        return ResMsCapsule(num_classes, dropout, pretrained=pre, img_size=C.IMG_SIZE, **kw)
    if s.builder == "rwcnet":
        return RWCNet(num_classes, dropout, pretrained=pre, **kw)
    if s.builder == "teacher":
        return EfficientNetB4Teacher(num_classes, dropout, pretrained=pre)
    raise ValueError(s.builder)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (sum(p.numel() * p.element_size() for p in model.parameters())
               + sum(b.numel() * b.element_size() for b in model.buffers())) / 2**20
    return {"params_total": int(total), "params_trainable": int(trainable), "size_mb": float(size_mb)}


@torch.no_grad()
def count_flops(model: nn.Module, size: int = C.IMG_SIZE) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    was = model.training
    model.eval()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    dev = next(model.parameters()).device
    try:
        with FlopCounterMode(display=False) as fc:
            model(torch.zeros(1, 3, size, size, device=dev))
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
        model.train(was)
    return int(fc.get_total_flops())


def model_info(model: nn.Module, name: str) -> dict:
    s = get_spec(name)
    info = count_params(model)
    flops = count_flops(model)
    info.update({"model": name, "family": s.family, "reference": s.reference, "doi": s.doi,
                 "pretrained": bool(s.pretrained and PRETRAINED), "normalization": s.normalization, "loss": s.loss,
                 "post_hoc": s.post_hoc, "defines_architecture": s.defines_architecture,
                 "deviations": list(s.deviations), "flops": flops, "gflops": flops / 1e9, "macs": flops // 2,
                 "gmacs": flops / 2e9, "input": [3, C.IMG_SIZE, C.IMG_SIZE]})
    return info
