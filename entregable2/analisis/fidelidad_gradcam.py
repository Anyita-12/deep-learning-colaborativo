"""Fidelidad de los mapas Grad-CAM: pruebas de borrado e inserción (Petsiuk et al., 2018).

Para cada imagen de la muestra de Grad-CAM (10 por clase) se retiran (borrado) o se insertan (inserción)
los píxeles en orden decreciente de activación del mapa de la clase verdadera, y se mide la probabilidad
de la clase verdadera. Un mapa fiel produce una caída más rápida al borrar (menor área bajo la curva) y
una subida más rápida al insertar (mayor área) que un orden aleatorio de píxeles. Los píxeles retirados
se sustituyen por la media del conjunto de entrenamiento (cero tras la normalización).

Requiere los pesos best_model.pt y los gradcam.npz de cada corrida (no versionados) y el código del
benchmark. Uso:
    python entregable2/analisis/fidelidad_gradcam.py <carpeta results con pesos> <carpeta del código>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

RESW, CODE = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(CODE))
from src.models import build_model  # noqa: E402
from src import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "trashnet" / "dataset-resized"
OUT = REPO / "entregable2" / "results" / "complementario"
MODELOS = ["RWCNet", "ResMsCapsule", "MbXception", "FocusRCNet"]
FRACS = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
N_RANDOM = 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm_train = __import__("pickle").load(open(RESW / "common" / "norm_stats.pkl", "rb"))


def probs(model, x):
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    return torch.softmax(out.float(), 1)


def curva(model, x, y, orden, modo):
    """x: (3, H, W) normalizada; orden: índices de píxeles de mayor a menor importancia."""
    H, W = x.shape[1:]
    n = H * W
    lotes = []
    for f in FRACS:
        k = int(round(f * n))
        mask = torch.zeros(n, device=dev)
        mask[orden[:k]] = 1
        mask = mask.view(1, H, W)
        lotes.append(x * (1 - mask) if modo == "borrado" else x * mask)
    p = probs(model, torch.stack(lotes))[:, y].cpu().numpy()
    return np.trapezoid(p, FRACS)


def main():
    rows = []
    rng = np.random.default_rng(2026)
    for m in MODELOS:
        for s in range(8):
            run = RESW / m / f"seed_{s:02d}"
            cfg = json.load(open(run / "config.json"))
            model = build_model(m, dropout=cfg["hp"]["dropout"]).to(dev).eval()
            model.load_state_dict(torch.load(run / "best_model.pt", map_location=dev, weights_only=False))
            kind = "imagenet" if m != "MbXception" else "train"
            mean, std = (C.IMAGENET_MEAN, C.IMAGENET_STD) if kind == "imagenet" else (norm_train["mean"], norm_train["std"])
            mt, st_ = torch.tensor(mean, device=dev).view(3, 1, 1), torch.tensor(std, device=dev).view(3, 1, 1)
            z = np.load(run / "gradcam.npz", allow_pickle=True)
            tp = __import__("pickle").load(open(run / "test_predictions.pkl", "rb"))
            for k, pos in enumerate(z["test_positions"]):
                rel = "/".join(str(tp["filepaths"][pos]).replace("\\", "/").split("/")[-2:])
                im = np.asarray(Image.open(RAW / rel).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255
                x = (torch.from_numpy(im).permute(2, 0, 1).to(dev) - mt) / st_
                y = int(z["y_true"][k])
                cam = torch.from_numpy(z["cam_true"][k].astype(np.float32))[None, None]
                cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False).view(-1).to(dev)
                orden = torch.argsort(cam, descending=True)
                fila = {"model": m, "seed": s, "imagen": rel, "clase": y}
                for modo in ("borrado", "insercion"):
                    fila[f"auc_{modo}_cam"] = curva(model, x, y, orden, modo)
                    fila[f"auc_{modo}_azar"] = np.mean([curva(model, x, y, torch.from_numpy(rng.permutation(224 * 224)).to(dev), modo)
                                                         for _ in range(N_RANDOM)])
                rows.append(fila)
            print(m, s, flush=True)
            del model
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fidelidad_gradcam.csv", index=False)
    print(df.groupby("model")[["auc_borrado_cam", "auc_borrado_azar", "auc_insercion_cam", "auc_insercion_azar"]].mean().round(3))


if __name__ == "__main__":
    main()
