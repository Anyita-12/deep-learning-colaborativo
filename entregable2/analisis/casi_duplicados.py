"""Auditoría de casi-duplicados entre particiones por similitud de embeddings (Sección 10.1 del enunciado).

Cada imagen se representa con el vector de 2048 dimensiones de ResNet-50 (ImageNet, torchvision) tras el
pooling global. Para cada imagen de prueba se busca la imagen más parecida de entrenamiento y validación
(similitud coseno). Los pares por encima del umbral se inspeccionan visualmente y la decisión (mismo
objeto o no) se registra en results/complementario/casi_duplicados_revision.csv.

Uso: python entregable2/analisis/casi_duplicados.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision import models as tvm

from comun import OUT, REPO, split_table

RAW = REPO / "data" / "raw" / "trashnet" / "dataset-resized"
UMBRAL = 0.93  # fijado antes de inspeccionar los pares


@torch.no_grad()
def embeddings(paths):
    w = tvm.ResNet50_Weights.IMAGENET1K_V2
    net = tvm.resnet50(weights=w).eval()
    net.fc = torch.nn.Identity()
    tf = w.transforms()
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]])
        out.append(torch.nn.functional.normalize(net(x), dim=1).numpy())
    return np.concatenate(out)


def main():
    torch.set_num_threads(8)
    s = split_table().sort_values("relpath").reset_index(drop=True)
    cache = OUT / "_embeddings_resnet50.npz"
    if cache.exists():
        E = np.load(cache)["E"]
    else:
        E = embeddings([RAW / r for r in s.relpath])
        np.savez_compressed(cache, E=E)
    te = np.flatnonzero(s.split == "test")
    ot = np.flatnonzero(s.split != "test")
    S = E[te] @ E[ot].T
    j = S.argmax(1)
    res = pd.DataFrame({"prueba": s.relpath[te].to_numpy(), "vecino": s.relpath[ot[j]].to_numpy(),
                        "particion_vecino": s.split[ot[j]].to_numpy(), "similitud": S.max(1)})
    res = res.sort_values("similitud", ascending=False)
    res.to_csv(OUT / "casi_duplicados_embeddings.csv", index=False)
    cand = res[res.similitud >= UMBRAL]
    print(f"pares con similitud >= {UMBRAL}: {len(cand)}")
    W, H = 150, 112
    canvas = Image.new("RGB", ((W * 2 + 20) * 2, (H + 14) * ((len(cand) + 1) // 2)), "white")
    dr = ImageDraw.Draw(canvas)
    for q, r in enumerate(cand.itertuples()):
        x, y = (q % 2) * (W * 2 + 20), (q // 2) * (H + 14)
        canvas.paste(Image.open(RAW / r.prueba).resize((W, H)), (x, y + 12))
        canvas.paste(Image.open(RAW / r.vecino).resize((W, H)), (x + W + 4, y + 12))
        dr.text((x + 2, y), f"{q}: {r.prueba.split('/')[1]} | {r.vecino.split('/')[1]} ({r.particion_vecino}) {r.similitud:.3f}",
                fill="black")
    canvas.save(OUT / "_casi_duplicados_candidatos.png")
    print(cand.to_string())


if __name__ == "__main__":
    main()
