"""Figura de ejemplos de Grad-CAM (requiere los gradcam.npz completos; ruta en TRASHNET_E2_FULL)."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from comun import FULL, OUT, REPO, load_pickle
import figuras as FG

FG.setup()
RAW = REPO / "data" / "raw" / "trashnet" / "dataset-resized"
modelos = ["RWCNet", "ViT-B16", "GFN", "Celik-HC"]
z0 = np.load(FULL / "RWCNet" / "seed_00" / "gradcam.npz", allow_pickle=True)
tp = load_pickle(FULL / "RWCNet" / "seed_00" / "test_predictions.pkl")
pos = z0["test_positions"]
elegidas = [int(np.flatnonzero(z0["y_true"] == c)[0]) for c in range(6)]
fig, axes = plt.subplots(len(modelos) + 1, 6, figsize=(12, 2.0 * (len(modelos) + 1)))
for j, k in enumerate(elegidas):
    fp = tp["filepaths"][pos[k]]
    rel = "/".join(str(fp).replace("\\", "/").split("/")[-2:])
    im = np.asarray(Image.open(RAW / rel).convert("RGB").resize((224, 224)))
    axes[0, j].imshow(im); axes[0, j].set_title(rel.split("/")[1], fontsize=8)
    for i, m in enumerate(modelos):
        z = np.load(FULL / m / "seed_00" / "gradcam.npz", allow_pickle=True)
        kk = int(np.flatnonzero(z["test_positions"] == pos[k])[0])
        cam = np.asarray(Image.fromarray((z["cam_true"][kk] * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255
        axes[i + 1, j].imshow(im); axes[i + 1, j].imshow(cam, cmap="magma", alpha=0.55)
        if j == 0:
            axes[i + 1, j].set_ylabel(m, fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Grad-CAM respecto a la clase verdadera (semilla 0)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "figura_gradcam_ejemplos.png", dpi=110)
print("ok")
