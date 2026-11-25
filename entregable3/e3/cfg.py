"""Configuración del Entregable 3. Todo lo que fija el protocolo está aquí y en experiments/protocolo.md.

Este módulo no importa torch ni los paquetes `src` del E1/E2, para que el orquestador pueda leerlo sin GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

E3_ROOT = Path(__file__).resolve().parents[1]          # entregable3/
REPO = E3_ROOT.parent                                   # raíz del repositorio
E2_ROOT = REPO / "entregable2"

# ------------------------------------------------------------------ rutas de trabajo (Colab)
WORK_BASE = Path(os.environ.get("E3_WORK", "/content/e3_work"))
DRIVE_BASE = Path(os.environ.get("E3_DRIVE", "/content/drive/MyDrive/E3_TrashNet"))
SMOKE = os.environ.get("E3_SMOKE") == "1"
WORK = WORK_BASE / "_smoke" if SMOKE else WORK_BASE      # resultados de la prueba de humo, aparte
DRIVE = DRIVE_BASE / "_smoke" if SMOKE else DRIVE_BASE

DATA = WORK_BASE / "data"                # TRASHNET_DATA del E2 (raw/ e interim/), compartido con la prueba de humo
RESULTS = {"e2": WORK / "results_e2",    # E2 y NAGFuse comparten el árbol (misma partición y common/)
           "e1": WORK / "results_e1"}
EXTRA_DIR = WORK_BASE / "extra"          # cachés de los conjuntos adicionales (datos, no resultados)
DRIVE_RESULTS = {"e2": DRIVE / "results_e2", "e1": DRIVE / "results_e1"}
DRIVE_CKPT = DRIVE / "checkpoints"       # espejo de checkpoint_last.pt de las corridas en curso
DRIVE_CLAIMS = DRIVE / "claims"          # reservas de tareas (varias sesiones en paralelo)
DRIVE_WEIGHTS = DRIVE / "weights"        # pesos conservados para exportación, dashboard y Grad-CAM
DRIVE_EXTRA = DRIVE_BASE / "extra"
DRIVE_LOGS = DRIVE / "logs"
NOISE_STATS = RESULTS["e2"] / "common" / "nagfuse_noise_stats.json"

# El checkpoint se escribe en disco local cada época. A Drive solo se refleja en corridas largas (cada 25 min):
# en Drive montado, cada archivo sobrescrito o borrado pasa a la papelera y sigue ocupando cuota, y un
# checkpoint de NAGFuse pesa ~1.5 GB. Las corridas terminadas sí se copian siempre a Drive.
MIRROR_EVERY_S = int(os.environ.get("E3_MIRROR_EVERY_S", "1500"))
CLAIM_STALE_S = 45 * 60                                              # reserva abandonada tras 45 min sin latido

# ------------------------------------------------------------------ protocolo (ver experiments/protocolo.md)
MODEL = "NAGFuse"
NAG_VARIANTS = ["NAGFuse/sin_denoising", "NAGFuse/solo_vit", "NAGFuse/solo_cnn"]
E2_MODELS = ["Celik-HC", "MbXception", "ViT-B16", "GFN", "FocusRCNet", "ResMsCapsule", "RWCNet"]
E1_MODELS = ["CNN", "MobileNetV2", "ResNet-18"]
SEEDS_MAIN = list(range(12))             # R = 12 para toda la familia confirmatoria (nivel 1)
SEEDS_RIVAL_NEW = list(range(8, 12))     # los rivales ya tienen 0-7 congelados en el E2
SEEDS_EXT = list(range(12, 20))          # nivel 2: R = 20 para NAGFuse y los rivales con sigma_d cercana a delta
TOP_RIVALS = ["RWCNet", "ViT-B16", "GFN", "Celik-HC", "ResMsCapsule"]
FREEZE = "2026-11-23 06:00 (America/Bogota)"   # el análisis confirmatorio usa lo terminado hasta esta hora
SEEDS_ABL_CORE = list(range(6))          # ablación: 6 semillas obligatorias
SEEDS_ABL_EXTRA = list(range(6, 12))     # y 6 más si alcanza el cómputo (se declara cuántas se corrieron)
HPO_TRIALS = list(range(8))
HPO_SEED = 100
BASE_LR_NAG = 1e-4                       # régimen de ajuste fino de Transformers (E2, sección 11.4)
RHO = 2.0                                # compuerta: abre en s_lo y queda abierta del todo en rho * s_lo
RHO_SENSITIVITY = [1.5, 3.0]             # análisis de sensibilidad (exploratorio)
LAMBDA_BRANCH = 0.5                      # peso de la supervisión de cada rama
NOISE_QUANTILE = 0.99                    # s_lo = cuantil 0.99 de sigma_hat en las imágenes limpias de train

# ------------------------------------------------------------------ opción B (protocolo v4, experiments/protocolo_v4_opcion_B.md)
# Cola que ejecuta el orquestador: "a" = cola congelada de los protocolos v1-v3 (por defecto, no cambia nada);
# "b_piloto" = piloto en VALIDACIÓN de los candidatos B1-B3 (sin conjunto de prueba);
# "b" = corridas confirmatorias del candidato elegido.
COLA = os.environ.get("E3_COLA", "a").strip().lower()
MODEL_B = "NAGFuse-B"
CANDIDATOS_B = ["NAGFuse-B1", "NAGFuse-B2", "NAGFuse-B3"]   # ver VARIANT_KW en nagfuse.py
PILOTO_TRIAL = 1                         # mismo intento de la búsqueda que eligió la versión A (lr 1e-4, wd 1e-4,
                                         # dropout 0.4, 30 épocas máx., semilla 100): F1 val de A = 0.9688
REF_PILOTO_A = 0.9688057                 # F1 de validación de A en ese mismo intento (hpo_selected.pkl)
ELEGIDO_B = "NAGFuse-B2"                 # fijado tras el piloto con la regla del protocolo v4
SEEDS_B = list(range(20))                # mismas semillas que NAGFuse A y los 5 rivales más cercanos

# Pesos que se conservan en Drive (el resto se borra tras la evaluación ampliada).
KEEP_WEIGHTS = {MODEL: [0, 1, 2], MODEL_B: [0, 1, 2], **{m: [8] for m in E2_MODELS + E1_MODELS}}

# ------------------------------------------------------------------ conjuntos adicionales (evaluación sin reentrenar)
EXTRA_SEED = 2026
EXTRA_MAX_PER_CLASS = 500
DEDUP_HAMMING = 6                        # dHash de 64 bits; <= 6 se considera casi duplicado de TrashNet
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTRA_DATASETS = {
    "realwaste": {
        "fuente": "UCI ML Repository, id 908 (Single, Iranmanesh y Raad, 2023)",
        "licencia": "CC BY 4.0",
        "url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
        "dominio": "vertedero real, fondos desordenados, 524x524",
        "mapa": {"Cardboard": "cardboard", "Glass": "glass", "Metal": "metal", "Paper": "paper",
                 "Plastic": "plastic", "Miscellaneous Trash": "trash"},
    },
    "gd_v2": {
        "fuente": "Kaggle sumn2u/garbage-classification-v2 (Kunwar, 2026)",
        "licencia": "declarada en la página del conjunto (se registra al descargar)",
        "kaggle": "sumn2u/garbage-classification-v2",
        "dominio": "fotos de celular, web y comunidad; fondos variados",
        "mapa": {"cardboard": "cardboard", "glass": "glass", "metal": "metal", "paper": "paper",
                 "plastic": "plastic", "trash": "trash"},
    },
    "warp_c": {
        "fuente": "Kaggle parohod/warp-waste-recycling-plant-dataset, parte WaRP-C (Yudin et al., 2024)",
        "licencia": "MIT",
        "kaggle": "parohod/warp-waste-recycling-plant-dataset",
        "dominio": "banda transportadora de planta de reciclaje; recortes de 40 a 700 px; solo test_crops",
        "mapa": "regla_warp",            # ver extra_data.warp_label
    },
}


def task_id(t: dict) -> str:
    parts = [t["fam"], t["kind"], t["model"].replace("/", "__")]
    if "trial" in t:
        parts.append(f"t{t['trial']:02d}")
    if "seed" in t:
        parts.append(f"s{t['seed']:02d}")
    return "-".join(parts)


def build_queue() -> list[dict]:
    """Cola según E3_COLA. Sin la variable devuelve la cola congelada de los protocolos v1-v3."""
    if COLA == "b_piloto":
        return _con_id([{"fam": "nag", "kind": "hpo", "model": m, "trial": PILOTO_TRIAL} for m in CANDIDATOS_B])
    if COLA == "b":
        if ELEGIDO_B is None:
            raise SystemExit("E3_COLA=b pero cfg.ELEGIDO_B no está fijado: corre antes el piloto y congela la elección.")
        return _con_id([{"fam": "nag", "kind": "final", "model": MODEL_B, "seed": s} for s in SEEDS_B])
    return build_queue_a()


def _con_id(q: list[dict]) -> list[dict]:
    for t in q:
        t["id"] = task_id(t)
    return q


def build_queue_a() -> list[dict]:
    """Orden de prioridad. Dentro de cada bloque se intercalan semillas para que, si el tiempo se acaba,
    todos los modelos queden con una cobertura pareja (lección de la Tarea 2)."""
    q: list[dict] = []
    q += [{"fam": "nag", "kind": "hpo", "model": MODEL, "trial": t} for t in HPO_TRIALS]
    rivals = [{"fam": "e2", "model": m} for m in E2_MODELS] + [{"fam": "e1", "model": m} for m in E1_MODELS]
    ours = iter(SEEDS_MAIN)
    for s in SEEDS_RIVAL_NEW:                 # 2 semillas propias, luego 1 semilla de los 10 rivales, ...
        for _ in range(2):
            q.append({"fam": "nag", "kind": "final", "model": MODEL, "seed": next(ours)})
        q += [{**r, "kind": "final", "seed": s} for r in rivals]
    q += [{"fam": "nag", "kind": "final", "model": MODEL, "seed": s} for s in ours]
    for s in SEEDS_ABL_CORE:                  # ablación obligatoria (6 semillas)
        q += [{"fam": "nag", "kind": "final", "model": v, "seed": s} for v in NAG_VARIANTS]
    for s in SEEDS_EXT:                       # nivel 2 de potencia, en orden fijo de semillas
        q.append({"fam": "nag", "kind": "final", "model": MODEL, "seed": s})
        q += [{"fam": "e2", "kind": "final", "model": m, "seed": s} for m in TOP_RIVALS]
    for s in SEEDS_ABL_EXTRA:                 # ablación ampliada si alcanza el cómputo
        q += [{"fam": "nag", "kind": "final", "model": v, "seed": s} for v in NAG_VARIANTS]
    for t in q:
        t["id"] = task_id(t)
    return q


def smoke_queue() -> list[dict]:
    if COLA in ("b", "b_piloto"):
        # prueba de humo de la opción B: un intento del piloto por candidato y una corrida final con el primero
        q = [{"fam": "nag", "kind": "hpo", "model": "NAGFuse", "trial": 0}]
        q += [{"fam": "nag", "kind": "hpo", "model": m, "trial": PILOTO_TRIAL} for m in CANDIDATOS_B]
        q += [{"fam": "nag", "kind": "final", "model": CANDIDATOS_B[0], "seed": 0}]
        return _con_id(q)
    q = [{"fam": "nag", "kind": "hpo", "model": MODEL, "trial": 0},
         {"fam": "nag", "kind": "final", "model": MODEL, "seed": 0},
         {"fam": "nag", "kind": "final", "model": "NAGFuse/solo_cnn", "seed": 0},
         {"fam": "e2", "kind": "final", "model": "ResMsCapsule", "seed": 8},
         {"fam": "e2", "kind": "final", "model": "Celik-HC", "seed": 8},
         {"fam": "e1", "kind": "final", "model": "CNN", "seed": 8}]
    for t in q:
        t["id"] = task_id(t)
    return q
