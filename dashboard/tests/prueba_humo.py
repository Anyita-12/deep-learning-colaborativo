"""Prueba de humo del dashboard (Sección 10.1, paso 6 del enunciado).

1. Dentro del proceso: construye cada página y ejecuta cada callback con una entrada de ejemplo, midiendo el tiempo.
2. Si se pasa una URL (por ejemplo la de Render), comprueba que responde con HTTP 200 en cada ruta.
Uso:  python dashboard/tests/prueba_humo.py [https://<servicio>.onrender.com]
El resultado se imprime y se guarda en dashboard/tests/prueba_humo_resultado.txt para anexarlo al informe.
"""
import base64
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import app  # noqa: E402

lineas = []


def paso(nombre, f, *a):
    t0 = time.perf_counter()
    try:
        f(*a); estado = "OK"
    except Exception as e:  # noqa: BLE001
        estado = f"FALLA: {e!r}"
    lineas.append(f"{nombre:<45} {estado:<8} {1000 * (time.perf_counter() - t0):8.0f} ms")


for ruta in ["/", "/eda", "/modelos", "/ficha"]:
    paso(f"página {ruta}", app.enrutar, ruta)
paso("tabla y curva de ruido", app.actualizar, ["NAGFuse", "ViT-B16"])
paso("curvas de entrenamiento", app.curvas, ["NAGFuse", "ViT-B16"])
for m in app.DIAG:
    paso(f"diagnóstico {m}", app.diagnostico, m)
paso("explorador de imagen de prueba", app.detalle, 10, "ruido_0.05")
img = "data:image/jpeg;base64," + base64.b64encode((HERE / "assets" / "trashnet_clases.jpg").read_bytes()).decode()
if app.MODEL_PATH.exists():
    paso("predicción en vivo NAGFuse (ONNX, CPU)", app.clasificar, img, [])
    paso("predicción en vivo + oclusión", app.clasificar, img, ["oc"])
else:
    lineas.append("predicción en vivo: modelo ONNX no disponible en este entorno (se muestra el aviso)")
    paso("aviso sin modelo", app.clasificar, img, [])

if len(sys.argv) > 1:
    base = sys.argv[1].rstrip("/")
    for ruta in ["/", "/eda", "/modelos", "/ficha", "/_dash-layout"]:
        t0 = time.perf_counter()
        try:
            cod = urllib.request.urlopen(base + ruta, timeout=120).status
        except Exception as e:  # noqa: BLE001
            cod = repr(e)
        lineas.append(f"GET {base + ruta:<60} {cod}  {1000 * (time.perf_counter() - t0):8.0f} ms")

txt = f"Prueba de humo · {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lineas)
print(txt)
(Path(__file__).parent / "prueba_humo_resultado.txt").write_text(txt, encoding="utf-8")
sys.exit(1 if any("FALLA" in x for x in lineas) else 0)
