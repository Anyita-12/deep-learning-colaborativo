# 29. Dashboard

El dashboard (Dash + Plotly) tiene tres páginas y lee **la misma tabla maestra que este libro** (`dashboard/data/tabla_maestra.csv`, copia de `entregable3/resultados/`). La prueba automática `dashboard/tests/test_consistencia.py` verifica dos cosas: que los CSV del libro y del dashboard son idénticos byte a byte (SHA-256) y que las cifras de los capítulos 27 y 28 coinciden con ellos.

| Página | Contenido |
|---|---|
| Contexto | Problema, tarjetas con las cifras principales e IC, veredicto de cada hipótesis y predicción, funcionamiento de NAGFuse |
| EDA | Imágenes por clase y partición, distribución de $\hat\sigma$ con los umbrales de la compuerta y resumen de los conjuntos adicionales |
| Modelos | Selector de modelos, tabla de métricas con IC, curva de robustez, gráficos de bosque de H1 y H2, explorador de las 361 imágenes de prueba con cada perturbación, y clasificación de una imagen propia |

**Inferencia.**

- **Solo en CPU**, con el modelo exportado a ONNX y `onnxruntime`. La caché guarda a lo sumo dos sesiones.
- **Salidas por imagen:** probabilidades de la mezcla y de cada rama, $\hat\sigma$, $\pi$, entropía normalizada y conjunto conforme al 90 % (umbral calibrado en validación).
- **Avisos automáticos** cuando:
  - se detecta ruido, con la compuerta abierta o parcialmente abierta;
  - el conjunto conforme no tiene exactamente una clase;
  - la imagen tiene menos del 20 % de fondo claro y probablemente está fuera de dominio.
- **Interpretabilidad:** un mapa de oclusión (rejilla de 4 × 4) muestra qué regiones sostienen la predicción. Se usa oclusión en lugar de Grad-CAM porque funciona sobre el modelo exportado sin acceso a gradientes.

**Despliegue.** Render (`dashboard/render.yaml`): Python 3.11.9 y `gunicorn app:server --workers 1 --threads 2 --timeout 120`. El modelo ONNX (360 MB) no cabe en la memoria del plan gratuito (512 MB). En ese plan, el dashboard muestra todo excepto la clasificación de imágenes nuevas y lo indica en pantalla. Con el modelo disponible (variable `NAGFUSE_ONNX`, contenedor con volumen o plan de más memoria) la clasificación funciona igual.

**Contingencia para la sustentación.** Ejecución local (`python app.py`) con el modelo, y capturas y un video corto de cada página por si falla la red.
