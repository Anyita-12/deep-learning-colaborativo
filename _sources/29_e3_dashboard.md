# 29. Dashboard

El dashboard (Dash + Plotly) tiene tres páginas y una ficha del modelo y lee **la misma tabla maestra que este libro** (`dashboard/data/tabla_maestra.csv`, copia de `entregable3/resultados/`). La prueba automática `dashboard/tests/test_consistencia.py` verifica dos cosas: que los CSV del libro y del dashboard son idénticos byte a byte (SHA-256) y que las cifras de los capítulos 27 y 28 coinciden con ellos.

| Página | Contenido |
|---|---|
| Contexto | Problema, tarjetas con las cifras principales e IC, veredicto de cada hipótesis y predicción, funcionamiento de NAGFuse |
| EDA | Imágenes por clase y partición, distribución de $\hat\sigma$ con los umbrales de la compuerta y resumen de los conjuntos adicionales |
| Modelos | Tabla de líderes ordenable y filtrable (familia, F1 macro media y DE, ECE, parámetros, GFLOPs, latencia); tabla de métricas con IC y curva de robustez; bosques de H1 y H2; diagnóstico del modelo seleccionado (matriz de confusión, ROC y precisión-exhaustividad por clase, diagrama de confiabilidad); diagrama de diferencia crítica y fronteras de Pareto frente a GFLOPs y latencia; interpretabilidad global (peso $\pi$ y F1 de cada rama por condición); subsección de NAGFuse (procedencia, pseudocódigo, curvas de entrenamiento con media ± DE entre semillas, tabla de pruebas con p de Holm y tamaños de efecto, ablación y veredicto de no superioridad); explorador de las 361 imágenes de prueba; clasificación de una imagen propia con latencia en CPU |
| Ficha del modelo | `MODEL_CARD.md` completo dentro de la aplicación |

**Inferencia.**

- **Solo en CPU**, con el modelo exportado a ONNX y `onnxruntime`. La caché guarda a lo sumo dos sesiones.
- **Salidas por imagen:** probabilidades de la mezcla y de cada rama, $\hat\sigma$, $\pi$, entropía normalizada y conjunto conforme al 90 % (umbral calibrado en validación).
- **Avisos automáticos** cuando:
  - se detecta ruido, con la compuerta abierta o parcialmente abierta;
  - el conjunto conforme no tiene exactamente una clase;
  - la imagen tiene menos del 20 % de fondo claro y probablemente está fuera de dominio.
- **Interpretabilidad:** un mapa de oclusión (rejilla de 4 × 4) muestra qué regiones sostienen la predicción. Se usa oclusión en lugar de Grad-CAM porque funciona sobre el modelo exportado sin acceso a gradientes.

**Despliegue.** Render (`dashboard/render.yaml`): Python 3.11.9 y `gunicorn app:server --workers 1 --threads 2 --timeout 120`. El modelo ONNX (360 MB) no cabe en la memoria del plan gratuito (512 MB). En ese plan, el dashboard muestra todo excepto la clasificación de imágenes nuevas y lo indica en pantalla. Con el modelo disponible (variable `NAGFUSE_ONNX`, contenedor con volumen o plan de más memoria) la clasificación funciona igual.

**Datos de la página Modelos.** Además de la tabla maestra, `entregable3/analisis/datos_dashboard_diagnostico.py` reorganiza, sin recalcular nada, los historiales por época (`curvas.csv`), las probabilidades en prueba de una semilla por modelo (`probs_prueba.csv.gz`, la menor semilla guardada) y el costo medido durante el entrenamiento (`costo.csv`: mediana entre semillas de parámetros, GFLOPs y latencia).

**Prueba de humo.** `dashboard/tests/prueba_humo.py` construye cada página, ejecuta cada callback con una entrada de ejemplo (incluida una predicción en vivo si el ONNX está disponible) y, si se le pasa la URL pública, comprueba que cada ruta responde. Su salida se anexa en el capítulo 31.

**Opción B.** La tabla de líderes, las curvas de entrenamiento y el diagnóstico por modelo incluyen a NAGFuse-B (capítulo 32), con los datos de la misma tabla maestra. La predicción en vivo usa el modelo exportado de la versión A; NAGFuse-B no se exportó.

**Limitaciones del dashboard (declaradas).**

- La aplicación enruta con `dcc.Location` en un solo `app.py` en lugar de `use_pages=True` y una carpeta `pages/`, y usa una hoja de estilos propia en lugar de `dash-bootstrap-components`. La separación entre interfaz, datos y artefactos se mantiene: la aplicación no importa PyTorch ni código de entrenamiento.
- La explicación local es un mapa de oclusión, no Grad-CAM: el modelo exportado a ONNX no expone gradientes en CPU. La interpretabilidad global se apoya en el propio mecanismo ($\pi$ y desempeño de cada rama), no en un método de atribución con prueba de fidelidad.
- La latencia de la tabla de líderes se midió en GPU (A100, AMP, lote de 1) durante el entrenamiento; la latencia en CPU solo se mide para NAGFuse al clasificar una imagen. No se midieron memoria en inferencia ni energía, y los baselines desde cero no registraron FLOPs.
- No se calculó la curva de aprendizaje frente al tamaño de muestra (10–50 % del entrenamiento) por falta de presupuesto de cómputo.
- Solo NAGFuse tiene inferencia en vivo; los demás modelos se muestran con resultados precalculados (Sección 9.2 del enunciado).

**Contingencia para la sustentación.** Ejecución local (`python app.py`) con el modelo, y capturas y un video corto de cada página por si falla la red.
