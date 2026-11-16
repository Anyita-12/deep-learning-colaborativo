# TrashNet: benchmark de siete modelos de la literatura (Entregable 2)

Este repositorio entrena y evalúa siete modelos publicados sobre TrashNet. Usa el mismo protocolo del Entregable 1 (repositorio `deep-learning-colaborativo`, versión `v1.0-entrega1`), así que los tres modelos base (CNN, MobileNetV2 y ResNet-18) entran en la misma comparación estadística.

**Dentro de este repositorio:** el código está en `entregable2/src`, los resultados en `entregable2/results` y los análisis complementarios en `entregable2/analisis`. Las corridas de los modelos base están en `../results`; para regenerar las tablas de resumen, define `TRASHNET_E1_RESULTS` con esa ruta. El informe correspondiente son los capítulos 8 a 17 del Jupyter Book (`docs/`).

## Modelos

| Modelo | Familia | Referencia | Preentrenado | lr base |
|---|---|---|---|---|
| **Celik-HC** (paper guía) | Fusión multinivel EfficientNet-B0 + InceptionV3, con XGBoost sobre el embedding | Celik (2025), *The Visual Computer*, 10.1007/s00371-025-04031-3 | ImageNet | 3e-4 |
| MbXception | CNN multirrama (Xception modificada) | Shi et al. (2020), *IEEE Access*, 10.1109/ACCESS.2020.3016116 | No | 1e-3 |
| ViT-B16 | Vision Transformer con ajuste fino | Huang et al. (2021), *Sustainability*, 10.3390/su132111572 | ImageNet | 1e-4 |
| GFN | ResNet50 + PPM + CBAM + codificador ViT | Wang et al. (2025), *Electronics*, 10.3390/electronics14010075 | ImageNet | 1e-4 |
| FocusRCNet | CNN ligera (Focus, sandglass, SimAM) con destilación desde EfficientNet-B4 | Zheng et al. (2023), *VCIBA*, 10.1186/s42492-023-00146-3 | No (el profesor sí) | 1e-3 |
| ResMsCapsule | ResNet-18 + cápsulas multiescala con enrutamiento dinámico | Huang et al. (2023), *ESPR*, 10.1007/s11356-023-27970-7 | ImageNet | 3e-4 |
| RWCNet | Fusión DenseNet201 + MobileNetV2 con salidas auxiliares | Hossen et al. (2024), *IEEE Access*, 10.1109/ACCESS.2024.3354774 | ImageNet | 3e-4 |

Celik-HC y MbXception se implementaron a partir de los repositorios oficiales de sus autores. Los otros cinco siguen la descripción de su artículo. Cada modelo registra sus desviaciones respecto al paper en `model_info.pkl`, y todas quedan en `summary/tabla_modelos.csv`.

## Protocolo

- **Partición v2 del E1** (`data/splits/trashnet_split_v2_s1234.csv`): 1800 / 360 / 361 imágenes. Excluye 6 imágenes con etiqueta contradictoria y mantiene cada grupo de casi duplicados dentro de una sola partición. Antes de entrenar se verifican el hash de cada imagen y la huella del conjunto de prueba.
- **Métrica primaria:** F1 macro. Diferencia mínima relevante δ = 0,02; α = 0,05.
- **Entrenamiento común:**
  - AdamW, warmup de 3 épocas y luego coseno, AMP.
  - Entropía cruzada ponderada por clase.
  - Aumentación del E1 y entrada de 224×224.
  - Lote efectivo de 32.
  - Máximo 60 épocas, con parada temprana de paciencia 12 sobre el F1 de validación.
- **Búsqueda de hiperparámetros:** grilla de 8 configuraciones por modelo (factor de lr {1; 0,3} × weight decay {1e-4; 5e-4} × dropout {0,2; 0,4}), semilla 100, 30 épocas. La configuración se elige solo con validación.
- **Corridas finales:** 8 semillas (0–7) por modelo. El conjunto de prueba se evalúa una sola vez por corrida.
- **Pérdidas específicas:**
  - FocusRCNet: destilación con T = 4 y α = 0,7.
  - ResMsCapsule: margin loss.
  - RWCNet: pérdidas auxiliares con pesos 1 / 0,5 / 0,25.
  - Celik-HC: XGBoost entrenado sobre el embedding de 64 dimensiones.

El preregistro completo se genera en `results/common/preregistro_<fecha>.md` antes de la primera corrida.

## Estructura

```
trashnet-e2/
├── README.md
├── requirements.txt
├── data/
│   ├── splits/trashnet_split_v2_s1234.csv   partición v2 heredada del E1
│   ├── SHA256SUMS.txt                       huellas del conjunto de prueba y del zip
│   └── e1_results/                          corridas finales de los 3 modelos base del E1
├── notebooks/
│   └── entrenamiento_colab_kaggle.ipynb     cuaderno para Colab y Kaggle
└── src/
    ├── config.py          rutas, hiperparámetros y constantes del protocolo
    ├── data.py            descarga, verificación, caché y cargadores
    ├── models/            los 7 modelos, el profesor y las variantes de ablación
    ├── losses.py          CE ponderada, auxiliar, margin y destilación
    ├── train.py           entrenamiento reanudable y evaluación final
    ├── evaluate.py        métricas, calibración, MC Dropout y tiempos
    ├── analysis.py        robustez a perturbaciones y Grad-CAM
    ├── stats.py           Friedman, post hoc, Wilcoxon/t con Holm, McNemar
    ├── design.py          espacio de búsqueda, preregistro y selección
    ├── replication.py     replicación del paper guía
    ├── report.py          tablas de resumen y empaquetado
    └── runner.py          línea de comandos
```

TrashNet (43 MB) se descarga automáticamente la primera vez desde el repositorio original.

## Ejecución

### Colab Pro (recomendado para ViT-B16 y GFN)

1. Sube `trashnet-e2.zip` a la raíz de *Mi unidad*.
2. Abre `notebooks/entrenamiento_colab_kaggle.ipynb` y selecciona una GPU (L4 o A100).
3. Ajusta `MODELOS` en la primera celda y ejecuta todo.

Cada corrida terminada se copia de inmediato a `MyDrive/trashnet-e2-salidas/results/`. Si la sesión se corta, vuelve a ejecutar el cuaderno: restaura lo respaldado y continúa.

### Kaggle

1. Sube `trashnet-e2.zip` como Dataset y añádelo al cuaderno.
2. Activa *Internet* (para los pesos de ImageNet) y la GPU.
3. Usa *Save Version → Save & Run All* para que las salidas queden en `/kaggle/working`. El límite es de 12 h por sesión.

### Local (Windows con GPU NVIDIA)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m src.runner estimate --models MbXception FocusRCNet ResMsCapsule RWCNet Celik-HC
python -m src.runner all --models MbXception FocusRCNet ResMsCapsule RWCNet Celik-HC
```

Con 8 GB de VRAM, si aparece un error de memoria en RWCNet o Celik-HC, define `set TRASHNET_ACCUM=2` y repite el comando. El lote efectivo se mantiene en 32.

### Etapas

| Comando | Qué hace |
|---|---|
| `python -m src.runner status` | Avance de la búsqueda, semillas terminadas y corridas parciales |
| `python -m src.runner estimate --models ...` | Mide segundos por época (3 épocas) y estima las horas de GPU |
| `python -m src.runner all --models ...` | Búsqueda de hiperparámetros, 8 semillas, resumen y empaquetado |
| `python -m src.runner hpo` / `final` | Las mismas etapas por separado (`--seeds` limita las semillas) |
| `python -m src.runner ablation` | Variantes del paper guía; `--variants` acepta otras (ver `src/models/__init__.py`) |
| `python -m src.runner replicate` | Replicación del paper guía con su protocolo original y con el estricto |
| `python -m src.runner gradcam --models ...` | Recalcula los Grad-CAM vacíos o ausentes; si la corrida no conservó sus pesos, reentrena la misma configuración y semilla y registra la concordancia con las predicciones originales en `gradcam_origen.json` |
| `python -m src.runner summary` | Regenera `results/summary` con todo lo terminado |
| `python -m src.runner package` | Comprime `results/` sin checkpoints en `resultados_entrega2_<fecha>.zip` |

Todas las etapas son reanudables: las corridas con `DONE` se saltan y las parciales continúan desde la última época guardada.

### Reparto entre máquinas

Cada máquina ejecuta `all` con su propio subconjunto de modelos. Al terminar, se copian las carpetas `results/<Modelo>/` (y `results/ablation/`, `results/replication/` si existen) a una sola carpeta `results/`, y allí se ejecuta `python -m src.runner summary` y luego `python -m src.runner package`. La partición y la caché son idénticas en todas las máquinas, así que los resultados se pueden combinar directamente.

### Variables de entorno

| Variable | Uso | Valor por defecto |
|---|---|---|
| `TRASHNET_RESULTS` | Carpeta de salidas | `./results` |
| `TRASHNET_DATA` | Datos descargados y caché | `./data` |
| `TRASHNET_BACKUP` | Copia de cada corrida terminada (por ejemplo, una carpeta de Drive) | sin copia |
| `TRASHNET_BACKUP_WEIGHTS` | `1` para incluir los pesos `.pt` en la copia | `0` |
| `TRASHNET_ACCUM` | Pasos de acumulación de gradiente | `1` |
| `TRASHNET_WORKERS` | `num_workers` del DataLoader | `min(4, CPU)` |
| `TRASHNET_E1_RESULTS` | Corridas de los modelos base del E1 | `./data/e1_results` |

## Salidas

### Por corrida: `results/<Modelo>/seed_XX/`

| Archivo | Contenido |
|---|---|
| `config.json`, `env.pkl` | Hiperparámetros, versión del código, GPU y versiones de librerías |
| `model_info.pkl` | Parámetros, tamaño, GMACs, referencia, DOI y desviaciones respecto al paper |
| `history.pkl` | Pérdida y F1 por época en entrenamiento y validación, lr y tiempo por época |
| `val_predictions.pkl`, `test_predictions.pkl` | Probabilidades, etiquetas e índices de cada imagen |
| `metrics_test.pkl` | Accuracy, F1, precisión y recall (macro, ponderado y por clase), AUC-ROC, AUC-PR, MCC, ECE, NLL, Brier y matriz de confusión |
| `metrics_test_head.pkl` | Solo Celik-HC: métricas de la cabeza softmax, para compararlas con XGBoost |
| `calibration.pkl` | Temperatura ajustada en validación y ECE/NLL antes y después |
| `uncertainty_mc_dropout.pkl` | 20 pasadas de MC Dropout: entropía, componente aleatoria e información mutua |
| `robustness.pkl` | F1 bajo ruido gaussiano (0,05 y 0,10), desenfoque, brillo ±30, oclusión y sustitución del fondo (las mismas perturbaciones del E1) |
| `gradcam.npz` | Mapas Grad-CAM de 10 imágenes por clase y fracción de energía en el fondo |
| `timing.pkl` | Tiempo de entrenamiento, épocas, mejor época e inferencia (ms por imagen) |
| `best_model.pt` | Pesos de la mejor época (no se incluyen en el zip de resultados) |

Celik-HC añade `xgb_model.json` y `embeddings.npz`.

### Resumen: `results/summary/`

| Archivo | Sección del entregable |
|---|---|
| `tabla_benchmark.csv` | Benchmark: media ± DE e IC 95 % de todas las métricas para los 10 modelos (7 nuevos y 3 del E1) |
| `tabla_modelos.csv` | Descripción de modelos: familia, referencia, parámetros, GMACs, tamaño y desviaciones |
| `metricas_por_corrida.csv`, `f1_macro_por_semilla.csv` | Datos crudos para gráficos de cajas y tablas por semilla |
| `estadistica_global.json` | Friedman por permutación, W de Kendall y descomposición de varianza entre y dentro de modelos |
| `posthoc_nemenyi.csv`, `posthoc_conover_holm.csv` | Comparaciones post hoc de rangos |
| `pruebas_pares.csv` | Diferencias por pares: IC t y bootstrap, Wilcoxon y t pareada con Holm y tamaño de efecto. La decisión (`relevante_y_significativa`) usa \|diferencia\| ≥ δ y la t pareada con Holm; con 8 semillas y 45 pares, Wilcoxon exacto con Holm no puede bajar de 0,35 (ver `LEEME.txt`) |
| `mcnemar_por_semilla.csv` | McNemar imagen a imagen sobre el mismo conjunto de prueba |
| `diferencia_minima_detectable.csv` | Diferencia mínima detectable con 8 semillas y potencia 0,8 |
| `f1_por_clase.csv` | Análisis de errores por clase |
| `calibracion.csv`, `incertidumbre_mc_dropout.csv` | Confiabilidad de las predicciones |
| `robustez.csv`, `robustez_modelos_base.csv` | Robustez a perturbaciones, comparada con los modelos del E1 |
| `gradcam_fondo.csv` | Interpretabilidad: qué tanto atiende cada modelo al fondo |
| `sobreajuste.csv` | Brecha entre entrenamiento y validación |
| `busqueda_hiperparametros.csv` | Las 8 configuraciones de cada modelo con su F1 de validación |
| `celik_xgb_vs_softmax.csv` | Aporte de XGBoost en el paper guía |
| `ablacion.csv` | Ablación: cada variante contra el modelo completo, con Wilcoxon-Holm |
| `replicacion_paper_guia.csv` | Replicación: valor reportado, valor obtenido, discrepancia e IC bootstrap |

En `results/common/` quedan el preregistro, la partición verificada, las estadísticas de normalización, los pesos de clase, la búsqueda de hiperparámetros, el profesor de FocusRCNet y `requirements-lock.txt`. En `results/logs/` queda el registro de cada ejecución.

## Replicación del paper guía

El código oficial de Celik (2025) hace una partición aleatoria 80/20 sobre las 2527 imágenes y elige la época con mejor accuracy sobre el mismo 20 % que luego reporta como prueba. `replicate` ejecuta dos protocolos, con 5 semillas cada uno:

- **Original:** el del repositorio oficial. Debería acercarse al 99,4 % reportado.
- **Estricto:** la época se elige con un 10 % de validación interno y el 20 % se evalúa una sola vez.

`results/replication/protocol.json` cuantifica además cuántas imágenes de prueba de esa partición tienen un casi duplicado en entrenamiento. La diferencia entre ambos protocolos y frente al benchmark con la partición v2 mide el sesgo optimista del resultado publicado.

## Limitaciones conocidas

- ResMsCapsule se evaluó en el paper con 4 clases; aquí se usan las 6 de TrashNet.
- Huang et al. (2021) no declaran la variante de ViT; se asume B/16.
- FocusRCNet y MbXception usan entrada de 224×224 en lugar de 380 y 229 píxeles, para que todos los modelos tengan el mismo costo de entrada y la misma caché.
- En Kaggle la descarga de los pesos de ImageNet requiere tener activado *Internet*.
