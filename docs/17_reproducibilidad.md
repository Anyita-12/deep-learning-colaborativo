# 17. Reproducibilidad del benchmark

En este capítulo describimos cómo regenerar los resultados de los capítulos 8 a 16 a partir del repositorio. Incluimos la ficha de ruta metodológica y la lista de verificación de reproducibilidad.

## 17.1 Estructura de los archivos

```
entregable2/
├── README.md            etapas del experimento y archivos de salida
├── requirements.txt     dependencias del entrenamiento
├── src/                 código del benchmark (datos, 7 modelos, entrenamiento, evaluación, estadística, réplica)
├── notebooks/           cuaderno para ejecutar el entrenamiento en Colab o Kaggle
├── analisis/            análisis complementarios sobre los resultados guardados (no entrenan redes)
├── revision_sistematica/  sección 3.1: cadenas con fecha y conteo, PRISMA, kappa, hoja de extracción,
│                          tabla de mecanismos, capturas, respuestas crudas de las APIs y scripts
└── results/
    ├── <Modelo>/seed_XX/     80 corridas finales (configuración, historia, predicciones, métricas, tiempos)
    ├── <Modelo>/hpo/         las 8 configuraciones de la búsqueda de cada modelo
    ├── ablation/             ablaciones de Celik-HC (5 variantes) y RWCNet (3 variantes), 8 semillas cada una
    ├── replication/          réplica del paper guía (protocolos original y estricto)
    ├── common/               preregistros, partición verificada, estadísticas de normalización, requirements-lock
    ├── summary/              tablas de resumen del benchmark
    ├── complementario/       tablas de los análisis complementarios
    ├── revision/             cribado de los 62 estudios para el benchmark, hojas PRISMA exportadas y cadenas preliminares
    └── logs/                 registros de ejecución con marca de tiempo (Colab y portátil)
```

Las corridas de los tres modelos base están en `results/` (capítulo 4). Los pesos (`.pt`), los mapas de Grad-CAM (`.npz`) y las imágenes no se versionan por tamaño. Los mapas y los pesos de las corridas del portátil se conservan fuera del repositorio.

## 17.2 Entorno

| Elemento | Valor |
|---|---|
| Python | 3.12.10 (portátil) y 3.13.16 (Colab) |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| CUDA | 13.0 |
| Otras dependencias | Versiones exactas en `entregable2/results/common/requirements-lock.txt` |
| GPU | NVIDIA A100-SXM4 (Colab) y RTX 4060 Laptop (portátil) |
| Sistema operativo | Linux (Colab) y Windows 11 (portátil) |
| Semillas | 0 a 7 en las corridas finales, 100 en la búsqueda y 2026 en las perturbaciones |
| Determinismo | `cudnn.deterministic = True`, `cudnn.benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

El entorno se fija con `requirements.txt` y `requirements-lock.txt` y se reconstruye con `venv`. En la etapa siguiente el entorno se empaquetará además en una imagen de contenedor.

## 17.3 Comandos

**Entrenamiento** (desde `entregable2/`, con GPU; ver `entregable2/README.md`):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.runner estimate --models RWCNet          # estimación de horas de GPU
.venv\Scripts\python.exe -m src.runner all --models MbXception FocusRCNet ResMsCapsule RWCNet
.venv\Scripts\python.exe -m src.runner ablation                           # las 5 variantes del paper guía
.venv\Scripts\python.exe -m src.runner ablation --variants RWCNet/sin_auxiliares RWCNet/solo_densenet201 RWCNet/solo_mobilenetv2
.venv\Scripts\python.exe -m src.runner replicate                          # réplica del paper guía
.venv\Scripts\python.exe -m src.runner summary                            # tablas de results/summary
```

Todas las etapas se pueden reanudar: las corridas con `DONE` se saltan y las parciales continúan desde la última época guardada.

**Análisis complementarios** (desde la raíz del repositorio, en CPU):

```powershell
.venv\Scripts\python.exe entregable2\analisis\complementos.py          # estadística, calibración, conformal, robustez, errores, eficiencia, ablaciones
.venv\Scripts\python.exe entregable2\analisis\baselines.py             # baselines trivial y clásico, sanidad con etiquetas permutadas
.venv\Scripts\python.exe entregable2\analisis\casi_duplicados.py       # auditoría de casi-duplicados con embeddings
.venv\Scripts\python.exe entregable2\analisis\fidelidad_gradcam.py <results con pesos> <código del benchmark>   # fidelidad de Grad-CAM
.venv\Scripts\python.exe entregable2\analisis\revision_e2.py puntuar   # cribado y puntaje de los 62 estudios
```

`baselines.py` y `casi_duplicados.py` necesitan las imágenes en `data/raw/trashnet/dataset-resized/`. La estabilidad de Grad-CAM y la figura de ejemplos necesitan los archivos `gradcam.npz`, cuya ruta se indica con la variable `TRASHNET_E2_FULL`. La prueba de fidelidad necesita además los pesos `best_model.pt` de cada corrida.

**Informe.** Los cuadernos 09 a 15 leen las tablas de `entregable2/results/` y no entrenan. Se ejecutan en orden y se guardan con sus salidas, porque el libro se compila sin ejecutar celdas.

## 17.4 Evidencia de reproducción

- **Misma máquina:** el reentrenamiento de las 8 semillas de ViT-B16 en una sesión nueva de Colab reprodujo la misma mejor época, el mismo F1 y el 100 % de las predicciones de prueba (sección 11.6). La discrepancia máxima observada fue 0.
- **Tablas de resumen:** `python -m src.runner summary` regenera las tablas de [`entregable2/results/summary/`](https://github.com/Anyita-12/deep-learning-colaborativo/tree/main/entregable2/results/summary) a partir de las corridas guardadas, sin GPU.
- **Combinación de máquinas:** las corridas de Colab y del portátil se combinaron en una sola carpeta, y la huella del conjunto de prueba se verificó en ambas máquinas antes de cada corrida.

- **Reproducción independiente de la tabla principal:** recalculamos el F1 macro y la accuracy de las 80 corridas finales directamente desde las predicciones guardadas, con un código distinto del que generó las tablas (`entregable2/results/complementario/reproduccion_tabla_principal.csv`). La discrepancia máxima frente a la tabla principal es 0 en las 80 corridas, y el estadístico de Friedman recalculado ($Q = 64.77$) coincide exactamente.

## 17.5 Ficha de ruta metodológica

| Elemento | Valor |
|---|---|
| Tipo de problema | 7. Clasificación de imágenes (familia F4) |
| Ruta seguida | F4a, clasificación de imágenes, pasos I1 a I8 (sección 8.2) |
| Unidad de independencia | Grupo de casi-duplicados; dependencia por objeto auditada en la sección 9.4 |
| Métrica primaria | F1 macro en prueba; $\delta = 0.02$ |
| Bondad de ajuste (tabla A) | Binomial frente al baseline trivial, permutación de etiquetas, AUC frente a 0.5 y MCC frente a 0 con IC bootstrap por grupo, pendiente e intercepto de calibración, Z de Spiegelhalter (sección 14.1) |
| Sobreajuste (tabla B) | Brecha entre entrenamiento y validación con margen (Wilcoxon y t unilaterales, Holm), validación frente a prueba con IC bootstrap por grupo, validación adversarial, sensibilidad a la semilla, Grad-CAM contra atajos (secciones 9.3, 14.2 y 15.2) |
| Diferencias significativas (tabla C) | Friedman con permutación e Iman-Davenport, ANOVA de medidas repetidas con Greenhouse-Geisser, t pareada, Wilcoxon, bootstrap pareado, Conover-Holm y Nemenyi con diagrama de diferencia crítica, TOST, prueba bayesiana con ROPE, McNemar, Q de Cochran, DeLong (capítulo 13) |
| Tamaños de efecto | $d_z$, rango-biserial, $A_{12}$ de Vargha-Delaney, $W$ de Kendall, $\eta^2$ e ICC(1) |
| Familias y corrección | 45 pares del benchmark (Holm, preregistrada); 7 contrastes frente al modelo de referencia; 5 variantes de la ablación de Celik-HC; 3 variantes de la ablación de RWCNet; 20 contrastes frente a los baselines |

## 17.6 Lista de verificación de reproducibilidad (Apéndice B)

| Ítem | Evidencia |
|---|---|
| Preregistro fechado antes del primer entrenamiento | `entregable2/results/common/preregistro_2026-11-14.md` (20:12) y `preregistro_colab_2026-11-15.md` (sección 11.1) |
| Hash del conjunto de datos y del conjunto de prueba | `data/SHA256SUMS.txt`; SHA-256 de cada imagen en la partición (sección 9.4) |
| Auditoría de fuga ejecutada | Sección 9.4; la dependencia por objeto se cuantificó con un análisis de sensibilidad |
| Particiones y semillas guardadas | `data/splits/trashnet_split_v2_s1234.csv`; semillas en `config.json` de cada corrida |
| Un pipeline por modelo, sin ajustes fuera de entrenamiento | Sección 9.4 (fuga por transformaciones) y capítulo 11 |
| Mismo presupuesto de ajuste y configuraciones registradas | 8 configuraciones por modelo en `results/<Modelo>/hpo/` y `busqueda_hiperparametros.csv` |
| Al menos 6 ejecuciones por modelo y resultados por ejecución | 8 semillas; `metricas_por_corrida.csv` |
| Análisis de potencia | Capítulo 4 y sección 13.10 |
| Baselines trivial, clásico y los tres modelos base | Sección 13.4; los modelos base conservan el protocolo del capítulo 4 |
| Versiones, hardware y determinismo documentados | Secciones 11.6 y 17.2 |
| Tablas A, B y C con $p$ exactos, ajustados, IC y tamaños de efecto | Capítulos 13 y 14 |
| Familias de comparaciones declaradas a priori | Familia de 45 pares fijada en el preregistro, con Holm; la regla de decisión por pares se precisó antes de ver los resultados (sección 11.2) |
| Techo de desempeño y $\gamma$ | Sección 15.4 |
| Búsqueda de arquitectura | No aplica: los siete artículos definen la arquitectura (sección 11.5) |
| Robustez con tres niveles; fidelidad y estabilidad de las explicaciones | Tres niveles con curva de degradación e índice de robustez en ruido y brillo; desenfoque, oclusión y fondo con el nivel preregistrado del capítulo 5. Estabilidad entre semillas en los siete modelos publicados; fidelidad por inserción en RWCNet, el mejor modelo, y en tres modelos más, 8 de 8 semillas por encima del azar; ViT-B16 y GFN se evalúan con estabilidad y atención sobre el fondo (sección 15.2) |
| Incertidumbre en los tres mejores modelos con cobertura empírica | MC Dropout y predicción conforme (sección 14.4) |
| Un solo acceso al conjunto de prueba por corrida | El código evalúa la prueba una vez, con el modelo de la mejor época de validación |
| Réplica del paper guía con criterio a priori y limitación | Capítulo 12: criterio fijado antes de ejecutar, discrepancia con IC y causas; pipeline validado con los siete modelos publicados (sección 12.2) |
| Reproducción independiente de la tabla principal | Recálculo desde las predicciones con código independiente: discrepancia máxima 0 en las 80 corridas; reentrenamiento exacto de ViT-B16 (sección 17.4) |
| Código de extremo a extremo con un comando y entorno reproducible | `python -m src.runner all`, `requirements-lock.txt` y entorno `venv`; contenedor en la etapa siguiente |
