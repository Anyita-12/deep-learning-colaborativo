# 17. Reproducibilidad del benchmark

En este capítulo describimos cómo regenerar los resultados de los capítulos 8 a 16 a partir del repositorio. Incluimos la ficha de ruta metodológica y la lista de verificación de reproducibilidad del Apéndice B del enunciado.

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
    ├── ablation/             ablaciones de Celik-HC (5 variantes) y RWCNet (1 variante), 8 semillas cada una
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

No construimos una imagen de contenedor. El enunciado la declara opcional en esta etapa, y queda pendiente para la siguiente.

## 17.3 Comandos

**Entrenamiento** (desde `entregable2/`, con GPU; ver `entregable2/README.md`):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.runner estimate --models RWCNet          # estimación de horas de GPU
.venv\Scripts\python.exe -m src.runner all --models MbXception FocusRCNet ResMsCapsule RWCNet
.venv\Scripts\python.exe -m src.runner ablation --variants Celik-HC/sin_preentrenamiento RWCNet/sin_auxiliares
.venv\Scripts\python.exe -m src.runner replicate                          # réplica del paper guía
.venv\Scripts\python.exe -m src.runner summary                            # tablas de results/summary
```

Todas las etapas se pueden reanudar: las corridas con `DONE` se saltan y las parciales continúan desde la última época guardada.

**Análisis complementarios** (desde la raíz del repositorio, en CPU):

```powershell
.venv\Scripts\python.exe entregable2\analisis\complementos.py          # estadística, calibración, conformal, robustez, errores, eficiencia, ablaciones
.venv\Scripts\python.exe entregable2\analisis\baselines.py             # baselines trivial y clásico, sanidad con etiquetas permutadas
.venv\Scripts\python.exe entregable2\analisis\casi_duplicados.py       # auditoría de casi-duplicados con embeddings
.venv\Scripts\python.exe entregable2\analisis\revision_e2.py puntuar   # cribado y puntaje de los 62 estudios
```

`baselines.py` y `casi_duplicados.py` necesitan las imágenes en `data/raw/trashnet/dataset-resized/`. La estabilidad de Grad-CAM y la figura de ejemplos necesitan los archivos `gradcam.npz`, cuya ruta se indica con la variable `TRASHNET_E2_FULL`.

**Informe.** Los cuadernos 09 a 15 leen las tablas de `entregable2/results/` y no entrenan. Se ejecutan en orden y se guardan con sus salidas, porque el libro se compila sin ejecutar celdas.

## 17.4 Evidencia de reproducción

- **Misma máquina:** el reentrenamiento de las 8 semillas de ViT-B16 en una sesión nueva de Colab reprodujo la misma mejor época, el mismo F1 y el 100 % de las predicciones de prueba (sección 11.6). La discrepancia máxima observada fue 0.
- **Tablas de resumen:** `python -m src.runner summary` regenera las tablas de `results/summary/` a partir de las corridas guardadas, sin GPU.
- **Combinación de máquinas:** las corridas de Colab y del portátil se combinaron en una sola carpeta, y la huella del conjunto de prueba se verificó en ambas máquinas antes de cada corrida.

No se hizo una reproducción independiente por otra persona en un entorno limpio, y la declaramos como pendiente.

## 17.5 Ficha de ruta metodológica

| Elemento | Valor |
|---|---|
| Tipo de problema | 7. Clasificación de imágenes (familia F4) |
| Ruta seguida | F4a (sección 7.4 del enunciado), pasos I1 a I8 (sección 8.2) |
| Unidad de independencia | Grupo de casi-duplicados; dependencia por objeto auditada en la sección 9.4 |
| Métrica primaria | F1 macro en prueba; $\delta = 0.02$ |
| Bondad de ajuste (tabla A) | Binomial frente al baseline trivial, permutación de etiquetas, AUC frente a 0.5 y MCC frente a 0 con IC bootstrap por grupo, pendiente e intercepto de calibración, Z de Spiegelhalter (sección 14.1) |
| Sobreajuste (tabla B) | Brecha entre entrenamiento y validación con margen (Wilcoxon y t unilaterales, Holm), validación frente a prueba con IC bootstrap por grupo, validación adversarial, sensibilidad a la semilla, Grad-CAM contra atajos (secciones 9.3, 14.2 y 15.2) |
| Diferencias significativas (tabla C) | Friedman con permutación e Iman-Davenport, ANOVA de medidas repetidas con Greenhouse-Geisser, t pareada, Wilcoxon, bootstrap pareado, Conover-Holm y Nemenyi con diagrama de diferencia crítica, TOST, prueba bayesiana con ROPE, McNemar, Q de Cochran, DeLong (capítulo 13) |
| Tamaños de efecto | $d_z$, rango-biserial, $A_{12}$ de Vargha-Delaney, $W$ de Kendall, $\eta^2$ e ICC(1) |
| Familias y corrección | 45 pares del benchmark (Holm, preregistrada); 7 contrastes frente al modelo de referencia; 5 variantes de la ablación de Celik-HC; 20 contrastes frente a los baselines |

## 17.6 Lista de verificación de reproducibilidad (Apéndice B)

| Estado | Ítem | Evidencia |
|---|---|---|
| ✅ | Preregistro fechado antes del primer entrenamiento | `entregable2/results/common/preregistro_2026-11-14.md` (20:12) y `preregistro_colab_2026-11-15.md`; sin *release* etiquetada previa (sección 11.1) |
| ✅ | Hash del conjunto de datos y del conjunto de prueba | `data/SHA256SUMS.txt`; SHA-256 de cada imagen en la partición (sección 9.4) |
| ✅ | Auditoría de fuga ejecutada | Sección 9.4; la dependencia por objeto se cuantificó con un análisis de sensibilidad |
| ✅ | Particiones y semillas guardadas | `data/splits/trashnet_split_v2_s1234.csv`; semillas en `config.json` de cada corrida |
| ✅ | Un pipeline por modelo, sin ajustes fuera de entrenamiento | Sección 9.4 (fuga por transformaciones) y capítulo 11 |
| ✅ | Mismo presupuesto de ajuste y configuraciones registradas | 8 configuraciones por modelo en `results/<Modelo>/hpo/` y `busqueda_hiperparametros.csv` |
| ✅ | Al menos 6 ejecuciones por modelo y resultados por ejecución | 8 semillas; `metricas_por_corrida.csv` |
| ✅ | Análisis de potencia | Capítulo 4 y sección 13.8 |
| ✅ | Baselines trivial, clásico y los tres modelos base | Sección 13.2; los modelos base conservan el protocolo del capítulo 4 |
| ✅ | Versiones, hardware y determinismo documentados | Secciones 11.6 y 17.2 |
| ✅ | Tablas A, B y C con $p$ exactos, ajustados, IC y tamaños de efecto | Capítulos 13 y 14 |
| 🟡 | Familias de comparaciones declaradas a priori | La familia de 45 pares sí; la prueba de decisión se aclaró antes de ver los resultados (sección 11.2) |
| ✅ | Techo de desempeño y $\gamma$ | Sección 15.4 |
| — | Búsqueda de arquitectura | No aplica: los siete artículos definen la arquitectura |
| 🟡 | Robustez con tres niveles; fidelidad y estabilidad de las explicaciones | Tres niveles en ruido y brillo; estabilidad calculada; fidelidad pendiente (sección 15.2) |
| ✅ | Incertidumbre en los tres mejores modelos con cobertura empírica | MC Dropout y predicción conforme (sección 14.4) |
| ✅ | Un solo acceso al conjunto de prueba por corrida | El código evalúa la prueba una vez, con el modelo de la mejor época de validación |
| ✅ | Réplica del paper guía con criterio a priori y limitación | Capítulo 12 |
| 🟡 | Reproducción independiente de la tabla principal | Reproducción exacta de ViT-B16 en la misma plataforma; sin reproducción por otra persona |
| 🟡 | Código de extremo a extremo con un comando y entorno reproducible | `python -m src.runner all` y `requirements-lock.txt`; sin contenedor |
