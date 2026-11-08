# Clasificación de residuos con TrashNet

Proyecto final de Deep Learning, Universidad del Norte (profesor: Dr. Lihki Rubio).

**Autores:** Anyely Sanjuan Benavides (Msc. Analítica de Datos) y Juan David Díaz Fernández (Msc. Ingeniería Mecánica).

**Informe (Jupyter Book):** https://anyita-12.github.io/deep-learning-colaborativo/

## Descripción

Comparamos tres arquitecturas de redes convolucionales entrenadas desde cero con el mismo protocolo (CNN estilo VGG reducida, MobileNetV2 y ResNet-18) en la clasificación de imágenes de TrashNet en seis clases de material: `cardboard`, `glass`, `metal`, `paper`, `plastic` y `trash`. La métrica primaria es el F1 macro en el conjunto de prueba. Cada modelo se entrena con 8 semillas y la comparación usa pruebas estadísticas pareadas con corrección de Holm. Las decisiones del diseño experimental se fijaron antes del entrenamiento y están en el capítulo 6 del informe.

## Resultados principales

| Modelo | F1 macro (media ± DE, 8 semillas) |
|---|---|
| ResNet-18 | 0.863 ± 0.022 |
| CNN estilo VGG reducida | 0.826 ± 0.022 |
| MobileNetV2 | 0.809 ± 0.025 |

La prueba de Friedman rechaza la igualdad entre los tres modelos ($p = 0.0048$). ResNet-18 supera a los otros dos con diferencias significativas y mayores que el margen de relevancia práctica ($\delta = 0.02$). El detalle está en los capítulos 4, 5 y 7 del informe.

## Estructura del repositorio

```
data/                datos y particiones (las imágenes no se versionan; ver "Datos")
  splits/            partición fija (CSV)
  SHA256SUMS.txt     huellas de verificación de los datos
docs/                informe en Jupyter Book: capítulos en .md y notebooks 02, 03 y 40–47
notebooks/           exploración inicial
src/                 código del experimento (datos, modelos, entrenamiento, evaluación, estadística)
results/             resultados por corrida y resúmenes (CSV, pickles, preregistro)
models/              checkpoints (no versionados)
environment.yml      entorno conda (alternativa)
requirements.txt     dependencias base
requirements-modelos.txt   dependencias del entrenamiento y del análisis estadístico
requirements-gpu.txt       dependencias para GPU
requirements-docs.txt      dependencias para compilar el Jupyter Book
requirements-lock.txt      versiones exactas del entorno de las corridas finales
```

Los pesos entrenados (`.pt`), el entorno virtual y las imágenes no se suben al repositorio.

## Instalación

Usamos Python 3.12 y PyTorch con CUDA. Los pasos corresponden a Windows con PowerShell.

```powershell
git clone https://github.com/Anyita-12/deep-learning-colaborativo.git
cd deep-learning-colaborativo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Las versiones exactas del entorno en que corrimos el experimento están en `requirements-lock.txt`.

Instalamos PyTorch con el comando que genera https://pytorch.org/get-started/locally/ para la versión de CUDA del equipo, por ejemplo:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-modelos.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Si PowerShell bloquea la activación del entorno: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Datos

Usamos TrashNet (Thung, 2017), disponible en https://github.com/garythung/trashnet. Las imágenes se colocan en `data/raw/` y no se versionan. La partición (1800 entrenamiento, 360 validación y 361 prueba) se genera en el notebook `docs/41_particion_y_preprocesamiento.ipynb` y queda en `data/splits/`. Los hashes de verificación están en `data/SHA256SUMS.txt`; el código comprueba la huella del conjunto de prueba antes de cada corrida. La sección 1.6 del informe describe el dataset, su licencia y la verificación de integridad.

## Reproducción del experimento

Todos los comandos se ejecutan desde la raíz del repositorio, con el entorno activado.

1. Estimar el tiempo total: `python -m src.runner estimate`
2. Prueba rápida del flujo completo: `python -m src.runner all --smoke`
3. Entrenamiento completo (búsqueda de hiperparámetros, piloto, cálculo del número de semillas y corridas finales): `python -m src.runner all`
4. Pruebas de robustez: `python -m src.runner robustness`
5. Avance, desde otra terminal: `python -m src.runner status`

Si el proceso se interrumpe, se vuelve a ejecutar `python -m src.runner all`: omite las corridas terminadas y continúa las parciales desde la última época guardada. El entrenamiento completo tomó cerca de 3.5 horas en una GPU NVIDIA RTX 4060 Laptop (24 corridas finales de unos 8 minutos cada una, más la búsqueda de hiperparámetros y el piloto).

Después, se ejecutan en orden los notebooks de `docs/` (40 a 47). Los notebooks 45, 46 y 47 deben correr en ese orden, porque el 47 lee archivos que generan el 45 y el 46. Estos notebooks leen los resultados de `results/`; no entrenan.

## Reproducibilidad

- Los tres modelos comparten partición, semillas y protocolo de entrenamiento.
- Cada corrida guarda su configuración, historia por época, predicciones, métricas y tiempos en `results/<modelo>/seed_XX/`.
- El conjunto de prueba se evalúa una vez por corrida, con el modelo de la mejor época. La normalización y los pesos de clase se calculan solo con el conjunto de entrenamiento.
- El preregistro de las decisiones (métricas, umbrales, partición y análisis) está en `results/common/preregistro_2026-11-08.md`.
- Los resultados dependen del hardware: la precisión mixta y algunas operaciones de cuDNN introducen una variación residual pequeña, y los tiempos no son comparables entre máquinas.

## Compilar el informe

```powershell
pip install -r requirements-docs.txt
jupyter-book build docs/ --all
ghp-import -n -p -f docs/_build/html
```

## Flujo de trabajo del equipo

Ramas `feature/<tarea>`, pull request revisado por la otra persona y merge a `main`.

## Uso de inteligencia artificial

Usamos Claude (Anthropic) como apoyo en la programación, el análisis de datos y la redacción. Los autores revisamos, verificamos y aprobamos todo el contenido y asumimos la responsabilidad por él.

## Referencia

Thung, G. (2017). *trashnet* [Conjunto de datos y código fuente]. GitHub. https://github.com/garythung/trashnet
