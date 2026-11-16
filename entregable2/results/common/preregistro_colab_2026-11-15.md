# Preregistro del benchmark del Entregable 2

Fecha: 2026-11-15 00:46. Se escribe antes de la búsqueda de hiperparámetros y de las corridas definitivas.

## Elementos heredados del Entregable 1 (sin cambios)
- Métrica primaria: F1 macro en prueba. alpha = 0.05, delta = 0.02, potencia objetivo = 0.8.
- Unidad de réplica: la semilla. Semillas [0, 1, 2, 3, 4, 5, 6, 7] (R = 8, el R_final del Entregable 1), compartidas por todos los modelos.
- Partición v2 (`trashnet_split_v2_s1234.csv`, 1800/360/361), huella del test verificada antes de cada corrida.
- Entrada 224x224 por redimensionado directo, aumentación solo en train: {'RandomResizedCrop': {'size': 224, 'scale': (0.7, 1.0)}, 'RandomHorizontalFlip': {'p': 0.5}, 'RandomVerticalFlip': {'p': 0.5}, 'RandomRotation': {'degrees': 15}, 'ColorJitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.02}}.
- Entropía cruzada ponderada por la inversa de la frecuencia de clase (o la pérdida propia del modelo con los mismos pesos).

## Modelos
| modelo | familia | preentrenado | lr base | normalización | pérdida | arquitectura definida |
|---|---|---|---|---|---|---|
| Celik-HC | Híbrido DL + ML (fusión multinivel de CNN + XGBoost) | sí | 0.0003 | imagenet | ce + XGBoost | sí (código oficial) |
| MbXception | CNN multirrama (Xception modificada) | no | 0.001 | train | ce | sí (código oficial) |
| ViT-B16 | Transformer (ViT con ajuste fino) | sí | 0.0001 | imagenet | ce | parcial (ViT estándar; variante no declarada, se asume B/16) |
| GFN | Híbrido CNN-Transformer con atención (ResNet50 + PPM + CBAM + ViT) | sí | 0.0001 | imagenet | ce | sí (dimensiones de cada etapa en el texto) |
| FocusRCNet | CNN ligera con atención y destilación de conocimiento | no | 0.001 | imagenet | kd | sí (Tabla 1) |
| ResMsCapsule | Red de cápsulas con extractor residual | sí | 0.0003 | imagenet | margin | parcial (estructura y dimensiones de cápsulas) |
| RWCNet | Fusión de CNN preentrenadas con salidas auxiliares | sí | 0.0003 | imagenet | aux | sí (puntos de fusión y pérdida auxiliar) |

Normalización: estadísticos de ImageNet para los modelos con pesos preentrenados (exigido por los pesos) y media y DE de train para los entrenados desde cero.

## Desviaciones respecto a los papers
- Celik-HC: PyTorch en lugar de Keras con pesos ImageNet de torchvision; Dropout antes de la primera capa densa para el espacio de búsqueda común; Selección de época con validación en lugar del conjunto de prueba
- MbXception: Entrada 224x224 en lugar de 229x229; AdamW en lugar de SGD con momento; Dropout antes del clasificador para el espacio de búsqueda común
- ViT-B16: Variante ViT-B/16 asumida; Lote 32 en lugar de 512
- GFN: Proyección 2048x7x7 -> 1000x4x4 mediante conv 1x1 y pooling adaptativo (el texto no la detalla); Embeddings de posición inicializados por interpolación de los de ViT-B/16
- FocusRCNet: Entrada 224x224 en lugar de 380x380; T = 4.0 y alpha = 0.7 (no declarados); Factor de reducción del bloque sandglass = 2 (no declarado)
- ResMsCapsule: ResNet-18 hasta layer3 con pesos ImageNet; 32 tipos de cápsula por rama y núcleos 5/3/1 (no declarados); El paper reporta 4 clases; aquí se usan las 6 de TrashNet
- RWCNet: Rama auxiliar con pooling adaptativo a 5x5 para unificar resoluciones

## Entrenamiento (igual para los siete)
- AdamW, calentamiento lineal de 3 épocas y coseno hasta lr x 0.01, lote efectivo 32, recorte de gradiente 5.0, precisión mixta.
- Máximo 60 épocas en las corridas definitivas y 30 en la búsqueda; parada temprana con paciencia 12 sobre F1 macro de validación.
- Búsqueda de hiperparámetros: malla de 8 configuraciones para cada modelo (mismo número de intentos), semilla 100. La tasa de aprendizaje es lr base x factor, con lr base según el régimen (desde cero 1e-3; ajuste fino de CNN 3e-4; ajuste fino de Transformers 1e-4), de modo que la amplitud relativa del espacio es la misma. Selección por F1 macro de validación; el test no se usa.

| trial | factor lr | weight decay | dropout |
|---|---|---|---|
| 0 | 1.0 | 0.0001 | 0.2 |
| 1 | 1.0 | 0.0001 | 0.4 |
| 2 | 1.0 | 0.0005 | 0.2 |
| 3 | 1.0 | 0.0005 | 0.4 |
| 4 | 0.3 | 0.0001 | 0.2 |
| 5 | 0.3 | 0.0001 | 0.4 |
| 6 | 0.3 | 0.0005 | 0.2 |
| 7 | 0.3 | 0.0005 | 0.4 |

- FocusRCNet: profesor EfficientNet-B4 (ImageNet) entrenado una vez con la partición de train y seleccionado en validación; destilación con T = 4.0 y alpha = 0.7.
- Celik-HC: la parte convolucional se entrena con el protocolo común; después XGBoost se ajusta solo con los embeddings de train. Se reportan ambas salidas (XGBoost y cabeza softmax).

## Análisis planeado
- Global: Friedman (bloques = semillas) sobre los diez modelos (siete nuevos y los tres base) con p por permutación y W de Kendall.
- Pares: t pareada y Wilcoxon, IC 95 % (t y bootstrap), d_z, r rango-biserial, Holm. Relevante si |diferencia| >= delta y p Holm < alpha.
- Por imagen: McNemar por par y semilla con Holm.
- Calibración: ECE, NLL y Brier antes y después del escalado de temperatura ajustado en validación.
- Incertidumbre: MC Dropout (20 pasadas) e IC bootstrap por semilla.
- Robustez: ruido gaussiano (0.05 y 0.10), desenfoque, brillo +-30 %, oclusión y fondo sustituido (hallazgo del EDA).
- Interpretabilidad: Grad-CAM sobre 10 imágenes de prueba por clase y fracción de la atención sobre el fondo.
- Replicación del paper guía (Celik, 2025) bajo su protocolo original y bajo el protocolo estricto.
