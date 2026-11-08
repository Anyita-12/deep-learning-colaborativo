# Preregistro del experimento de modelos base

Fecha: 2026-11-08 02:12. Se escribe antes de la búsqueda de hiperparámetros y de las corridas definitivas.

## Hipótesis
- H0 global: las tres arquitecturas (CNN estilo VGG reducida, MobileNetV2 y ResNet-18), entrenadas desde cero con el mismo protocolo, tienen la misma distribución de F1 macro en prueba.
- H1 global: al menos una difiere.
- Por pares: H0: la diferencia media de F1 macro pareada por semilla es 0.

## Métrica, umbrales y unidad de réplica
- Métrica primaria: F1 macro en el conjunto de prueba. Secundarias: accuracy, balanced accuracy, F1 ponderado y por clase, precisión y recall macro, AUC-ROC macro (OvR), AUC-PR macro, MCC y ECE.
- alpha = 0.05; potencia objetivo = 0.8; delta (relevancia práctica) = 0.02 en F1 macro.
- Unidad de réplica: la semilla (inicialización, orden de lotes, aumentación y dropout). Las tres arquitecturas comparten partición y semillas: datos pareados por semilla.

## Datos
- Partición v2 del EDA (`trashnet_split_v2_s1234.csv`): 1800/360/361, estratificada y agrupada por casi-duplicados, sin las 6 imágenes con etiqueta contradictoria. Huella del test verificada antes de cada corrida.
- Entrada: 224x224, redimensionado directo (todas las imágenes son 512x384). Normalización con media y DE por canal de train. Pesos de clase con train.
- Aumentación solo en train: {'RandomResizedCrop': {'size': 224, 'scale': (0.7, 1.0)}, 'RandomHorizontalFlip': {'p': 0.5}, 'RandomVerticalFlip': {'p': 0.5}, 'RandomRotation': {'degrees': 15}, 'ColorJitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.02}}.

## Entrenamiento (igual para los tres)
- Régimen: desde cero (weights=None) para los tres. La CNN no tiene pesos preentrenados; preentrenar solo dos rompería la equidad. La transferencia desde ImageNet queda para el Entregable 2.
- Valores por defecto: {'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.0001, 'dropout': 0.3, 'batch_size': 32, 'max_epochs': 100, 'warmup_epochs': 3, 'eta_min_factor': 0.01, 'patience': 15, 'min_delta': 0.0001, 'grad_clip': 5.0, 'amp': True, 'class_weighted_loss': True}.
- Búsqueda de hiperparámetros: malla de 8 configuraciones, idéntica para los tres modelos, semilla 100, máximo 50 épocas, selección por F1 macro de validación. El test no se usa.

| trial | lr | weight_decay | dropout |
|---|---|---|---|
| 0 | 0.001 | 0.0001 | 0.2 |
| 1 | 0.001 | 0.0001 | 0.4 |
| 2 | 0.001 | 0.0005 | 0.2 |
| 3 | 0.001 | 0.0005 | 0.4 |
| 4 | 0.0003 | 0.0001 | 0.2 |
| 5 | 0.0003 | 0.0001 | 0.4 |
| 6 | 0.0003 | 0.0005 | 0.2 |
| 7 | 0.0003 | 0.0005 | 0.4 |

## Tamaño de muestra (piloto interno)
- Orden: búsqueda de hiperparámetros, luego piloto con las semillas [0, 1, 2, 3, 4] y la configuración elegida. Las corridas del piloto son las primeras corridas definitivas (misma configuración).
- R = (z_(1-alpha/2) + z_(1-beta))^2 * sd_d^2 / delta^2 por par de modelos, con sd_d de las diferencias pareadas de F1 macro de validación del piloto. Se toma el máximo y se redondea hacia arriba.
- R_final = min(max(R, 6), 10). El mínimo de 6 permite pruebas por rangos; el máximo de 10 lo impone el tiempo de GPU. Si R supera 10, se reporta la diferencia mínima detectable (MDE) como limitación.

## Análisis planeado
- Global: Friedman (bloques = semillas, tratamientos = 3 modelos) con p exacto por permutación y W de Kendall.
- Pares (3): t pareada y Wilcoxon de rangos con signo; diferencia media con IC 95 % (t y bootstrap); d_z y r rango-biserial; p sin ajustar y ajustados por Holm. Nemenyi y Conover como complemento.
- Por imagen: McNemar por par y semilla (exacto si b + c < 25), con Holm dentro de cada semilla.
- AUC: bootstrap estratificado pareado sobre las imágenes de prueba (10000 remuestreos).
- Una diferencia es relevante si |diferencia media| >= delta y significativa tras Holm.
- Robustez: ruido gaussiano (0.05 y 0.10), desenfoque, brillo +-30 %, oclusión y fondo sustituido (hallazgo del EDA, 2.7).
