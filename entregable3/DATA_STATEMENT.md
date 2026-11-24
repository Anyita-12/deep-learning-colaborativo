# Declaración de datos

| Conjunto | Fuente | Licencia | Uso en el proyecto | Preprocesamiento |
|---|---|---|---|---|
| TrashNet | Yang y Thung (2016), https://github.com/garythung/trashnet (`dataset-resized.zip`) | Según el repositorio original | Entrenamiento, validación (desarrollo) y prueba (confirmación) | Partición v2 agrupada por casi duplicados (`data/splits/trashnet_split_v2_s1234.csv`), 6 imágenes contradictorias retiradas, redimensionado a 224 × 224 |
| RealWaste | UCI ML Repository 908 (Single, Iranmanesh y Raad, 2023) | CC BY 4.0 | Solo evaluación, sin reentrenar | Mapeo a 6 clases, sin casi duplicados de TrashNet, ≤ 500 por clase (semilla 2026) |
| Garbage Dataset v2 | Kaggle `sumn2u/garbage-classification-v2` (Kunwar, 2026) | Declarada en la página del conjunto | Solo evaluación | Igual; se retiraron 7220 casi duplicados de TrashNet |
| WaRP-C | Kaggle `parohod/warp-waste-recycling-plant-dataset` (Yudin et al., 2024) | MIT | Solo evaluación | Solo `test_crops`; mapeo a 4 clases |

**Datos personales.** Ninguno de los conjuntos contiene personas identificables según su documentación. No se recolectaron datos nuevos.

**Sesgos conocidos.**
- TrashNet se tomó con fondo blanco, una sola pieza por foto y en un solo lugar.
- La clase trash es pequeña (≈ 5 %) y heterogénea.
- Los conjuntos adicionales tienen otra distribución de clases y de fondos, y por eso miden generalización fuera de dominio.

**Registro.** Fecha de descarga, hash y conteos de cada conjunto adicional en `entregable3/results/extra/*.json`. Hash de la partición de prueba: `4e4651dc…cbd87`, verificado en cada una de las 232 corridas.
