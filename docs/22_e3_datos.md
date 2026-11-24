# 22. Datos

## 22.1 TrashNet

Se usa la misma partición v2 del E2 (`trashnet_split_v2_s1234.csv`): 1800 imágenes de entrenamiento, 360 de validación y 361 de prueba, agrupadas por casi duplicados (2512 grupos), seis clases (cardboard, glass, metal, paper, plastic, trash). La partición no se modificó. El conjunto de validación es la **partición de desarrollo** (todas las decisiones de diseño de NAGFuse); el de prueba es la **partición de confirmación** (una evaluación por corrida).

## 22.2 Conjuntos adicionales

Se eligieron antes de ver cualquier resultado en ellos (`protocolo.md`, sección 9) para que difieran de TrashNet en dominio, fuente y condiciones de captura. Se evalúan **sin reentrenar**: cada corrida entrenada en TrashNet predice sobre cada conjunto, de modo que miden generalización bajo cambio de distribución con un presupuesto de ajuste igual a cero para todos los modelos.

| Conjunto | Fuente y licencia | Diferencia con TrashNet | Imágenes evaluadas | Casi duplicados de TrashNet retirados |
|---|---|---|---|---|
| RealWaste | UCI ML Repository 908 (Single et al., 2023), CC BY 4.0 | Vertedero real, fondos desordenados | 2640 (6 clases) | 387 |
| Garbage Dataset v2 | Kaggle `sumn2u/garbage-classification-v2` (Kunwar, 2026) | Fotos de celular, web y comunidad | 3000 (6 clases, 500 por clase) | 7220 |
| WaRP-C | Kaggle `parohod/warp-waste-recycling-plant-dataset` (Yudin et al., 2024), MIT | Banda transportadora industrial, recortes pequeños y deformados | 863 (4 clases: sin paper ni trash) | 1 |

**Reglas comunes.** Las clases se mapean a las de TrashNet con reglas fijas (`protocolo.md`); se toman como máximo 500 imágenes por clase con semilla 2026; el F1 macro se calcula sobre las clases presentes en cada conjunto (`labels` = clases presentes), igual para todos los modelos. Cada conjunto tiene su registro (`entregable3/results/extra/<conjunto>.json`) con fuente, fecha de descarga, hash y conteos.

**Hallazgo de la auditoría.** Garbage v2 contiene buena parte de TrashNet: 7220 imágenes casi duplicadas. Sin el filtro, ese conjunto habría medido memoria y no generalización.

**Limitaciones.** WaRP-C no tiene las clases paper ni trash; los modelos pueden predecirlas y eso cuenta como error, pero no entran en el promedio. En RealWaste, la clase *Miscellaneous Trash* no coincide exactamente con *trash* de TrashNet.
