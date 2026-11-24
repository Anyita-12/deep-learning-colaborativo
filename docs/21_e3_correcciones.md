# 21. Correcciones del Entregable 2

## 21.1 Trazabilidad
 La tabla recoge las correcciones que encontramos al revisar nuestro propio trabajo; la última fila queda abierta.

| N.° | Hallazgo | Dónde | Corrección | Efecto en resultados |
|---|---|---|---|---|
| 1 | El texto decía que el ruido gaussiano se añade "sobre la imagen normalizada"; el código (`entregable2/src/analysis.py`, `perturb`) lo añade a la imagen en $[0, 1]$ **antes** de normalizar y recorta a $[0, 1]$ | Capítulos 12 (sección de limitaciones) y 15 (definición de perturbaciones) | Texto corregido para describir lo que hace el código. Se fija la definición única: $\sigma$ en la escala de píxeles $[0, 1]$, igual para todos los modelos | Ninguno: todas las cifras se calcularon con el código. La escala es la misma para todos los modelos, sin importar su normalización |
| 2 | La salida guardada del cuaderno 47 muestra `FileNotFoundError` con una ruta de Windows | `47_analisis_cualitativo_robustez.ipynb`, celda 15 | La ruta del código ya es relativa (`C.SUMMARY`); el error es una salida antigua, de antes de que existiera `stats.pkl`. Se reejecuta el cuaderno | Ninguno |
| 3 | RWCNet obtiene F1 $0.945$ con las semillas 0–7 (RTX 4060) y $0.931$ con las 8–19 (A100); diferencia $-0.014$, $p = 0.011$ sin corregir (Welch) | Hallazgo exploratorio del E3 | Se declara como posible efecto del hardware. No se filtran semillas: el análisis confirmatorio usa todas, como fija el protocolo | El ranking de los dos primeros cambia (sección 21.3). Ninguna conclusión del E2 depende de la diferencia RWCNet–ViT-B16, que ya era de equivalencia |
| 4 | Observaciones del docente sobre el E2 | — | Pendiente de recibirlas | — |

## 21.2 Auditoría de fuga

- **Huella del conjunto de prueba.** Cada corrida guarda el SHA-256 de la lista de imágenes de prueba en `config.pkl`. Las 232 corridas (E1, E2 y E3) tienen la misma huella (`4e4651dc…cbd87`).
- **Casi duplicados.** La partición v2 se agrupó por casi duplicados en el E2. En los conjuntos adicionales se retiraron las imágenes casi duplicadas de cualquier imagen de TrashNet (dHash, Hamming $\le 6$): 387 en RealWaste, **7220 en Garbage v2** (que incluye buena parte de TrashNet) y 1 en WaRP-C.
- **Decisiones.** Ninguna decisión del E3 usó el conjunto de prueba ni los adicionales (capítulo 27).

## 21.3 Ranking antes y después

| Modelo | F1 semillas 0–7 (E2) | Rango E2 | F1 todas las semillas (E3) | Rango E3 |
|---|---|---|---|---|
| RWCNet | 0.9450 | 1 | 0.9369 | 1 |
| NAGFuse | — | — | 0.9365 | 2 |
| ViT-B16 | 0.9378 | 2 | 0.9363 | 3 |
| GFN | 0.9210 | 4 | 0.9240 | 4 |
| Celik-HC | 0.9139 | 5 | 0.9092 | 5 |
| ResMsCapsule | 0.8990 | 6 | 0.9056 | 6 |
| MbXception | 0.8613 | 8 | 0.8647 | 7 |
| ResNet-18 | 0.8628 | 7 | 0.8627 | 8 |
| CNN | 0.8261 | 9 | 0.8229 | 9 |
| FocusRCNet | 0.8112 | 10 | 0.8080 | 10 |
| MobileNetV2 | 0.8095 | 11 | 0.8049 | 11 |

El orden de los rivales se conserva salvo el intercambio de MbXception y ResNet-18, que difieren en 0.002.

## 21.4 Congelamiento

Los benchmarks quedan congelados con estos SHA-256 (rutas y contenido de cada archivo, en orden), que calcula `entregable3/analisis/congelar.py`. Cualquier cambio posterior cambia el hash; regenerar el análisis desde cero produce los mismos valores.

| Directorio | SHA-256 |
|---|---|
| `results/summary` (E1) | `28987f9386bfa12ec17d4ff881d91b8b9b6f4e90be676d149041a5bc8fc4a385` |
| `entregable2/results/summary` | `f9a837e5c3ea458c98936f7de7908e31e537cea9908a924e52a0ae5cb329d601` |
| `entregable3/results` | `85559d88625eb96255ed8a76357e1890a75aff00bd5f257f221df9e232168953` |
| `entregable3/resultados` | `f479ad1c43f375f40ea0b2dbe81168e619c2d87c73f084a5575d7d8bb5ff4e5d` |
