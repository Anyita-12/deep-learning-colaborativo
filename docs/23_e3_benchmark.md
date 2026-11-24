# 23. Benchmark: configuraciones, semillas y artefactos

## 23.1 Mejor configuración por modelo

Los diez modelos del E2 conservan la configuración elegida en su búsqueda de hiperparámetros (8 configuraciones por modelo, semilla 100, criterio F1 macro de validación; capítulos 13 y 14). NAGFuse recibió **el mismo presupuesto**: 8 configuraciones (factor de tasa $\{1; 0.3\}$ × weight decay $\{10^{-4}; 5\times10^{-4}\}$ × dropout $\{0.2; 0.4\}$), semilla 100, 30 épocas, mismo criterio (`entregable3/results/results_e2/NAGFuse/hpo`).

El entrenamiento final es el mismo para todos: AdamW, calentamiento de 3 épocas y coseno, lote 32, hasta 60 épocas con paciencia 12, precisión mixta, entropía cruzada ponderada por clase, los mismos aumentos y la misma entrada de 224 × 224. Ningún modelo se entrena con ruido.

## 23.2 Semillas nuevas

| Grupo | Modelos | Semillas |
|---|---|---|
| Nivel 1 | Los 10 modelos del Top 10 | 0–11 (8 del E2 + 4 nuevas) |
| Nivel 2 | NAGFuse, RWCNet, ViT-B16, GFN, Celik-HC, ResMsCapsule | 0–19 |
| Ablación | NAGFuse sin contracción, solo ViT, solo CNN | 0–11 |

Las semillas nuevas de los rivales usan exactamente el código congelado del E1 y del E2 (`src/` y `entregable2/src/`, sin modificaciones). Cómputo del E3: 144 corridas, 19.1 h en una A100 (Colab Pro): 9.4 h para los rivales del E2, 2.0 h para los del E1, 6.9 h para NAGFuse y su ablación, 0.9 h para su búsqueda.

**Desviación declarada.** El profesor EfficientNet-B4 de FocusRCNet no estaba versionado en el E2; se reentrenó con la misma configuración y semilla 100, como preveía el protocolo.

## 23.3 Artefactos exportados

NAGFuse (semilla 0) se exportó en dos formatos con verificación de ida y vuelta sobre 12 imágenes (4 limpias, 4 con $\sigma = 0.05$, 4 con $\sigma = 0.10$):

| Formato | Tamaño | Diferencia máxima en probabilidades | Diferencia en $\hat\sigma$ |
|---|---|---|---|
| TorchScript | 361 MB | $1.8 \times 10^{-7}$ | — |
| ONNX (opset 17) | 360 MB | $9.2 \times 10^{-6}$ | 0 |

Ambas están por debajo del umbral de $10^{-4}$. Para exportar se reemplazaron dos operaciones sin equivalente en ONNX por formas matemáticamente idénticas: la mediana por ordenamiento (mediana inferior) y `logaddexp` por $m + \log(e^{a-m} + e^{b-m})$ con $m = \max(a, b)$. El modelo exportado devuelve cinco salidas: $p$, $\hat\sigma$, $\pi$, $p_V$ y $p_C$.

Metadatos (`nagfuse_seed00_meta.json`): 89 818 505 parámetros, $s_{lo} = 0.01454$, $\rho = 2$, $\pi_0 = 0.509$, entrada normalizada con media y desviación de ImageNet, SHA-256 de los pesos y de cada archivo exportado.
