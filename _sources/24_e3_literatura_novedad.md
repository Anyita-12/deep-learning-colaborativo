# 24. Literatura y verificación de novedad

## 24.1 Qué se verifica

La afirmación de novedad es acotada: NAGFuse combina (a) un estimador **analítico** del nivel de ruido (mediana de los coeficientes diagonales de Haar, Donoho y Johnstone, 1994) que **fija el peso de una mezcla de dos clasificadores** sin parámetros entrenables en la compuerta, (b) contracción de ondícula solo en la entrada del experto convolucional y (c) una garantía formal: con ruido suficiente la salida coincide exactamente con la del experto ViT (Teorema 2). Ninguno de los componentes es nuevo por separado; la verificación busca si la combinación, o una equivalente, ya existe.

## 24.2 Matriz de búsqueda

Búsquedas realizadas el 22 de noviembre de 2026. Las API de OpenAlex y Semantic Scholar no estuvieron disponibles desde el entorno de trabajo; se usaron los buscadores de cada editorial.

| N.° | Cadena | Base | Resultados revisados | Relevantes |
|---|---|---|---|---|
| C1 | noise level estimation gate mixture CNN vision transformer robustness gaussian noise classification | arXiv | 10 | Kim et al. (2025); Sun et al. (2026); Yang et al. (2022) |
| C2 | noise level estimation selects expert classifier mixture input-dependent gating denoising | IEEE Xplore | 10 | Ninguno en imágenes; compuertas con SNR en comunicaciones y voz |
| C3 | "noise-aware" ensemble CNN vision transformer noise estimate weighting image classification | ScienceDirect, Springer, MDPI | 10 | HN-ViT (2026, detección) |
| C4 | wavelet shrinkage MAD noise estimate preprocessing CNN classifier robustness Gaussian noise gating | arXiv, CVF Open Access | 10 | WaveCNet (Li et al., 2020, 2021); MoPE (arXiv 1904.12426) |
| C5 | quality-aware fusion classifier switch denoised image noisy input noise level estimator deep network | Web (búsqueda ampliada) | 9 | MIND (arXiv 2508.07817), fusiones con confianza en eliminación de ruido |
| C6 | TrashNet waste classification gaussian noise robustness CNN transformer fusion | Web, PMC, MDPI | 10 | Ninguno con compuerta de ruido; solo aumentación con ruido o fusiones de CNN |

**Snowballing.** Hacia atrás desde MoPE y WaveCNet (sus referencias sobre preprocesamiento contra ruido) y hacia adelante desde Donoho y Johnstone (1994) en trabajos que usan el estimador MAD dentro de redes. No apareció ningún trabajo que use el estimador para ponderar clasificadores.

## 24.3 Trabajos más cercanos y diferencia

| Trabajo | Qué hace | Diferencia con NAGFuse |
|---|---|---|
| MoPE: *Mixture of pre-processing experts* (arXiv 1904.12426) | Una red de compuerta pequeña elige entre un autocodificador que elimina ruido y la identidad, antes de un detector | La compuerta es una red **entrenada** y mezcla **preprocesamientos**; NAGFuse mezcla **clasificadores** con una compuerta analítica, calibrada solo con imágenes limpias y con garantía formal |
| WaveCNet (Li et al., 2020; 2021) | Sustituye el submuestreo de las CNN por la transformada de ondícula y descarta las altas frecuencias | Cambia la arquitectura interna; no estima el ruido ni combina expertos. NAGFuse usa contracción adaptativa con umbral universal |
| Kim et al. (2025) | Meta-análisis de 1174 modelos: los ViT supervisados son más robustos al ruido gaussiano | Respalda la premisa de NAGFuse (el experto ViT es el robusto); no propone una fusión |
| HN-ViT (Springer, 2026) | Módulo de mejora consciente del ruido y doble flujo CNN–ViT para detección | Fusiona rasgos con módulos aprendidos; no hay estimador explícito ni compuerta analítica |
| MIND (arXiv 2508.07817) | Estimador de nivel de ruido aprendido que guía la atención en eliminación de ruido médica | Tarea de restauración, no de clasificación; estimador aprendido |
| Sun et al. (2026) | Teoría de por qué las mezclas de expertos son robustas al ruido en los rasgos | Marco teórico afín; no imágenes ni compuerta por nivel de ruido |

**Conclusión.** No se encontró un trabajo que pondere dos clasificadores con un estimador de ruido sin entrenamiento y garantice que la salida se reduce al experto robusto. La novedad es **de mecanismo**, una combinación nueva de piezas conocidas, no de componente.

