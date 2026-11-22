# Protocolo experimental preregistrado del Entregable 3

Fecha: 2026-11-21. Tag de Git: `e3-protocolo-v1`. Este documento se congela antes de las corridas definitivas. Toda desviación posterior se declara en el informe con su motivo; los análisis no previstos aquí son exploratorios.

## 1. Estándar

El de `estandar.md`.

## 2. Problema, métricas y margen (heredados del E1 y del E2)

- Clasificación de TrashNet en 6 clases con la partición v2 (`trashnet_split_v2_s1234.csv`, 1800 / 360 / 361 imágenes, agrupada por casi duplicados, huella del test verificada antes de cada corrida).
- **Métrica primaria:** F1 macro en el conjunto de prueba. **Margen de relevancia:** δ = 0.02. α = 0.05.
- Métricas secundarias: accuracy, accuracy balanceada, MCC, AUC-ROC y AUC-PR macro, ECE, NLL, Brier y F1 macro bajo las perturbaciones del E2 (ruido gaussiano σ = 0.05 y 0.10 sobre la imagen en [0, 1] antes de normalizar, desenfoque, brillo ±30 %, oclusión y fondo sustituido; semilla 2026).

## 3. Modelo original

NAGFuse (fusión ViT-B/16 + EfficientNet-B0 con compuerta adaptativa al nivel de ruido). Formulación en `procedencia.md` y propiedades en `demostracion.md`. Hiperparámetros fijados a priori: ρ = 2, s_lo = cuantil 0.99 de σ̂ en las 1800 imágenes limpias de entrenamiento, λ = 0.5 para la supervisión de cada rama, contracción de Haar de 2 niveles con umbral universal. La compuerta solo actúa en inferencia.

## 4. Hipótesis

**H1 (confirmatoria, condición R3).** Para cada rival k de la familia, E[F1_NAGFuse − F1_k] > 0 en prueba, con diferencia media mayor que δ.

**H2 (confirmatoria secundaria, familia propia).** Para cada rival k, E[F1^ruido0.05_NAGFuse − F1^ruido0.05_k] > 0, con diferencia mayor que δ.

**H3 (no inferioridad).** F1 limpio de NAGFuse no menor que el del mejor rival en más de δ (TOST, margen δ).

**Hipótesis del mecanismo.** La compuerta M (estimador MAD sin entrenamiento que transfiere el peso de la mezcla a la rama ViT) corrige la fragilidad L de las CNN ante ruido gaussiano porque P: el estimador detecta el ruido con una cota demostrable (Teorema 1) y la rama de atención global se degrada mucho menos que las CNN (E2: caída de 0.04 frente a 0.30 a 0.67). Por tanto, NAGFuse mejora el F1 macro con ruido σ = 0.05 frente a cada CNN de la familia en al menos δ, sin perder más de δ en limpio.

**Predicciones secundarias (declaradas antes de los experimentos finales).**

| N.° | Predicción | Cómo se contrasta |
|---|---|---|
| P1 | Apagar la compuerta reduce el F1 con ruido σ = 0.05 en al menos δ | NAGFuse con compuerta frente a compuerta apagada, mismas corridas |
| P2 | Apagar la compuerta no cambia el F1 limpio en más de 0.005 | Misma comparación sin perturbación |
| P3 | La ventaja de NAGFuse sobre la mejor CNN de la familia crece de σ = 0.05 a σ = 0.10 | Diferencias pareadas por semilla |
| P4 | σ̂ crece con σ en las imágenes de prueba, y con σ = 0.05 la compuerta queda abierta del todo (π = 1) en al menos el 95 % de las imágenes | `nagfuse_analisis.pkl` |
| P5 | La contracción de Haar mejora el F1 de la rama CNN con ruido | Rama CNN de NAGFuse frente a la de NAGFuse/sin_denoising |

**Criterio de éxito del diseño.** H1 se acepta si se cumplen todas las condiciones de la sección 8.4 del enunciado (unión-intersección sobre los 12 contrastes). Si no, se reporta el resultado de H1 tal cual y el trabajo se sustenta en el hallazgo alternativo de `estandar.md`.

## 5. Familia de contrastes, pruebas y decisión

- **Familia principal (m = 12):** NAGFuse frente a Celik-HC, MbXception, ViT-B16, GFN, FocusRCNet, ResMsCapsule, RWCNet, CNN, MobileNetV2, ResNet-18, HistGradientBoosting (color + HOG) y el baseline trivial. No se excluye ningún rival.
- **Tres pruebas por contraste** (Tabla 4, familia F4):
  1. t pareada sobre semillas (bilateral);
  2. Wilcoxon de rangos con signo exacto sobre semillas (con 12 o más pares el p mínimo es 2/4096 < 0.05/12);
  3. bootstrap pareado jerárquico: se remuestrean semillas y, dentro de cada semilla, grupos de casi duplicados del conjunto de prueba (10 000 réplicas), con p bilateral por inversión del intervalo.
- **Ajuste:** Holm sobre los 12 contrastes, por separado para cada prueba. **Decisión:** gobierna el máximo de los tres p ajustados.
- **Tamaño del efecto:** δ de Cliff (umbral 0.147) y d_z de Cohen, con IC 95 % bootstrap; límite inferior del IC de la diferencia mayor que cero.
- **Baselines deterministas:** HistGradientBoosting y el trivial no varían con la semilla; su F1 se trata como constante y las pruebas se aplican a F1_NAGFuse − constante.
- **Global y diagrama:** Friedman con corrección de Iman-Davenport sobre los 11 modelos profundos (bloques = semillas comunes 0 a 11) y diagrama de diferencia crítica de Nemenyi que incluye a NAGFuse.
- **Por imagen (complementarias):** McNemar por semilla con Holm y DeLong para AUC-ROC.
- **Robustez de la conclusión (8.4, condición 6):** repetir con la t corregida de Nadeau y Bengio (razón 361/1800), con la métrica secundaria MCC y con el subconjunto de semillas 0 a 9.
- **Familias secundarias, cada una con su Holm:** H2 (ruido σ = 0.05, m = 12); conjuntos adicionales (por conjunto, m = 12); ablación (4 contrastes: compuerta, contracción, rama ViT sola y rama CNN sola, frente al modelo completo).
- **Entre conjuntos:** Friedman con Iman-Davenport sobre TrashNet y los tres adicionales, y consistencia por conjunto (mejor media en al menos 2 de 3 y no significativamente peor que ningún rival en ninguno).

## 6. Semillas, potencia y regla de corte

- Unidad de réplica: la semilla. Las semillas 0 a 7 de los rivales son las corridas congeladas del E2; las nuevas usan exactamente el mismo código, hiperparámetros elegidos y partición.
- **Nivel 1:** R = 12 (semillas 0 a 11) para NAGFuse y los 10 modelos del Top 10.
- **Nivel 2:** R = 20 (semillas 12 a 19) para NAGFuse, RWCNet, ViT-B16, GFN, Celik-HC y ResMsCapsule.
- **Ablación:** semillas 0 a 5 obligatorias y 6 a 11 si alcanza el cómputo.
- **Regla de corte por tiempo (fijada aquí, independiente de los resultados):** el análisis confirmatorio usa todas las corridas terminadas al **lunes 23 de noviembre de 2026, 06:00 (Bogotá)**. Cada contraste usa las semillas comunes a ambos modelos en orden (0, 1, 2, ...), sin selección. El número final de semillas por modelo, la diferencia mínima detectable y la potencia conjunta se reportan.
- **Análisis de potencia a priori** (t no central, nivel 0.05/12, δ = 0.02, σ_d del E2): potencia por contraste con R = 12 de 0.92 si σ_d = 0.013, 0.56 si σ_d = 0.0187 (mediana de los cinco mejores) y 0.35 si σ_d = 0.023; con R = 20, 0.999, 0.92 y 0.73. La potencia conjunta con 12 contrastes de potencia 0.90 sería de 0.28. Un resultado no concluyente no se interpreta como equivalencia.
- **No se miran métricas de prueba antes de la hora de corte.** El estado del entrenamiento solo muestra tareas terminadas y tiempos.

## 7. Equidad de la comparación

- Mismo presupuesto de ajuste: 8 configuraciones (factor de tasa {1; 0.3} × weight decay {1e-4; 5e-4} × dropout {0.2; 0.4}), semilla 100, 30 épocas, selección por F1 macro de validación. Tasa base de NAGFuse 1e-4, la del régimen de ajuste fino de Transformers del E2.
- Mismo entrenamiento final: AdamW, calentamiento de 3 épocas y coseno, lote 32, máximo 60 épocas, paciencia 12, precisión mixta, entropía cruzada ponderada por clase. Mismos aumentos de datos, misma entrada 224 × 224, mismos pesos preentrenados de torchvision (ImageNet-1k).
- La contracción de Haar es parte de la arquitectura (se aplica igual en entrenamiento, validación y prueba), no un aumento de datos. Ningún modelo se entrena con ruido.
- Capacidad: NAGFuse tiene unos 90 M de parámetros; la familia ya incluye modelos de tamaño comparable o mayor (ViT-B16, 85.8 M; GFN, 113.7 M).
- Cómputo: se reportan horas de GPU, tipo de GPU, latencia, memoria y energía estimada por modelo.

## 8. Particiones de desarrollo y de confirmación

- **Desarrollo:** el conjunto de validación (360 imágenes) y las predicciones de validación del E2. Ahí se tomaron todas las decisiones de diseño (`design_log.md`, Nd = 7).
- **Confirmación:** el conjunto de prueba (361 imágenes, una evaluación por corrida) y los tres conjuntos adicionales. Ninguno se usó para diseñar.

## 9. Conjuntos adicionales (elegidos antes de ver cualquier resultado en ellos)

Se evalúan **sin reentrenar**: cada corrida entrenada en TrashNet predice sobre cada conjunto. El presupuesto de ajuste por conjunto es cero e igual para todos los modelos, y la evaluación mide la generalización bajo cambio de distribución. Las clases se mapean a las de TrashNet con reglas fijas; las imágenes casi duplicadas de cualquier imagen de TrashNet (dHash de 64 bits, distancia de Hamming ≤ 6) se retiran; se toman como máximo 500 imágenes por clase con semilla 2026. El F1 macro se calcula sobre las clases presentes en cada conjunto.

| Conjunto | Fuente y licencia | Difiere de TrashNet en | Mapeo |
|---|---|---|---|
| RealWaste | UCI ML Repository 908 (Single et al., 2023), CC BY 4.0 | dominio (vertedero real), fondo, resolución, desbalance | Cardboard, Glass, Metal, Paper, Plastic y Miscellaneous Trash a sus clases; se excluyen Food Organics, Textile Trash y Vegetation |
| Garbage Dataset v2 | Kaggle `sumn2u/garbage-classification-v2` (Kunwar, 2026) | fuente (celular, web, comunidad), fondos variados, tamaño | metal, glass, paper, trash, cardboard y plastic; se excluyen biological, battery, shoes y clothes |
| WaRP-C | Kaggle `parohod/warp-waste-recycling-plant-dataset` (Yudin et al., 2024), MIT | planta de reciclaje (banda transportadora), recortes pequeños, deformación, 4 clases | glass-* a glass; cans a metal; *-cardboard y detergent-box a cardboard; bottle-*, detergent-* y canister a plastic; solo `test_crops` |

Se registran la fuente, la fecha de descarga, el hash del archivo y los conteos (`<conjunto>.json`).

## 10. Determinismo, exclusiones y registro

- `cudnn.deterministic = True`, `benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Orden de lotes y aumentación determinados por (semilla, época).
- Toda corrida terminada se incluye. Una corrida que falla por un error de infraestructura se repite con la misma semilla desde su último checkpoint; nunca se reemplaza por otra semilla.
- Las semillas nuevas de los rivales se ejecutan en GPU de Colab (L4 o A100); las métricas de desempeño son comparables entre GPU y los tiempos solo dentro de la misma GPU.
- Desviación prevista: si los pesos del profesor EfficientNet-B4 de FocusRCNet (no versionados en el E2) no están disponibles, se reentrena con la misma configuración y semilla 100, y se compara su F1 de validación con el del E2.
