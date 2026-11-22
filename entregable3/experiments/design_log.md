# Bitácora de diseño de NAGFuse

Partición de desarrollo: conjunto de validación (360 imágenes) y predicciones de validación de las 8 semillas del E2. Ninguna decisión usó el conjunto de prueba ni los conjuntos adicionales. Métrica: F1 macro de validación, media ± DE sobre 8 semillas; la fusión se evaluó promediando las probabilidades de ambos modelos con la misma semilla.

| N.° | Variante evaluada | F1 val (desarrollo) | Decisión | Motivo y fecha |
|---|---|---|---|---|
| 0 | Referencias individuales: ViT-B16 0.9679 ± 0.0059; DenseNet201 (rama de RWCNet) 0.9618 ± 0.0046; RWCNet 0.9572 ± 0.0079; EfficientNet-B0 (rama de Celik-HC) 0.9377 ± 0.0115 | — | Línea de partida | 2026-11-21 |
| 1 | Fusión ViT-B16 + RWCNet | 0.9689 ± 0.0075 | Descartar como base | +0.001 sobre ViT-B16: la fusión por sí sola no aporta en limpio; RWCNet es costoso (25.6 M, 4.6 GMAC). 2026-11-21 |
| 2 | Fusión ViT-B16 + DenseNet201 | 0.9687 ± 0.0046 | Descartar como base | Igual aporte que 1, con 18 M de parámetros más que EfficientNet-B0. 2026-11-21 |
| 3 | Fusión ViT-B16 + EfficientNet-B0 | 0.9680 ± 0.0051 | **Conservar** | Mismo F1 que 1 y 2, la menor correlación de errores con ViT-B16 (0.435 frente a 0.481 y 0.539) y el menor costo (5.3 M de parámetros, 0.39 GFLOPS según torchvision). Coincide con el extractor de la hipótesis del E2 (sección 18.8). 2026-11-21 |
| 4 | Fusión ViT-B16 + GFN | 0.9660 ± 0.0098 | Descartar | Menor F1 y modelo de 113.7 M. 2026-11-21 |
| 5 | Fusión ViT-B16 + Celik-HC | 0.9690 ± 0.0042 | Descartar | Mismo F1 que 3 con dos redes convolucionales y XGBoost (componentes sin aporte en la ablación del E2). 2026-11-21 |
| 6 | Fusión ViT-B16 + ResMsCapsule | 0.9660 ± 0.0052 | Descartar | Menor F1; cápsulas mal calibradas en el E2. 2026-11-21 |
| 7 | **Versión congelada:** ViT-B/16 + EfficientNet-B0 con contracción de Haar en la rama CNN, mezcla en probabilidades con π₀ aprendido, compuerta MAD con ρ = 2 y s_lo = cuantil 0.99 de σ̂ en train, λ = 0.5 | se mide en el protocolo | Congelar | ρ, el cuantil, λ y los 2 niveles de Haar se fijaron a priori, sin ajustarlos en validación. Tag `e3-protocolo-v1`. 2026-11-21 |

**Nd = 7** variantes de desarrollo (6 fusiones evaluadas y la versión congelada). Las fusiones de desarrollo difieren entre sí en a lo sumo 0.003 de F1 de validación, por debajo de la DE entre semillas, de modo que el sesgo de selección por elegir la variante 3 es pequeño; la elección se basó en costo y complementariedad, no en el máximo F1.

**Propuesta considerada y no evaluada.** Se consideró usar VGG-16 como rama convolucional. Se descartó sin entrenarla: tiene 138 M de parámetros y 15.5 GFLOPS según torchvision (EfficientNet-B0: 0.39), su accuracy top-1 en ImageNet (71.6 %; 73.4 % con BN) es menor que la de EfficientNet-B0 (77.7 %) y DenseNet201 (76.9 %), y la evidencia de desarrollo muestra que el F1 de la fusión casi no depende de la red convolucional elegida (filas 1 a 6). Además, VGG-16 ya aparece combinada con otras CNN en la literatura de TrashNet, lo que restaría novedad.

**Lectura de la bitácora para el diseño.** La evidencia de desarrollo indica que ninguna fusión supera a ViT-B16 en F1 limpio por más de 0.002 en validación. El aporte esperado de NAGFuse es la robustez sin pérdida en limpio, y así se refleja en el hallazgo alternativo de `estandar.md`.
