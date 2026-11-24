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

## 2026-11-22: corrección `e3-protocolo-v2` (declarada)
- **Hallazgo:** la variante de ablación `NAGFuse/sin_denoising` falló en `_nag_analysis` con `ValueError: Input contains NaN` (probabilidades no finitas) al evaluar con la compuerta apagada o con otro ρ: la rama CNN sin contracción recibe imágenes muy ruidosas y sus logits dejan de ser finitos en fp16. En la evaluación normal el defecto quedaba oculto porque con π = 1 la mezcla devuelve exactamente la rama ViT (`torch.where`). La variante completa no lo sufre porque su rama CNN ve imágenes contraídas.
- **Corrección (no cambia ningún resultado de entrenamiento ni de evaluación ya producido):**
  1. `finite_logits`: si los logits de una rama no son finitos en inferencia, se reevalúan en fp32; si siguen sin serlo, se produce un error explícito. Cuando son finitos no se altera nada.
  2. El análisis exploratorio de NAGFuse (`_nag_analysis`) ya no hace fallar una corrida entrenada: registra el aviso y el error (`nagfuse_analisis_ERROR.json`) y continúa. Se comprueba además que las probabilidades sean finitas, para localizar el caso.
- **Alcance:** solo afecta a las corridas de ablación `sin_denoising` (semillas 3 a 5 y siguientes). Las corridas ya terminadas no se tocan.

## Opción B: segundo intento autorizado (protocolo v4, `protocolo_v4_opcion_B.md`)
- **Motivo:** la versión A no cumplió R3 (empate con RWCNet y ViT-B16 en F1 limpio). El docente autorizó un segundo intento con declaración completa. Los resultados de A en prueba ya eran conocidos; ver la sección 2 del protocolo v4.
- **Antes del piloto (tag `e3-protocolo-v4-piloto`):** candidatos fijados a priori, sin usar prueba ni conjuntos adicionales: B1 = Swin-S + ConvNeXt-T, B2 = Swin-S + EfficientNet-B0, B3 = ViT-B/16 + ConvNeXt-T. Mecanismo, pérdida, hiperparámetros (intento 1 de la búsqueda de A) y protocolo de entrenamiento idénticos a A.

| N.° | Variante evaluada (piloto, semilla 100, solo validación) | F1 val | Decisión | Fecha |
|---|---|---|---|---|
| 8 | NAGFuse-B1 (Swin-S + ConvNeXt-T) | _completar_ | _completar_ | _completar_ |
| 9 | NAGFuse-B2 (Swin-S + EfficientNet-B0) | _completar_ | _completar_ | _completar_ |
| 10 | NAGFuse-B3 (ViT-B/16 + ConvNeXt-T) | _completar_ | _completar_ | _completar_ |

Referencia: versión A en el mismo intento de la búsqueda, F1 val 0.9688. **Nd = 10.** La tabla se completa con la salida de `python -m e3.orchestrator piloto` antes del tag `e3-protocolo-v4`.
