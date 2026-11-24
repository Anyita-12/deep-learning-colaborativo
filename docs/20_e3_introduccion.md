# 20. Entregable 3: introducción

Los capítulos 20 a 31 forman el Entregable 3. Siguen uno a uno la estructura pedida (capítulos 0 a 11 del enunciado):

| Enunciado | Capítulo del libro | Contenido |
|---|---|---|
| 0. Introducción | 20 | Objetivo, preguntas y resumen de resultados |
| 1. Correcciones | 21 | Trazabilidad de correcciones del E2, auditoría de fuga, congelamiento |
| 2. Datos | 22 | Partición de TrashNet y los tres conjuntos adicionales |
| 3. Benchmark | 23 | Configuraciones, semillas nuevas y artefactos exportados |
| 4. Literatura y novedad | 24 | Matriz de búsqueda y trabajos más cercanos |
| 5. Diseño | 25 | NAGFuse: limitación, mecanismo e hipótesis |
| 6. Implementación | 26 | Código, pruebas y demostración formal |
| 7. Protocolo y validación | 27 | Preregistro, potencia y pruebas de hipótesis |
| 8. Evaluación ampliada | 28 | Robustez, conjuntos adicionales, ablación, calibración |
| 9. Dashboard | 29 | Aplicación desplegada |
| 10. Conclusiones | 30 | Respuestas, límites y trabajo futuro |
| 11. Reproducibilidad | 31 | Entorno, artefactos, autoría y uso de herramientas |

## 20.1 Objetivo

El E2 dejó una limitación medida con inferencia: con ruido gaussiano de $\sigma = 0.05$ las CNN de la familia pierden entre 0.30 y 0.67 de F1 macro, mientras ViT-B16 pierde 0.04. El E3 propone un modelo original, **NAGFuse** (*Noise-Adaptive Gated Fusion*), que combina ViT-B/16 y EfficientNet-B0 con una compuerta guiada por un estimador del nivel de ruido, y lo compara con la familia completa del E2 bajo un protocolo preregistrado.

## 20.2 Preguntas

- **P1 (H1).** ¿NAGFuse supera en F1 macro limpio a cada uno de los 12 rivales (10 modelos y 2 baselines) por más de $\delta = 0.02$?
- **P2 (H2).** ¿Los supera con ruido $\sigma = 0.05$?
- **P3 (H3).** ¿Es no inferior al mejor rival en limpio?
- **P4 (mecanismo).** ¿Funciona por las razones previstas? Se contrasta con cinco predicciones secundarias y una ablación.
- **P5 (generalización).** ¿Se mantiene la ventaja en tres conjuntos de otra procedencia, sin reentrenar?

## 20.3 Resumen de resultados

Se entrenaron 144 corridas nuevas (19.1 h de GPU A100) y se analizaron junto con las 88 del E2: 20 semillas para NAGFuse y los cinco rivales más cercanos, 12 para el resto.

- **H1 no se cumple en su forma fuerte.** NAGFuse (F1 $0.9365 \pm 0.011$) supera a 10 de 12 rivales, pero empata con RWCNet ($-0.0004$) y ViT-B16 ($+0.0002$). La condición de superioridad sobre toda la familia (R3) no se alcanza.
- **H3 se cumple.** NAGFuse es equivalente al mejor rival, RWCNet, dentro de $\pm\delta$ (TOST $p < 0.001$).
- **H2 se cumple frente a todas las CNN** (ventajas de 0.12 a 0.76), pero no frente a ViT-B16 ($+0.012$, no significativo).
- **El mecanismo se verifica en parte.** El estimador detecta el ruido como predice el Teorema 1 y la contracción de Haar protege a la rama convolucional. Con $\sigma = 0.05$, sin embargo, la compuerta casi no aporta (P1 falla); su efecto aparece con $\sigma = 0.10$.
- **En los conjuntos adicionales NAGFuse es peor que ViT-B16** (entre $-0.06$ y $-0.07$), aunque supera a las CNN.

El trabajo se sustenta por tanto en el hallazgo alternativo declarado antes de los experimentos (`estandar.md`): robustez frente a las CNN sin pérdida en limpio, no inferioridad frente al mejor rival y una verificación del mecanismo que identifica qué parte funciona y cuál no.
