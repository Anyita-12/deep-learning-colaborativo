# Procedencia del mecanismo de NAGFuse

| Campo | Contenido |
|---|---|
| Origen del mecanismo | Transferido y combinado. (1) Estimación del nivel de ruido por la mediana de los coeficientes de detalle de una ondícula y contracción suave con umbral universal, de la eliminación de ruido en señales e imágenes (Donoho y Johnstone, 1994; Donoho, 1995). (2) Mezcla de expertos con pesos que dependen de la entrada (Jacobs et al., 1991). |
| Cita completa de las fuentes | Donoho, D. L., y Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika, 81*(3), 425–455. Donoho, D. L. (1995). De-noising by soft-thresholding. *IEEE Transactions on Information Theory, 41*(3), 613–627. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., y Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation, 3*(1), 79–87. No se consultó ni copió código de terceros para estos componentes. |
| Motivo de la transferencia | Propiedad estructural compartida: en las imágenes de TrashNet (objeto sobre fondo claro y liso) la mayoría de los coeficientes diagonales de Haar de nivel 1 son casi nulos, de modo que, como en las señales suaves por tramos de Donoho y Johnstone, su mediana absoluta mide el ruido y no el contenido. El E2 cuantificó la limitación que motiva el mecanismo: con ruido σ = 0.05 las CNN pierden entre 0.30 y 0.67 de F1 macro, mientras ViT-B16 pierde 0.04. |
| Modificaciones algorítmicas | (a) En la fuente, el estimador fija el umbral de un filtro; aquí además **condiciona el peso de una mezcla de clasificadores** mediante una compuerta analítica, monótona y sin parámetros entrenables, g(σ̂) = clip((σ̂ − s_lo)/((ρ − 1) s_lo), 0, 1), calibrada solo con imágenes limpias de entrenamiento. (b) En la mezcla de expertos original la compuerta es una red entrenada; aquí el peso base π0 se aprende y el desplazamiento hacia el experto robusto lo dicta el estimador, lo que permite demostrar que con ruido suficiente la salida coincide exactamente con la del experto ViT (Teorema 2). (c) El estimador se calcula de forma conjunta sobre los tres canales y la contracción se aplica solo a la entrada del experto convolucional, el sensible al ruido, como parte de la red y de forma idéntica en entrenamiento y en inferencia. (d) La mezcla se hace en el espacio de probabilidades (no de logits) y se entrena con la verosimilitud de la mezcla más la supervisión de cada experto. |
| Clasificación de la diferencia | Mecanismo (arquitectura de inferencia con compuerta) y entrenamiento (pérdida de mezcla). No es solo novedad de aplicación. La verificación de novedad (matriz de búsqueda en al menos tres bases, cinco cadenas y revisión por un tercero) se documenta en el capítulo de literatura. |
| Implementación | `entregable3/e3/nagfuse.py`, escrito a partir de la formulación: transformada de Haar ortonormal, estimador MAD, contracción, compuerta y mezcla. De torchvision solo se importan las redes ViT-B/16 y EfficientNet-B0 con sus pesos de ImageNet, la misma fuente de pesos que usan los rivales. Pruebas en `entregable3/tests/`. |

## Formulación

Para una imagen x con valores en [0, 1]:

1. d = HH₁(x): coeficientes diagonales de la transformada de Haar ortonormal de nivel 1 en los tres canales.
2. σ̂(x) = mediana(|d|) / Φ⁻¹(3/4).
3. x_c = W⁻¹ S_τ(W x), con W la transformada de Haar de 2 niveles, S_τ la contracción suave de los detalles y τ = σ̂ √(2 ln n), n = 224².
4. p_V = softmax(f_V(x)) (ViT-B/16) y p_C = softmax(f_C(x_c)) (EfficientNet-B0).
5. π(x) = π0 + (1 − π0) g(σ̂(x)) en inferencia; π(x) = π0 = sigmoid(w) en entrenamiento.
6. p(x) = π(x) p_V + (1 − π(x)) p_C.
7. Pérdida: L = −w_y log p_y + λ (CE_w(f_V, y) + CE_w(f_C, y)), con λ = 0.5 y w los pesos de clase del E2.

**Pseudocódigo de inferencia**

```
entrada: x normalizada (B, 3, 224, 224)
x01 <- x * std + mean
sigma <- mediana(|HH1(x01)|) / 0.6745            # por imagen
x_c <- normalizar(Haar^-1(contraer(Haar(x01), sigma * sqrt(2 ln n))))
z_v <- ViT(x);  z_c <- EffNet(x_c)
pi <- pi0 + (1 - pi0) * clip((sigma - s_lo) / ((rho - 1) * s_lo), 0, 1)
log p <- logaddexp(log pi + log_softmax(z_v), log(1 - pi) + log_softmax(z_c))   # si pi = 1, log p = log_softmax(z_v)
```

**Complejidad.** El estimador y la contracción son O(n) en memoria y tiempo, salvo la mediana, que se implementa por ordenamiento (O(n log n), n = 3 × 112²). Frente a las 17.6 GMAC de ViT-B/16 y las 0.4 GMAC de EfficientNet-B0, su costo es despreciable.
