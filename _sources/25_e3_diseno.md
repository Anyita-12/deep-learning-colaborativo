# 25. Diseño de NAGFuse

## 25.1 Limitación, mecanismo y predicción

- **Limitación (E2).** Con ruido gaussiano $\sigma = 0.05$, las CNN pierden entre 0.30 y 0.67 de F1 macro; ViT-B16 pierde 0.04 (capítulo 15).
- **Mecanismo.** Un estimador sin entrenamiento mide el ruido de cada imagen y desplaza el peso de una mezcla de dos expertos hacia el robusto (ViT) cuando el ruido supera lo que se ve en imágenes limpias. El experto convolucional recibe además la imagen con contracción de ondícula.
- **Predicción.** NAGFuse conserva el F1 limpio (no pierde más de $\delta$) y mejora el F1 con ruido frente a cada CNN en al menos $\delta$. Las predicciones secundarias P1 a P5 están en el capítulo 27.

## 25.2 Formulación

Para una imagen $x \in [0,1]^{3\times H\times W}$:

$$
\hat\sigma(x) = \frac{\operatorname{mediana}\left(|HH_1(x)|\right)}{\Phi^{-1}(3/4)}, \qquad
x_c = W^{-1} S_\tau(Wx),\quad \tau = \hat\sigma\sqrt{2\ln n}
$$

$$
\pi(x) = \pi_0 + (1-\pi_0)\, g(\hat\sigma), \qquad g(s) = \operatorname{clip}\!\left(\frac{s - s_{lo}}{(\rho-1)s_{lo}}, 0, 1\right)
$$

$$
p(y\mid x) = \pi(x)\, p_V(y\mid x) + \bigl(1-\pi(x)\bigr)\, p_C(y\mid x_c)
$$

| Símbolo | Forma | Papel |
|---|---|---|
| $HH_1(x)$ | $(3, H/2, W/2)$ | Coeficientes diagonales de Haar de nivel 1; en zonas lisas casi solo contienen ruido |
| $\Phi^{-1}(3/4) \approx 0.6745$ | escalar | Convierte la mediana de $\lvert Z\rvert$ en la desviación de $Z$ |
| $W$, $S_\tau$ | — | Transformada de Haar de 2 niveles y contracción suave de los detalles |
| $p_V$, $p_C$ | $(6,)$ | Probabilidades de ViT-B/16 y de EfficientNet-B0 |
| $\pi_0 = \sigma(w)$ | escalar | Peso base aprendido (0.509 en la semilla 0) |
| $s_{lo}$ | escalar | Cuantil 0.99 de $\hat\sigma$ en las 1800 imágenes limpias de entrenamiento (0.01454) |
| $\rho = 2$ | escalar | Por encima de $\rho s_{lo}$ la compuerta se abre del todo ($\pi = 1$) |

La compuerta solo actúa en inferencia; en entrenamiento $\pi = \pi_0$. La pérdida es $L = -w_y\log p_y + \lambda\,(\mathrm{CE}_w(f_V, y) + \mathrm{CE}_w(f_C, y))$ con $\lambda = 0.5$ y los pesos de clase del E2.

**Intuición.** Es un control por programación de ganancia (*gain scheduling*): una variable medida, el nivel de ruido, elige entre dos controladores ya sintonizados, en lugar de que una red aprenda la conmutación.

## 25.3 Decisiones de diseño (bitácora)

Las decisiones se tomaron en validación con las predicciones de las 8 semillas del E2 (`design_log.md`, $N_d = 7$). Las seis fusiones de ViT-B16 con otro modelo difieren en a lo sumo 0.003 de F1 de validación. Se eligió EfficientNet-B0 por su menor costo (5.3 M de parámetros, 0.39 GFLOPS) y su menor correlación de errores con ViT-B16 (0.435). $\rho$, el cuantil de $s_{lo}$, $\lambda$ y los dos niveles de Haar se fijaron a priori, sin ajustarlos.

## 25.4 Procedencia

El estimador y la contracción vienen de la eliminación de ruido por ondículas (Donoho y Johnstone, 1994; Donoho, 1995) y la mezcla, de las mezclas de expertos (Jacobs et al., 1991). Las modificaciones están en `procedencia.md`: compuerta analítica en lugar de red entrenada, estimador conjunto sobre tres canales, contracción solo para el experto sensible y mezcla en el espacio de probabilidades.
