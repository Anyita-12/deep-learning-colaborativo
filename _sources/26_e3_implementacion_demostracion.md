# 26. Implementación y demostración

## 26.1 Código

El núcleo de NAGFuse está escrito a partir de la formulación del capítulo 25, sin código de terceros (`entregable3/e3/nagfuse.py`): transformada de Haar ortonormal y su inversa, estimador MAD, contracción suave, compuerta y mezcla estable en el espacio logarítmico. De torchvision solo se importan las redes ViT-B/16 y EfficientNet-B0 con sus pesos de ImageNet, la misma fuente que usan los rivales.

| Archivo | Contenido |
|---|---|
| `e3/nagfuse.py` | Modelo, variantes de ablación (`sin_denoising`, `solo_vit`, `solo_cnn`) y salidas intermedias (`details`) |
| `e3/theory.py` | Verificación numérica de los Teoremas 1 y 2 |
| `e3/worker.py`, `e3/orchestrator.py`, `e3/sync.py` | Entrenamiento reanudable en Colab con persistencia atómica en Drive |
| `tests/` | 26 casos de `pytest` (20 funciones, algunas parametrizadas) de teoría y del modelo, sin GPU ni pesos preentrenados; también corren en la integración continua |
| `analisis/` | Análisis confirmatorio, tabla maestra y figuras |

**Mezcla estable.** $\log p = \operatorname{logaddexp}(\log\pi + \log p_V,\ \log(1-\pi) + \log p_C)$. Si $\pi = 1$, se devuelve $\log p_V$ exactamente con un doble `where`, que evita el gradiente $\log 0$.

**Corrección declarada (`e3-protocolo-v2`).** En la variante sin contracción, la rama CNN recibe imágenes muy ruidosas y sus logits dejaron de ser finitos en fp16. Ahora, si una rama produce valores no finitos en inferencia, se reevalúa en fp32; si siguen sin ser finitos, se lanza un error explícito. Ninguna corrida terminada cambió (`design_log.md`).

**Complejidad.** El estimador y la contracción son $O(n)$, salvo la mediana ($O(n\log n)$ con $n = 3\cdot112^2$). Su costo es despreciable frente a las 17.6 GMAC de ViT-B/16 y las 0.4 GMAC de EfficientNet-B0. Total: 89.8 M de parámetros.

## 26.2 Propiedades demostradas

La demostración completa está en `entregable3/experiments/demostracion.md`. Aquí se resumen el enunciado y la idea.

**Lema 1.** Si el ruido es blanco gaussiano $N(0,\sigma^2)$, los coeficientes diagonales de Haar de nivel 1 son iid $N(0,\sigma^2)$. *Idea:* cada uno es una combinación de 4 píxeles con pesos $\pm 1/2$, y $4\cdot(1/2)^2 = 1$.

**Teorema 1 (estimador bajo contaminación).** Si a lo sumo una fracción $\varepsilon < 1/2$ de los coeficientes contiene contenido de la imagen, entonces con probabilidad al menos $1 - 2e^{-2n_0t^2}$

$$
\frac{G^{-1}\!\left(\frac{1/2-\varepsilon}{1-\varepsilon}-t\right)}{\Phi^{-1}(3/4)} \le \frac{\hat\sigma}{\sigma} \le \frac{G^{-1}\!\left(\frac{1}{2(1-\varepsilon)}+t\right)}{\Phi^{-1}(3/4)}, \qquad G(u) = 2\Phi(u)-1.
$$

*Idea:* la mediana solo se mueve si más de la mitad de los datos se mueve. La desigualdad de Dvoretzky–Kiefer–Wolfowitz controla la distribución empírica de los coeficientes limpios, y los contaminados solo pueden desplazar la posición de la mediana en $\varepsilon$. Es el mismo argumento de "punto de ruptura" de la estadística robusta, el mismo que hace que la mediana de varias lecturas de un sensor resista a un valor atípico.

**Teorema 2 (mezcla con compuerta).** (a) $-\log p_y \le \pi(-\log p_{V,y}) + (1-\pi)(-\log p_{C,y})$, por Jensen, porque $-\log$ es convexa. (b) $\pi(s)$ es no decreciente; vale $\pi_0$ si $s\le s_{lo}$ y 1 si $s \ge \rho s_{lo}$. (c) Si $\hat\sigma \ge \rho s_{lo}$, entonces $p = p_V$ exactamente.

**Corolario.** Bajo los supuestos del Teorema 1, si el ruido real supera $\rho s_{lo}/c_{lo}$, NAGFuse predice exactamente lo mismo que su rama ViT, con la probabilidad del Teorema 1.

## 26.3 Verificación numérica

| Propiedad | Verificación | Resultado |
|---|---|---|
| Lema 1 | Reconstrucción y energía de Haar | Error $3\times10^{-16}$ |
| Teorema 1 | Simulación con $\varepsilon \in \{0, 0.1, \dots, 0.4\}$ y contaminación adversa | Cocientes $\hat\sigma/\sigma$ dentro de las cotas |
| Teorema 2 | 100 000 casos aleatorios | Sin violaciones; $\pi = 1$ reproduce la rama ViT exactamente |
| Teorema 1 en datos reales | $\hat\sigma$ en las 361 imágenes de prueba (semilla 0) | Con $\sigma = 0.05$: mediana 0.0507, P5–P95 [0.049, 0.054]. Con $\sigma = 0.10$: mediana 0.098, P5–P95 [0.091, 0.102] |
| Teorema 2 en datos reales | $\pi$ con $\sigma = 0.05$ y 0.10, 20 semillas | $\pi = 1$ en el 100 % de las imágenes |

En limpio, $\hat\sigma$ tiene mediana 0 y el 99.2 % de las imágenes queda por debajo de $s_{lo}$. El 0 se explica por los fondos blancos saturados de TrashNet: más de la mitad de los coeficientes diagonales son exactamente cero. Esto muestra que el supuesto (A2) se cumple con holgura en este conjunto, y también que **no se cumple en dominios con fondos texturizados** (capítulo 28).

La figura muestra la separación entre las tres condiciones y los umbrales de la compuerta.

```{figure} images/e3/estimador_compuerta.png
:width: 85%
Distribución de $\hat\sigma$ en las 361 imágenes de prueba (semilla 0). Las líneas marcan $s_{lo}$ (la compuerta empieza a abrirse) y $\rho s_{lo}$ (abierta del todo).
```
