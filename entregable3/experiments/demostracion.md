# Propiedades formales de NAGFuse

Notación: Φ es la función de distribución normal estándar y G(u) = P(|Z| ≤ u) = 2Φ(u) − 1 para Z ~ N(0, 1). G es continua y estrictamente creciente en [0, ∞), con inversa G⁻¹(q) = Φ⁻¹((1 + q)/2). Note que G⁻¹(1/2) = Φ⁻¹(3/4) ≈ 0.6745.

## Lema 1 (Haar ortonormal y ruido blanco)

Si e ∈ ℝ^{H×W} tiene entradas independientes N(0, σ²), los coeficientes diagonales de nivel 1, d_{ij} = (e_{2i,2j} − e_{2i,2j+1} − e_{2i+1,2j} + e_{2i+1,2j+1})/2, son independientes e idénticamente distribuidos N(0, σ²).

*Demostración.* Cada d_{ij} es una combinación lineal de cuatro variables gaussianas independientes con coeficientes ±1/2, luego es gaussiana con media 0 y varianza 4 · (1/4) σ² = σ². Los bloques 2 × 2 de índices (i, j) distintos son disjuntos, así que los d_{ij} dependen de conjuntos disjuntos de variables independientes y son independientes. ∎

## Teorema 1 (estimador MAD bajo contaminación)

**Supuestos.** (A1) La imagen observada es y = x + e, con e como en el Lema 1 en cada canal e independiente entre canales (sin recorte a [0, 1]). (A2) De los n coeficientes diagonales de nivel 1 de la imagen limpia x en los tres canales, un subconjunto fijo 𝒩 con n₀ = |𝒩| ≥ (1 − ε) n, ε < 1/2, es exactamente cero; los demás n₁ = n − n₀ coeficientes observados pueden tomar cualquier valor ("contaminados" por el contenido de la imagen).

Sea m̂ la mediana inferior de |d₁|, …, |d_n| y σ̂ = m̂ / Φ⁻¹(3/4). Para todo t > 0, con probabilidad al menos 1 − 2 exp(−2 n₀ t²),

$$
\frac{G^{-1}\!\left(\frac{1/2-\varepsilon}{1-\varepsilon}-t\right)}{\Phi^{-1}(3/4)} \;\le\; \frac{\hat\sigma}{\sigma} \;\le\; \frac{G^{-1}\!\left(\frac{1}{2(1-\varepsilon)}+t\right)}{\Phi^{-1}(3/4)},
$$

siempre que los argumentos de G⁻¹ estén en [0, 1).

*Demostración.* Por (A2) y el Lema 1, los |d_i| con i ∈ 𝒩 son iid con función de distribución G(·/σ). Sea Ĝ₀ su función de distribución empírica. Por la desigualdad de Dvoretzky–Kiefer–Wolfowitz (con la constante de Massart), el evento E = {sup_u |Ĝ₀(u) − G(u/σ)| ≤ t} tiene probabilidad al menos 1 − 2 exp(−2 n₀ t²). Trabajamos en E. Sea k = ⌈n/2⌉ la posición de la mediana inferior y N(u) = #{i ∈ 𝒩 : |d_i| ≤ u}.

*Cota inferior.* Si m̂ ≤ a, al menos k de todos los coeficientes cumplen |d_i| ≤ a; como a lo sumo n₁ = n − n₀ son contaminados, N(a) ≥ k − n + n₀, y con k ≥ n/2,

$$\hat G_0(a) \ge \frac{n/2 - n + n_0}{n_0} = 1 - \frac{n}{2 n_0} \ge 1 - \frac{1}{2(1-\varepsilon)} = \frac{1/2-\varepsilon}{1-\varepsilon},$$

donde la última desigualdad usa n₀ ≥ (1 − ε) n. En E, G(a/σ) ≥ Ĝ₀(a) − t ≥ (1/2 − ε)/(1 − ε) − t. Con a = m̂ y aplicando G⁻¹, que es creciente, m̂/σ ≥ G⁻¹((1/2 − ε)/(1 − ε) − t).

*Cota superior.* Para todo b < m̂, a lo sumo k − 1 coeficientes cumplen |d_i| ≤ b, luego N(b) ≤ k − 1 ≤ n/2 y Ĝ₀(b) ≤ n/(2 n₀) ≤ 1/(2(1 − ε)). En E, G(b/σ) ≤ 1/(2(1 − ε)) + t para todo b < m̂; por continuidad de G, también en b = m̂. Aplicando G⁻¹ se obtiene la cota superior. Dividir por Φ⁻¹(3/4) da el enunciado. ∎

**Corolario 1.** Con ε = 0 y n → ∞, σ̂ → σ en probabilidad (las dos cotas tienden a 1 cuando t → 0). Con n = 3 · 112² = 37 632 coeficientes y probabilidad 0.99 (t elegido con n₀), σ̂/σ está entre 0.98 y 1.02 si ε = 0, entre 0.85 y 1.16 si ε = 0.1, y entre 0.52 y 1.62 si ε = 0.3. La implementación de la verificación (`e3/theory.py`) usa una versión algo más conservadora, con t/(1 − ε) y t calculado con n, que también es válida porque solo ensancha el intervalo.

## Teorema 2 (mezcla con compuerta)

Sean p_V, p_C distribuciones de probabilidad sobre las K clases, π ∈ [0, 1] y p = π p_V + (1 − π) p_C. Sea g(s) = clip((s − s_lo)/((ρ − 1) s_lo), 0, 1) con s_lo > 0, ρ > 1, y π(s) = π₀ + (1 − π₀) g(s) con π₀ ∈ (0, 1).

(a) Para toda clase y: −log p_y ≤ π(−log p_{V,y}) + (1 − π)(−log p_{C,y}) ≤ máx(−log p_{V,y}, −log p_{C,y}).

(b) π(s) es no decreciente en s; π(s) = π₀ si s ≤ s_lo y π(s) = 1 si s ≥ ρ s_lo.

(c) Si σ̂ ≥ ρ s_lo, entonces p = p_V: la predicción de NAGFuse coincide exactamente con la de la rama ViT.

*Demostración.* (a) −log es convexa en (0, ∞); por la desigualdad de Jensen con pesos π y 1 − π aplicada a p_{V,y} y p_{C,y}, −log(π p_{V,y} + (1 − π) p_{C,y}) ≤ π(−log p_{V,y}) + (1 − π)(−log p_{C,y}). Una combinación convexa no supera su máximo. (b) s ↦ (s − s_lo)/((ρ − 1) s_lo) es creciente porque (ρ − 1) s_lo > 0, y clip(·, 0, 1) es no decreciente; vale 0 para s ≤ s_lo y 1 para s ≥ ρ s_lo. Como 1 − π₀ > 0, π hereda la monotonía. (c) Por (b), π = π₀ + (1 − π₀) = 1, luego p = p_V. ∎

**Corolario 2 (garantía conjunta).** Bajo (A1)–(A2), si el ruido real cumple σ ≥ ρ s_lo / c_lo(ε, t), con c_lo(ε, t) la cota inferior del Teorema 1, entonces con probabilidad al menos 1 − 2 exp(−2 n₀ t²) NAGFuse predice exactamente lo mismo que su rama ViT. En consecuencia, bajo ese nivel de ruido su F1 coincide con el de la rama ViT, que el E2 mostró poco sensible al ruido.

## Supuestos que los datos reales no cumplen exactamente

- **Coeficientes limpios no nulos (A2).** El contenido del objeto (bordes y texturas) produce coeficientes diagonales distintos de cero. En TrashNet el fondo liso ocupa la mayor parte de la imagen, por lo que se espera ε pequeño; la verificación numérica estima ε en imágenes de validación.
- **Recorte a [0, 1] (A1).** Las perturbaciones del E2 recortan la imagen ruidosa a [0, 1], y el fondo de TrashNet es cercano al blanco. El recorte reduce la varianza del ruido en esos bloques y hace que σ̂ subestime σ. Como el recorte depende del propio ruido, esos bloques no se pueden tratar como contaminación fija; su efecto se mide en la verificación numérica.
- **Cuantización JPEG y redimensionado.** Introducen pequeños coeficientes de alta frecuencia en las imágenes limpias; su efecto se absorbe en s_lo, calculado con imágenes limpias reales.

En los casos en que algún supuesto falla, los teoremas son una motivación y no una garantía, y así se presentan en el informe.

## Verificación numérica

1. **Lema 1 y Teorema 1 sintéticos** (`e3/theory.py: verify_haar, verify_mad_bounds`; pruebas en `tests/test_theory.py`): reconstrucción exacta (error 3e-16), conservación de la energía y, para ε ∈ {0, 0.1, 0.2, 0.3, 0.4} con contaminación en los dos peores sentidos, los cocientes σ̂/σ observados quedan dentro de las cotas y pegados a la cota poblacional.
2. **Teorema 2** (`verify_mixture_bound`): 100 000 casos aleatorios sin violaciones; prueba de que π = 1 reproduce exactamente la rama ViT en el modelo (`tests/test_nagfuse.py`).
3. **Datos reales** (`nagfuse_analisis.pkl` de cada corrida): σ̂ y π para cada perturbación del conjunto de prueba, que contrastan la predicción P4 y muestran la consecuencia observable de la propiedad (robustez al ruido).
