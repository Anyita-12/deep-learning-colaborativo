# 30. Conclusiones

## 30.1 Respuestas

- **P1 (H1). No.** NAGFuse supera a 9 de los 12 rivales en F1 limpio por más de $\delta$, pero empata con RWCNet y ViT-B16. La potencia frente a ellos (0.99 y 0.83) indica que el empate es real, no falta de semillas. La condición de superioridad sobre toda la familia no se cumple.
- **P2 (H2). Frente a las CNN, sí; frente a ViT-B16, no.** Con $\sigma = 0.05$ supera a todas las CNN por 0.12 a 0.76; frente a ViT-B16 la ventaja (+0.012) no es significativa.
- **P3 (H3). Sí.** Es equivalente al mejor rival, RWCNet, dentro de $\pm\delta$ (TOST $p < 0.001$), y tiene la mejor calibración del benchmark, empatada con RWCNet.
- **P4 (mecanismo). En parte.**
  - El estimador detecta el ruido como predice el Teorema 1 y la compuerta conmuta como garantiza el Teorema 2 ($\pi = 1$ en el 100 % de las imágenes con ruido).
  - La contracción de Haar protege a la rama CNN.
  - Con $\sigma = 0.05$ la compuerta casi no cambia el F1 (P1 falla); su aporte aparece con $\sigma = 0.10$.
- **P5 (generalización). No.** En los tres conjuntos adicionales NAGFuse es peor que ViT-B16, aunque mejor que la mayoría de las CNN.

## 30.2 Qué aporta

1. **Un modelo que reúne lo mejor de dos familias.** Iguala al mejor modelo en limpio (RWCNet), conserva la robustez al ruido de ViT-B16 y está mejor calibrado que ambos. Ningún rival tiene las tres propiedades a la vez.
2. **Un mecanismo con garantía.** La compuerta no tiene parámetros entrenables y su comportamiento está demostrado. Se puede auditar: el estimador $\hat\sigma$ y el peso $\pi$ salen con cada predicción.
3. **Una comparación con inferencia.** Hasta 20 semillas por modelo, tres pruebas por contraste, bootstrap jerárquico por casi duplicados y tres conjuntos externos. Ninguno de los 59 estudios revisados en el E2 reúne estas condiciones.
4. **Un resultado negativo útil.** La ganancia con ruido viene sobre todo de la rama ViT entrenada en conjunto; la conmutación solo importa con ruido fuerte.

## 30.3 Límites

- **Ruido sintético.** Solo ruido gaussiano aditivo, independiente por píxel. El ruido real de cámara depende de la señal y está correlacionado entre canales.
- **El estimador depende del fondo.** En TrashNet, $\hat\sigma = 0$ en limpio porque el fondo blanco está saturado. Con fondos texturizados el supuesto (A2) falla y la compuerta puede reaccionar a la textura (capítulo 28).
- **Costo.** 89.8 M de parámetros y la inferencia de dos redes. Es el modelo más pesado después de GFN.
- **Hardware mezclado.** Las semillas 0–7 de los rivales del E2 se entrenaron en otra GPU. Con RWCNet se observó una diferencia de −0.014 entre las dos máquinas (capítulo 21).
- **Conjuntos adicionales.** Para los rivales, solo semillas 8 a 19 (u 8 a 11); con 4 semillas las pruebas no pueden ser significativas.
- **Bloque B incompleto.** No se midió la latencia en CPU con mediana y p95 para todos los modelos (solo NAGFuse en el dashboard), ni la memoria en inferencia, y no se probó cuantización, poda ni destilación con TOST. Los costos reportados son parámetros, GFLOPs y latencia en GPU.
- **Sin curva de aprendizaje frente al tamaño de muestra** (10–50 % del entrenamiento), por presupuesto de cómputo.
- **Interpretabilidad sin prueba de fidelidad.** El dashboard usa oclusión (no Grad-CAM, porque el ONNX en CPU no da gradientes) y la explicación global se apoya en el propio mecanismo ($\pi$ y F1 de cada rama), sin pruebas de fidelidad ni de estabilidad.
- **Verificaciones por terceros pendientes.** La repetición de la búsqueda de novedad y la reproducción independiente del análisis no las hizo una persona ajena al código antes de la entrega (capítulos 24 y 31).

## 30.4 Trabajo futuro

1. **Una compuerta robusta a la textura.** Estimar $\sigma$ solo en bloques lisos, o con la mediana de varias subbandas, y verificar (A2) por imagen.
2. **Medir $\pi$ y $\hat\sigma$ en los conjuntos adicionales**, para separar el efecto de la rama CNN del de la compuerta.
3. **Ruido realista.** Poisson-gaussiano y compresión JPEG, en el marco de ImageNet-C (Hendrycks y Dietterich, 2019).
4. **Destilar NAGFuse** en una sola red ligera con la compuerta como señal de entrenamiento.

## 30.5 Opción B

El segundo intento (protocolo v4, capítulo 32) cambia solo las redes de las ramas: Swin-S como rama robusta y EfficientNet-B0 como rama convolucional. Se decide con $\alpha_B = 0.025$ porque el conjunto de prueba ya se había usado con A.

- **P1 (H1-B). No.** NAGFuse-B supera a 10 de los 12 rivales, uno más que A (GFN). Frente a ViT-B16 y RWCNet la diferencia es positiva (+0.0115 y +0.0109) y su IC excluye el cero, pero es menor que $\delta$. La potencia para detectar $\delta$ era de 0.99, de modo que el resultado no se debe a falta de semillas.
- **P2 (H2-B). Frente a las CNN, sí; frente a ViT-B16, no** (+0.012, no significativo), igual que A.
- **P3 (H3-B). Sí.** La diferencia con RWCNet está dentro de $\pm\delta$ (TOST $p = 0.006$).
- **P4 (mecanismo).** La compuerta se abre del todo con ruido en todas las imágenes, como en A. Apagarla reduce el F1 con $\sigma = 0.05$ en 0.006, más que en A pero menos que $\delta$.
- **P5 (generalización). En parte.** B tiene la mejor media en RealWaste y Garbage v2, y en WaRP-C queda por debajo de ViT-B16 sin diferencia significativa: cumple el criterio de consistencia entre conjuntos, que A no cumplía.

Frente a A, B gana +0.011 de F1 en limpio con 41 % menos de parámetros y la mitad de GFLOPs, y queda por debajo con ruido $\sigma = 0.10$ (0.827 frente a 0.859), porque con ruido la salida es la de la rama Swin-S, que se degrada más que ViT-B/16. Estas comparaciones con A son exploratorias y mezclan hardware: las semillas 3 a 19 de B se entrenaron en T4 y todas las de A en A100.
