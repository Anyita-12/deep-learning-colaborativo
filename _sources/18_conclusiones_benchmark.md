# 18. Conclusiones del benchmark

En este capítulo respondemos las preguntas de investigación del capítulo 8 con los resultados de los capítulos 9 a 15, recomendamos un modelo, fijamos la referencia para la etapa siguiente y delimitamos el alcance del estudio. Todas las cifras corresponden a 8 semillas por modelo, evaluadas en las 361 imágenes del conjunto de prueba de la partición v2.

## 18.1 Respuesta a las preguntas de investigación

| RQ | Respuesta | Evidencia |
|---|---|---|
| RQ1. ¿Difieren los diez modelos? | **Sí.** Rechazamos $H_0$ | Friedman $Q = 64.8$, $p < 10^{-5}$, $W = 0.90$; Iman-Davenport $F(9, 63) = 62.7$; el modelo explica el 90 % de la varianza ($\eta^2 = 0.90$) |
| RQ2. ¿Supera algún modelo publicado al modelo de referencia por al menos 0.036? | **Sí, cuatro.** RWCNet, ViT-B16, GFN y Celik-HC, todos preentrenados | t pareada con Holm sobre 45 pares: $p \leq 0.021$; $d_z$ entre 1.9 y 5.3; IC bootstrap que excluyen el umbral; Wilcoxon y Conover en el mismo sentido |
| RQ3. ¿Se reproduce la accuracy publicada del paper guía? | **No.** Ninguna de las 10 corridas cumple el criterio fijado a priori; el pipeline sí queda validado | Protocolo original: $0.925 \pm 0.005$ frente a 0.994 publicado; seleccionar la época con el conjunto de prueba infla la accuracy en 2.8 puntos, IC [1.4, 4.2]. Con el mismo pipeline, RWCNet y GFN reproducen su valor publicado (sección 12.2) |
| RQ4. ¿Aportan la *hypercolumn*, la segunda red y el preentrenamiento del paper guía? | **Solo el preentrenamiento** (exploratoria) | Sin preentrenamiento: −0.085 de F1, $p$ Holm < 0.001. Sin *hypercolumn*, con capas intermedias y con solo EfficientNet-B0: equivalentes al modelo completo dentro de $\pm\delta$ (TOST $p \leq 0.03$) |
| RQ5. ¿Son más robustos al ruido los modelos con atención global? | **Sí** (exploratoria) | Caída de F1 con $\sigma = 0.05$: ViT-B16 0.04 [0.03, 0.07] y GFN 0.12 [0.10, 0.15], frente a 0.30–0.67 en las arquitecturas convolucionales |

La comparación de cada modelo con el modelo de referencia del capítulo 7 (ResNet-18 desde cero, F1 de 0.863) se resume así:

| Modelo | Diferencia en F1 macro [IC 95 % bootstrap] | $p$ Holm (45 pares) | $d_z$ | Decisión |
|---|---|---|---|---|
| RWCNet | +0.082 [0.072, 0.092] | < 0.001 | 5.25 | Superior, significativa y relevante |
| ViT-B16 | +0.075 [0.057, 0.094] | 0.004 | 2.67 | Superior, significativa y relevante |
| GFN | +0.058 [0.039, 0.079] | 0.021 | 1.87 | Superior, significativa y relevante |
| Celik-HC | +0.051 [0.034, 0.068] | 0.015 | 1.98 | Superior, significativa y relevante |
| ResMsCapsule | +0.036 [0.018, 0.055] | 0.13 | 1.22 | No concluyente |
| MbXception | −0.002 [−0.017, 0.014] | 1.00 | −0.06 | No concluyente |
| FocusRCNet | −0.052 [−0.068, −0.037] | 0.010 | −2.17 | Inferior, significativa y relevante |

ResMsCapsule supera el umbral en la media y es significativa en la subfamilia de 7 contrastes con el modelo de referencia ($p$ Holm = 0.022), pero no con la familia completa de 45 pares, que es la regla preregistrada. Entre los modelos superiores, dos pares son equivalentes dentro de $\pm\delta$: RWCNet y ViT-B16 (TOST $p = 0.024$) y GFN y Celik-HC (TOST $p = 0.014$).

## 18.2 Modelo recomendado y referencia para la etapa siguiente

El modelo recomendado es **RWCNet**, con F1 macro de $0.945 \pm 0.010$ (IC 95 % [0.938, 0.951]) y accuracy de 0.952. Es el primero en F1, accuracy, MCC, AUC-ROC y AUC-PR, el mejor calibrado (ECE de 0.027) y el que mejor generaliza, con la menor diferencia entre validación y prueba. Sus mapas de Grad-CAM son los más estables entre semillas ($\rho = 0.81$) y su fidelidad se confirma con la prueba de inserción. ViT-B16 rinde igual en F1 y es la alternativa cuando la prioridad es la robustez al ruido.

Para la etapa siguiente, el modelo propio se compara con el paper guía, **Celik-HC**, reproducido con nuestro protocolo: F1 macro de $0.914 \pm 0.013$ (IC 95 % [0.905, 0.923]). Con 8 semillas, la diferencia mínima detectable en los pares con Celik-HC está entre 0.013 y 0.023, cercana a $\delta = 0.02$, por lo que el diseño conserva la resolución necesaria para contrastar la hipótesis de la sección 18.8. RWCNet fija el desempeño más alto alcanzado con el mismo protocolo.

## 18.3 Desempeño por clase y errores

`cardboard` es la clase con mayor F1 en los diez modelos, y `trash` la de menor F1 en todos. La confusión dominante en siete modelos es plástico clasificado como vidrio, dos materiales que comparten transparencia y brillo. Los demás patrones son propios de cada arquitectura: vidrio clasificado como metal en ResMsCapsule, plástico clasificado como papel en FocusRCNet y papel clasificado como `trash` en la CNN.

El preentrenamiento reduce los tres patrones de error de forma pareja. La precisión en `trash` pasa de 0.50–0.80 en los modelos entrenados desde cero a 0.78–0.91 en los preentrenados, y el recall en plástico de 0.69–0.79 a 0.86–0.91. RWCNet comete 17 errores por semilla en promedio, frente a 46.5 de ResNet-18 y 62 de MobileNetV2.

De las 23 imágenes que fallan al menos la mitad de las 80 corridas, 9 son objetos transparentes, 7 son de material mixto, 5 tienen un encuadre que oculta el material, 1 es un error de etiqueta probable (`glass186`) y 1 queda sin categoría. El techo estimado de F1 macro está entre 0.98 y 1.00, y el mejor modelo deja sin cubrir entre el 4 % y el 6 % del margen sobre el baseline trivial.

## 18.4 Generalización, calibración y robustez

**Generalización.** RWCNet tiene una brecha entre entrenamiento y validación de 0.037, significativamente menor que $\delta_g = 0.05$, y la menor diferencia entre validación y prueba (0.012, con un IC que incluye el cero). ViT-B16 y GFN muestran un F1 de validación mayor que el de prueba (0.030 y 0.033), en parte por una diferencia leve de distribución entre ambos conjuntos (AUC adversarial de 0.56). Todos los modelos superan con amplio margen al baseline trivial (F1 de 0.064) y al baseline clásico con color y HOG (0.760).

**Calibración.** Las redes sin recalibrar son sobreconfiadas, con pendiente de calibración entre 0.29 y 0.80 en nueve modelos. El escalado de temperatura ajustado en validación lleva el ECE a 0.02–0.04. ResMsCapsule es subconfiado y también se corrige con ese ajuste. Con predicción conforme ($\alpha = 0.10$), los modelos preentrenados producen conjuntos de 0.89 a 0.95 clases en promedio, con una cobertura de 0.87 a 0.89, ligeramente por debajo de la nominal por la misma diferencia entre validación y prueba. Los modelos entrenados desde cero alcanzan la cobertura nominal (0.89 a 0.92) con conjuntos mayores.

**Robustez.** Con ruido gaussiano de $\sigma = 0.05$, los modelos se ordenan por arquitectura y no por su F1 sin perturbar:

- **Transformer puro (ViT-B16):** pierde 0.04 de F1 (índice de robustez de 0.95).
- **Híbrido con codificador Transformer (GFN):** pierde 0.12 (0.83).
- **CNN preentrenadas:** RWCNet pierde 0.30, ResMsCapsule 0.48 y Celik-HC 0.66.
- **CNN desde cero:** pierden entre 0.47 y 0.67.

Ante el fondo sustituido, los modelos preentrenados pierden entre 0.02 y 0.08 de F1 y los entrenados desde cero entre 0.18 y 0.26. El preentrenamiento reduce la dependencia del fondo blanco que el capítulo 2 identificó como atajo. La robustez al ruido sigue siendo la principal limitación de las CNN, incluido el paper guía.

## 18.5 Costo computacional y huella energética

| Modelo | Parámetros | GMAC por imagen | Inferencia (ms/imagen, lote 32) | Entrenamiento (min por corrida) | GPU |
|---|---:|---:|---:|---:|---|
| RWCNet | 25.6 M | 4.59 | 2.28 | 12.3 | RTX 4060 |
| ViT-B16 | 85.8 M | 17.56 | 0.46 | 3.5 | A100 |
| GFN | 113.7 M | 5.67 | 0.97 | 5.0 | A100 |
| Celik-HC | 28.4 M | 3.22 | 1.12 | 6.2 | A100 |
| ResMsCapsule | 4.2 M | 1.42 | 0.40 | 5.3 | RTX 4060 |
| ResNet-18 | 11.2 M | 1.81 | 0.33 | 8.3 | RTX 4060 |
| MbXception | 28.4 M | 6.19 | 1.42 | 10.2 | RTX 4060 |
| CNN | 1.2 M | 2.59 | 0.68 | 8.4 | RTX 4060 |
| FocusRCNet | 0.5 M | 0.13 | 0.37 | 9.2 | RTX 4060 |
| MobileNetV2 | 2.2 M | 0.30 | 0.37 | 8.5 | RTX 4060 |

RWCNet domina a ViT-B16 y a GFN en la frontera de Pareto de parámetros y operaciones: obtiene igual o mayor F1 con menos de un tercio de sus parámetros. Por F1 ganado por cada GMAC adicional sobre el modelo de referencia, Celik-HC (0.036) y RWCNet (0.030) son los más eficientes. Para un dispositivo de borde, ResMsCapsule ofrece el mejor equilibrio: 0.899 de F1 con 4.2 M de parámetros y 1.4 GMAC. Los tiempos se comparan entre modelos de una misma GPU.

El benchmark completo, incluidas la búsqueda de hiperparámetros, las corridas finales y las ablaciones, consumió 26.9 horas de GPU y unos 2.4 kWh, equivalentes a 0.78 kg de CO₂e. La potencia se midió en la RTX 4060, y la cota superior con la potencia nominal es de 3.6 kWh (sección 13.12).

## 18.6 Relación con la literatura y con el paper guía

Con preentrenamiento, la brecha de 0.089 con la mediana publicada en TrashNet que señalamos en la sección 7.6 casi desaparece: la accuracy de RWCNet (0.952) queda a 0.008 de esa mediana. La diferencia restante es coherente con un protocolo más estricto, con partición agrupada, varias semillas y conjunto de prueba usado una sola vez.

El mismo pipeline reproduce la accuracy publicada de RWCNet (95.2 % frente a 95.01 %) y de GFN (a 0.7 puntos), y las brechas de los demás modelos son heterogéneas y tienen causas documentadas (sección 12.2). El pipeline queda así validado, y las discrepancias se atribuyen a cada artículo y no a la implementación. Celik-HC, el paper guía, queda 6.8 puntos por debajo: con su protocolo original obtenemos 0.925, y la selección de la época con el conjunto de prueba explica 2.8 de esos puntos. Su ablación muestra que la *hypercolumn*, la segunda red y XGBoost no aportan sobre un único extractor preentrenado; el preentrenamiento sí aporta, con 0.085 de F1. En RWCNet, la ablación de tres componentes muestra el mismo patrón: aportan el extractor principal (DenseNet201, 0.033 de F1) y la supervisión profunda (0.022), mientras que la segunda red (MobileNetV2) no tiene un aporte significativo.

La contribución de este estudio es una comparación de diez modelos de cuatro familias amplias de arquitectura con el mismo protocolo, réplicas, inferencia estadística con control de comparaciones múltiples y análisis de calibración, robustez e interpretabilidad. Ningún estudio de la revisión sistemática del capítulo 10 reúne estas condiciones sobre TrashNet.

## 18.7 Alcance del estudio

Los resultados se refieren a las siguientes condiciones del estudio:

1. **Datos:** un solo conjunto, de imágenes de estudio con un objeto sobre fondo blanco, y un conjunto de prueba de 361 imágenes con 19 de `trash`. Las perturbaciones sintéticas aproximan el cambio de distribución hacia una planta de reciclaje.
2. **Robustez:** el ruido y el brillo se evalúan con tres niveles de severidad; el desenfoque, la oclusión y el fondo, con el nivel preregistrado que permite comparar con los modelos base.
3. **Interpretabilidad:** la estabilidad entre semillas y la fracción de atención sobre el fondo cubren los siete modelos publicados, y los modelos base conservan su análisis de Grad-CAM del capítulo 5; la fidelidad por inserción se midió en los cuatro modelos con pesos guardados, entre ellos el mejor (RWCNet).
4. **Ablaciones:** ocho variantes con 8 semillas cada una: tres componentes del mejor modelo (RWCNet) y cinco del paper guía, incluido el preentrenamiento. Ambos modelos comparten la estructura de dos CNN preentrenadas fusionadas.
5. **Tamaño de entrenamiento:** todos los modelos se entrenan con las 1800 imágenes de la partición v2. El efecto de más datos sobre los modelos entrenados desde cero se infiere de su convergencia (sección 14.2), no de curvas de aprendizaje por tamaño.
6. **Ruido de etiqueta:** se estima con el consenso de las 80 corridas y la revisión visual de los 23 casos difíciles (sección 15.4).
7. **Margen de sobreajuste:** $\delta_g = 0.05$ en F1 macro, valor de referencia para este margen; el análisis es exploratorio.
8. **Energía:** la potencia se midió en la RTX 4060; en la A100 se estima con la misma fracción de la potencia nominal, con la potencia nominal como cota superior (sección 13.12).
9. **Revisión:** foco en estudios evaluados en TrashNet y lectura a texto completo de los registros de prioridad Alta (sección 10.10).

## 18.8 Brecha, hipótesis y trabajo futuro

**La brecha de la revisión y lo que aporta este benchmark.** La revisión sistemática (sección 10.9) identificó esta brecha:

> Ninguno de los 59 estudios incluidos compara modelos publicados de al menos cuatro familias sobre TrashNet con una partición que controle los casi-duplicados, varias semillas e inferencia estadística, ni compara con inferencia estadística la robustez de esos modelos ante ruido de imagen.

| Parte de la brecha | Qué aporta el benchmark | Preguntas |
|---|---|---|
| Comparación controlada de modelos publicados | Diez modelos de cuatro familias amplias con la misma partición agrupada, 8 semillas, prueba de Friedman y 45 contrastes pareados con Holm. Cuatro modelos superan al de referencia y dos pares son equivalentes dentro de $\pm\delta$ | RQ1, RQ2 |
| La misma comparación aplicada al paper guía | Réplica con criterio a priori y ablación de cinco componentes: la accuracy publicada no se reproduce y solo el preentrenamiento aporta | RQ3, RQ4 |
| Robustez ante ruido con inferencia estadística | Caída de F1 con ruido gaussiano $\sigma = 0.05$ y $0.10$, con IC bootstrap por grupo en los siete modelos publicados y la misma perturbación en los tres modelos base: ViT-B16 pierde 0.04 con $\sigma = 0.05$, GFN 0.12, y las CNN, incluido el paper guía, entre 0.30 y 0.67 | RQ5 |

El benchmark cubre así las dos partes de la brecha. La segunda deja además un problema abierto y cuantificado: ningún modelo del benchmark reúne a la vez un F1 del grupo superior, robustez al ruido y bajo costo. ViT-B16 es robusto, pero es el modelo con más operaciones (17.6 GMAC). Las CNN preentrenadas más eficientes, Celik-HC y RWCNet, pierden 0.66 y 0.30 de F1 con ruido moderado, y la fusión multinivel del paper guía no corrige esa fragilidad. Ese problema es el punto de partida de la etapa siguiente.

**Hipótesis contrastable.** Un modelo con un único extractor convolucional preentrenado (EfficientNet-B0, que en la ablación rinde igual que Celik-HC completo), al que se incorpora un mecanismo de robustez de la sección 10.8, entrenado con el protocolo de este benchmark, cumplirá las dos condiciones siguientes:

- **Robustez:** reducirá la caída de F1 macro con ruido gaussiano $\sigma = 0.05$ en al menos 0.20 frente a Celik-HC, cuya caída es de 0.66. Los mecanismos candidatos son el entrenamiento con ruido y corrupciones (M2), la destilación desde ViT-B16 como maestro (M1) y la atención global (M4).
- **No inferioridad:** conservará su F1 limpio dentro de $\delta = 0.02$ del de Celik-HC.

Con $\Delta$ = caída de Celik-HC − caída del modelo propio, la primera condición se contrasta como:

- $H_0$: $\Delta \leq 0.20$.
- $H_1$: $\Delta > 0.20$.

La prueba es la t pareada unilateral por semilla, con IC bootstrap por grupo de casi-duplicados. La segunda condición se contrasta con TOST de no inferioridad sobre el F1 limpio, con margen $\delta$. Con 8 semillas, la diferencia mínima detectable en F1 frente a Celik-HC (0.013 a 0.023) es del orden de $\delta$, y el margen de 0.20 en la caída queda muy por encima de su incertidumbre (IC 95 % de la caída de Celik-HC: [0.63, 0.68]).

**Otras líneas.**

- Repetir el benchmark con una partición agrupada por objeto, construida con *embeddings* (sección 9.4).
- Retirar el preentrenamiento en RWCNet y ViT-B16 para separar arquitectura y transferencia.
- Ampliar la malla en tasa de aprendizaje de FocusRCNet, que elige el extremo de la malla y es sensible a ese valor, y el presupuesto de épocas de los modelos entrenados desde cero.
- Extender la prueba de fidelidad por inserción a ViT-B16, GFN y Celik-HC, y usar borrado por regiones contiguas, que no se confunde con la sensibilidad al ruido.
- Auditar las etiquetas empezando por las 23 imágenes difíciles.
