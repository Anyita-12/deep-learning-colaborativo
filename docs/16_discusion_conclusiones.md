# 16. Discusión y conclusiones del benchmark

En este capítulo respondemos las preguntas de investigación del capítulo 8 con los resultados de los capítulos 9 a 15, discutimos qué explica las diferencias entre modelos y señalamos las amenazas a la validez y las limitaciones. Todas las cifras corresponden a 8 semillas por modelo, evaluadas en las 361 imágenes del conjunto de prueba de la partición v2.

## 16.1 Respuesta a las preguntas de investigación

| RQ | Respuesta | Evidencia |
|---|---|---|
| RQ1. ¿Difieren los diez modelos? | **Sí.** Rechazamos $H_0$ | Friedman $Q = 64.8$, $p < 10^{-5}$, $W = 0.90$; Iman-Davenport $F(9, 63) = 62.7$; el modelo explica el 90 % de la varianza ($\eta^2 = 0.90$) |
| RQ2. ¿Supera algún modelo publicado al modelo de referencia por al menos 0.036? | **Sí, cuatro.** RWCNet (+0.082), ViT-B16 (+0.075), GFN (+0.058) y Celik-HC (+0.051), todos preentrenados | t pareada con Holm sobre 45 pares: $p \leq 0.021$; $d_z$ entre 1.9 y 5.3; IC bootstrap que excluyen el umbral; Wilcoxon y Conover en el mismo sentido |
| RQ3. ¿Se reproduce la accuracy publicada del paper guía? | **No.** Ninguna de las 10 corridas cumple el criterio fijado a priori | Protocolo original: $0.925 \pm 0.005$ frente a 0.994 publicado; la selección de la época con el conjunto de prueba infla la accuracy en 2.8 puntos, IC [1.4, 4.2] |
| RQ4. ¿Aportan la *hypercolumn*, la segunda red y el preentrenamiento del paper guía? | **Solo el preentrenamiento** (exploratoria) | Sin preentrenamiento: −0.085 de F1, $p$ Holm < 0.001. Sin *hypercolumn*, con capas intermedias y con solo EfficientNet-B0: equivalentes al modelo completo dentro de $\pm\delta$ (TOST $p \leq 0.03$) |
| RQ5. ¿Son más robustos al ruido los modelos con atención global? | **Sí** (exploratoria) | Caída de F1 con $\sigma = 0.05$: ViT-B16 0.04 [0.03, 0.07] y GFN 0.12 [0.10, 0.15], frente a 0.30–0.67 en las arquitecturas convolucionales |

## 16.2 Qué modelo funciona mejor y por qué

**RWCNet y ViT-B16 son los mejores modelos, y entre ellos no hay un ganador.** RWCNet obtiene el mayor F1 macro ($0.945 \pm 0.010$) y ViT-B16 el segundo ($0.938 \pm 0.014$). Su diferencia, 0.007, es equivalente dentro de $\pm\delta$ (TOST $p = 0.024$; probabilidad bayesiana de la ROPE de 1.00), y McNemar no detecta diferencias en ninguna semilla. Dos arquitecturas muy distintas alcanzan el mismo desempeño: una fusión de dos CNN con supervisión profunda y un Transformer puro. Su único rasgo común es el preentrenamiento en ImageNet.

**La evidencia de ablación apunta al preentrenamiento como causa principal de la mejora.**

- **Los cuatro modelos que superan al de referencia son preentrenados.** Los dos modelos publicados entrenados desde cero no lo superan: MbXception queda a −0.002 y FocusRCNet a −0.052.
- **En el paper guía:** retirar el preentrenamiento cuesta 0.085 de F1, más que la diferencia entre el mejor y el peor modelo preentrenado.
- **En Celik-HC, los componentes añadidos no aportan:** la *hypercolumn*, la segunda red y XGBoost. Las tres primeras variantes son equivalentes al modelo completo dentro de $\pm\delta$.
- **En RWCNet, la supervisión profunda sí aporta:** sin las salidas auxiliares, el F1 baja 0.022 (IC [0.014, 0.030]), apenas por encima de $\delta$ (sección 15.5). Esa contribución explica parte de su ventaja sobre GFN y Celik-HC, pero es una cuarta parte de la pérdida que produce retirar el preentrenamiento en el paper guía.

Una explicación coherente es que, con unas 2500 imágenes, la representación aprendida en ImageNet aporta rasgos de material (brillo, transparencia, textura) que una arquitectura no aprende desde cero, por ingeniosa que sea. Sobre esa base, solo ayudan los componentes que regularizan el entrenamiento, como la supervisión de las capas intermedias, y no los que añaden capacidad o etapas de clasificación. El benchmark no aísla del todo el preentrenamiento del resto de la arquitectura, salvo en la ablación de Celik-HC. Retirar el preentrenamiento en RWCNet y ViT-B16 queda como experimento pendiente.

**La interpretabilidad respalda a RWCNet y obliga a cautela con ViT-B16.** Los mapas de Grad-CAM de RWCNet se concentran en el objeto y son estables entre semillas ($\rho = 0.81$). Los de ViT-B16 también evitan el fondo, pero cambian mucho entre semillas ($\rho = 0.16$), por lo que no sirven como explicación por imagen.

## 16.3 Generalización y calibración

**RWCNet es el modelo que mejor generaliza y mejor se calibra:**

- **Brecha entre entrenamiento y validación:** 0.037, significativamente menor que $\delta_g = 0.05$.
- **Diferencia entre validación y prueba:** la menor del benchmark (0.012, con un IC que incluye el cero).
- **Calibración:** el menor ECE (0.027 sin recalibrar y 0.020 con escalado de temperatura) y la menor NLL.

Es el único modelo competitivo que cumple a la vez las cuatro condiciones del criterio de adecuación de la sección 7.2 del enunciado. FocusRCNet también las cumple, pero con un F1 de 0.811.

ViT-B16 y GFN tienen un F1 de validación significativamente mayor que el de prueba (0.030 y 0.033), y su predicción conforme queda por debajo de la cobertura nominal. Parte de esta brecha se debe a una diferencia leve de distribución entre validación y prueba (AUC adversarial de 0.56). Todos los modelos comparten esa validación, por lo que la comparación no se sesga.

Las redes sin recalibrar son sobreconfiadas: la pendiente de calibración está entre 0.29 y 0.80 en nueve modelos. El escalado de temperatura ajustado en validación lleva el ECE a 0.02–0.04 y debe formar parte de cualquier despliegue. ResMsCapsule es la excepción: es subconfiado, porque las longitudes de las cápsulas no son probabilidades.

## 16.4 Robustez y cambio de distribución

ViT-B16 es el modelo más robusto. Conserva el 95 % de su F1 con ruido gaussiano de $\sigma = 0.05$ y el 89 % con $\sigma = 0.10$, y es el que menos pierde con desenfoque (0.07) y con el fondo sustituido (0.02). Con ruido, los modelos se ordenan por su arquitectura y no por su F1 sin perturbar:

- **Transformer puro (ViT-B16):** el índice de robustez es 0.95.
- **Híbrido con codificador Transformer (GFN):** 0.83.
- **CNN preentrenadas (RWCNet, Celik-HC, ResMsCapsule):** entre 0.42 y 0.69.
- **CNN desde cero:** entre 0.36 y 0.51.

Ante un fondo distinto, que emula condiciones de planta, los modelos preentrenados pierden entre 0.02 y 0.08 de F1 y los entrenados desde cero entre 0.18 y 0.26. El preentrenamiento reduce la dependencia del fondo blanco que el capítulo 2 identificó como atajo.

Las perturbaciones son sintéticas y aplicadas a imágenes de estudio. Estos resultados indican qué arquitecturas son más frágiles, pero no estiman el desempeño en una planta de reciclaje.

## 16.5 Significancia estadística y relevancia práctica

De los 45 pares, 32 son significativos y relevantes con la prueba preregistrada, y 37 con el bootstrap o con Conover. Las decisiones que cambian entre pruebas corresponden a diferencias cercanas a $\delta$ (0.017 a 0.024), que reportamos como no concluyentes.

Ninguna diferencia significativa del benchmark es menor que $\delta$: con 8 semillas, solo alcanzan significancia diferencias de al menos 0.02, por lo que no hay casos de significancia sin relevancia práctica. Las comparaciones confirman equivalencia práctica en dos pares: RWCNet y ViT-B16, y GFN y Celik-HC. La ausencia de significancia en los demás pares no demuestra equivalencia. Para los pares con más varianza, la diferencia mínima detectable con 8 semillas llega a 0.036, y detectar $\delta$ con potencia de 0.80 requeriría hasta 27 semillas.

## 16.6 Desempeño frente a costo

RWCNet domina a ViT-B16 y a GFN en la frontera de Pareto: obtiene igual o mayor F1 con menos parámetros (25.6 M frente a 85.8 M y 113.7 M) y menos operaciones (4.6 GMAC frente a 17.6 y 5.7). Por F1 ganado por cada GMAC adicional sobre el modelo de referencia, Celik-HC (0.036) y RWCNet (0.030) son los más eficientes, y ViT-B16 el menos (0.005).

Para un dispositivo de borde, ResMsCapsule (4.2 M parámetros, 1.4 GMAC, F1 de 0.899) y FocusRCNet (0.5 M parámetros, 0.13 GMAC, F1 de 0.811) son los modelos no dominados más livianos. Cuando la robustez al ruido es prioritaria, el costo adicional de ViT-B16 se justifica por esa robustez y no por su F1.

## 16.7 Techo de desempeño y margen de mejora

TrashNet no tiene una segunda anotación. Estimamos el techo de F1 macro entre 0.98 y 1.00 a partir de los 23 casos que fallan al menos la mitad de las 80 corridas. Solo uno es un error de etiqueta probable; 16 son ambigüedades de material, por transparencia, reflejo o material mixto.

El mejor modelo deja sin cubrir entre el 4 % y el 6 % del margen sobre el baseline trivial, y entre el 29 % y el 39 % del margen sobre el modelo de referencia. Los 3 a 5 puntos de F1 que quedan se concentran en materiales que una fotografía RGB no distingue con seguridad. Mejorar más el F1 limpio exigiría otra información, y no solo otra arquitectura. En cambio, en robustez queda un margen grande: la mayoría de los modelos pierde más de la mitad de su F1 con ruido moderado.

## 16.8 Relación con la literatura y con el capítulo 7

El modelo de referencia del capítulo 7 (ResNet-18 desde cero, F1 de 0.863) queda superado por cuatro modelos, y el mejor lo mejora en 0.082. La accuracy del mejor modelo (0.952) está a 0.008 de la mediana publicada en TrashNet (~96 %, capítulo 3). Con preentrenamiento, la brecha de 0.089 con la literatura señalada en la sección 7.6 casi desaparece, y lo que queda se explica por el protocolo más estricto.

La comparación de cada modelo con su valor publicado muestra un sesgo de reporte heterogéneo:

| Modelo | Accuracy publicada | Accuracy obtenida | Diferencia |
|---|---|---|---|
| Celik-HC | 99.40 % | 92.6 % | −6.8 |
| MbXception | 94.34 % | 87.4 % | −6.9 |
| ViT-B16 | 96.98 % | 94.4 % | −2.6 |
| GFN | 94.21 % | 93.5 % | −0.7 |
| FocusRCNet | ~92 % | 83.2 % | −8.8 |
| ResMsCapsule | 91.41 % (4 clases) | 91.1 % | no comparable |
| RWCNet | 95.01 % | 95.2 % | +0.2 |

RWCNet y GFN reproducen su valor publicado. RWCNet es, además, el estudio con el protocolo más cuidadoso: validación cruzada e IC.

Las mayores discrepancias tienen dos orígenes:

- **Celik-HC:** selección de la época con el conjunto de prueba, cuantificada en el capítulo 12.
- **MbXception y FocusRCNet:** entrenados aquí con una entrada menor que la de su artículo y con un tope de épocas que alcanzan. Esta explicación es plausible, pero el experimento no la aísla.

## 16.9 Amenazas a la validez

**Validez interna.**

- **Hardware distinto:** las corridas se repartieron entre una A100 y una RTX 4060, con el mismo código, semillas y configuración. La reproducción exacta de ViT-B16 en otra sesión muestra determinismo dentro de una misma máquina, pero no entre máquinas. La ablación de Celik-HC se ejecutó en una GPU distinta de la del modelo completo.
- **Dependencia por objeto:** la auditoría con *embeddings* encontró vistas distintas de un mismo objeto en entrenamiento y prueba que la partición no agrupó. Excluir las 95 imágenes de prueba sospechosas baja el F1 entre 0.01 y 0.03 y no cambia el orden de los modelos (sección 9.4).
- **Hiperparámetros en el borde de la malla:** seis de los siete modelos eligen el *weight decay* máximo de la malla.
- **Subajuste:** MbXception y FocusRCNet alcanzan el tope de 60 épocas en la mayoría de sus semillas.
- **Protocolo de los modelos base:** CNN, MobileNetV2 y ResNet-18 siguen el protocolo del capítulo 4 (hasta 100 épocas), más generoso en épocas que el de los modelos publicados.

**Validez externa.** Un solo conjunto de datos, de imágenes de estudio con un objeto sobre fondo blanco, y un único conjunto de prueba de 361 imágenes. Los resultados describen este conjunto y no las condiciones de una planta de reciclaje.

**Validez de constructo.**

- **F1 macro con 19 imágenes de `trash`:** la métrica es sensible a pocos errores en esa clase.
- **Perturbaciones sintéticas:** la robustez y el fondo sustituido son aproximaciones del cambio de distribución real.
- **Medidas indirectas de interpretabilidad:** la fracción de energía de Grad-CAM sobre el fondo depende del umbral que define el fondo, y no verificamos la fidelidad de los mapas.

**Validez estadística.**

- **Potencia:** 8 semillas no detectan diferencias del tamaño de $\delta$ en los pares con más varianza.
- **Prueba de decisión:** Wilcoxon con Holm no puede rechazar con 45 pares y 8 semillas, por lo que fijamos la t pareada; la decisión se tomó antes de ver los resultados y está documentada.
- **Análisis exploratorios:** son numerosos (unas 25 tablas de análisis, 13 métricas por modelo, 8 perturbaciones y 2 ablaciones), por lo que sus hallazgos se presentan como generadores de hipótesis.
- **Independencia del conjunto de prueba:** las pruebas por imagen (McNemar, DeLong) tratan las imágenes como independientes dentro de cada grupo de casi-duplicados, un supuesto que la dependencia por objeto debilita.

## 16.10 Limitaciones

1. **Siete modelos publicados en lugar de diez:** el benchmark tiene diez modelos profundos contando los tres modelos base (sección 10.5).
2. **Robustez:** el ruido y el brillo tienen tres niveles de severidad con la imagen original; el desenfoque, la oclusión y el fondo tienen uno.
3. **Interpretabilidad:** no se verificó la fidelidad de los mapas de Grad-CAM con pruebas de borrado o inserción.
4. **Ablación del mejor modelo:** un componente de RWCNet; los otros dos se evaluaron en Celik-HC, que tiene la misma estructura.
5. **Sin curvas de aprendizaje por tamaño de entrenamiento:** no sabemos si más datos mejorarían a los modelos entrenados desde cero.
6. **Ruido de etiqueta:** se estimó con la revisión visual de 23 casos difíciles y no con una segunda anotación.
7. **Margen de sobreajuste $\delta_g$:** no se preregistró.
8. **Consumo energético:** se estimó con la potencia nominal de la GPU, no se midió.
9. **Revisión:** búsqueda original restringida a «TrashNet», 78 candidatos fuera del corte de prioridad y búsqueda ampliada sin Scopus (sección 10.10).

## 16.11 Trabajo futuro e hipótesis para la etapa siguiente

**Brecha.** Ningún estudio revisado compara modelos publicados de al menos cuatro familias sobre TrashNet con una partición agrupada, varias semillas e inferencia estadística, ni evalúa si su ventaja se mantiene ante ruido de imagen. Este benchmark cubre la primera parte. La segunda deja una limitación cuantificada: las arquitecturas convolucionales, incluido el paper guía, pierden entre 0.30 y 0.67 de F1 con ruido gaussiano moderado, y la fusión multinivel no corrige esa fragilidad.

**Hipótesis contrastable.** Un modelo basado en un único extractor convolucional preentrenado (EfficientNet-B0, que en la ablación rinde igual que Celik-HC completo) cumplirá, entrenado con el protocolo de este benchmark, una de estas dos condiciones:

- incorpora un mecanismo de robustez de la tabla de mecanismos candidatos (sección 10.8): entrenamiento con ruido y corrupciones (M2), destilación desde un maestro Transformer robusto como ViT-B16 (M1) o atención global (M4);
- conserva su F1 limpio dentro de $\delta = 0.02$ del de Celik-HC.

Además, reducirá la caída de F1 con ruido gaussiano $\sigma = 0.05$ en al menos 0.20 frente a Celik-HC.

- $H_0$: la caída de F1 del modelo propio es igual a la de Celik-HC.
- $H_1$: es al menos 0.20 menor.

La prueba sería la t pareada por semilla con IC bootstrap por grupo, junto con TOST de no inferioridad en F1 limpio.

**Otras líneas.**

- Repetir el benchmark con una partición agrupada por objeto, construida con *embeddings* (sección 9.4).
- Retirar el preentrenamiento en RWCNet y ViT-B16 para separar arquitectura y transferencia.
- Ampliar la malla en *weight decay* y el presupuesto de épocas de los modelos entrenados desde cero.
- Verificar la fidelidad de las explicaciones con pruebas de borrado.
- Auditar las etiquetas empezando por las 23 imágenes difíciles.
