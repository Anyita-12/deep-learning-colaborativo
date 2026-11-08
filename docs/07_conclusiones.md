# 7. Conclusiones

En este capítulo respondemos la pregunta de investigación de la sección 1.3 con los resultados de los capítulos 4 y 5, definimos el modelo de referencia y señalamos las limitaciones del estudio. Todas las cifras corresponden a 8 semillas por modelo, evaluadas en las 361 imágenes del conjunto de prueba.

## 7.1 Respuesta a la pregunta de investigación

Las tres arquitecturas, entrenadas desde cero con el mismo protocolo, no tienen el mismo desempeño. La prueba de Friedman rechaza la hipótesis nula global ($Q = 9.75$, $p$ exacto $= 0.0048$, $W$ de Kendall $= 0.61$), y la arquitectura explica el 52 % de la varianza del F1 macro entre corridas ($\eta^2 = 0.52$).

| Comparación | Diferencia media en F1 macro [IC 95 %] | $p$ Holm (t / Wilcoxon) | $d_z$ | ¿Significativa y $\geq \delta$? |
|---|---|---|---|---|
| ResNet-18 − CNN | 0.037 [0.018, 0.055] | 0.005 / 0.031 | 1.64 | Sí |
| ResNet-18 − MobileNetV2 | 0.053 [0.029, 0.077] | 0.003 / 0.023 | 1.87 | Sí |
| CNN − MobileNetV2 | 0.017 [−0.014, 0.047] | 0.24 / 0.31 | 0.45 | No |

ResNet-18 supera a los otros dos modelos con diferencias significativas y mayores que el margen de relevancia práctica $\delta = 0.02$. La ventaja se mantiene en todas las métricas secundarias (accuracy 0.871, balanced accuracy 0.861, MCC 0.842, AUC-ROC 0.979) y ResNet-18 es también el modelo mejor calibrado (ECE 0.051). Entre la CNN y MobileNetV2 no encontramos una diferencia significativa. Como la diferencia mínima detectable del experimento para ese par es 0.036 (sección 7.2), este resultado no demuestra que ambos modelos sean equivalentes.

Una explicación coherente con estos resultados es el sesgo inductivo de ResNet-18: los atajos identidad facilitan la propagación del gradiente y permiten entrenar desde cero una red más profunda. El experimento no aísla este mecanismo.

## 7.2 Modelo de referencia

El modelo de referencia es **ResNet-18**, con F1 macro de $0.863 \pm 0.022$ (IC 95 % [0.845, 0.881]) y accuracy de 0.871. Supera a la CNN en 4.4 % y a MobileNetV2 en 6.6 % de F1 macro relativo.

Un modelo posterior debe superar este valor en al menos **0.036 de F1 macro**, con significancia y bajo el mismo protocolo. El umbral es el máximo entre $\delta = 0.02$ y la diferencia mínima detectable con 8 semillas (0.036). El experimento tiene menos resolución que la prevista: el piloto estimó desviaciones de las diferencias pareadas de 0.016 a 0.019, y en el experimento completo fueron de 0.022 a 0.037. Con esa variabilidad, 8 semillas no permiten detectar con potencia de 0.80 diferencias del tamaño de $\delta$.

## 7.3 Desempeño por clase y errores

`cardboard` es la clase mejor clasificada por los tres modelos (F1 entre 0.94 y 0.96) y `trash` la peor (0.61 en la CNN, 0.65 en MobileNetV2 y 0.79 en ResNet-18). En `trash`, ResNet-18 obtiene la mayor ventaja sobre los otros modelos. Las confusiones más frecuentes, sumadas en las 24 corridas, son vidrio y plástico (259), vidrio y metal (190), papel y plástico (180) y papel y `trash` (162). Son los pares de materiales visualmente parecidos que anticipamos en la sección 1.6.

Hay 19 imágenes de prueba que la mayoría de las semillas de los tres modelos clasifica mal, con predicciones coincidentes entre arquitecturas (por ejemplo, `glass278` y `plastic455` como `paper`). Estos casos pueden ser objetos ambiguos o errores de etiqueta. Sin la auditoría de la sección 2.8 no podemos distinguir entre las dos explicaciones.

## 7.4 Generalización y robustez

ResNet-18 tiene la mayor brecha entre entrenamiento y validación en la mejor época (0.065, frente a 0.039 de MobileNetV2 y 0.011 de la CNN) y, al mismo tiempo, el mejor desempeño en validación y prueba. Su mayor capacidad le permite ajustar mejor el entrenamiento sin que ese ajuste perjudique el desempeño en imágenes de la misma distribución.

Ante perturbaciones de la imagen de prueba observamos tres comportamientos:

- **Ruido y desenfoque:** los tres modelos pierden la mayor parte de su desempeño. Con ruido gaussiano de $\sigma = 0.05$, el F1 macro cae a 0.19 (CNN), 0.28 (MobileNetV2) y 0.39 (ResNet-18). Estas perturbaciones no se incluyeron en el aumento de datos.
- **Brillo y oclusión:** las caídas son moderadas (entre 0.04 y 0.11), coherentes con el aumento fotométrico y el recorte aleatorio usados en el entrenamiento.
- **Fondo sustituido:** el F1 macro cae entre 0.21 y 0.26. La prueba definida antes del entrenamiento, que compara esta caída con el promedio de las perturbaciones genéricas, no indica una dependencia del fondo mayor que la de esas perturbaciones. Sin embargo, ese promedio está dominado por el ruido y el desenfoque. Frente al brillo y la oclusión, la caída con fondo sustituido es entre 2 y 6 veces mayor.

En Grad-CAM, la energía del mapa en el fondo es menor que la proporción de área que este ocupa (cociente mediano entre 0.65 y 0.68), es decir, los mapas se concentran en el objeto. Concluimos que los tres modelos dependen en parte del fondo, sin que ese efecto domine su comportamiento. Ambas medidas son indirectas, por lo que no establecen el desempeño de los modelos con fondos reales.

## 7.5 Costo computacional

| | CNN | MobileNetV2 | ResNet-18 |
|---|---:|---:|---:|
| Parámetros | 1.2 M | 2.2 M | 11.2 M |
| GMAC por imagen | 2.59 | 0.30 | 1.81 |
| Inferencia en GPU (ms/imagen, lote 32) | 0.68 | 0.37 | 0.33 |
| Entrenamiento (min por corrida) | 8.4 | 8.5 | 8.3 |

En GPU, el mayor número de parámetros de ResNet-18 no aumenta el tiempo de inferencia ni de entrenamiento. MobileNetV2 requiere 6 veces menos operaciones que ResNet-18, pero no es más rápida en GPU, donde las convoluciones separables en profundidad aprovechan peor el paralelismo. La CNN es la más lenta porque sus primeras capas operan a resolución completa. Medimos los tiempos solo en GPU. En dispositivos de borde sin GPU el orden puede cambiar, y la menor cantidad de operaciones de MobileNetV2 podría compensar su menor desempeño.

## 7.6 Relación con la literatura

La accuracy de nuestro modelo de referencia (0.871) está 0.089 por debajo de la mediana reportada en TrashNet (~96 %, capítulo 3). La diferencia combina dos efectos que este estudio no separa: el preentrenamiento en ImageNet, que usan todos los estudios revisados, y el protocolo, porque entrenamos desde cero, con partición agrupada por casi-duplicados y varias semillas. La contribución del estudio está en el protocolo: la comparación usa réplicas, intervalos de confianza y pruebas pareadas, y su resultado es una cota de desempeño verificable para los modelos base.

## 7.7 Limitaciones

1. **Régimen de entrenamiento:** solo evaluamos modelos entrenados desde cero; no medimos el efecto del preentrenamiento.
2. **Búsqueda de hiperparámetros:** usamos una única partición de validación, una semilla y 50 épocas por configuración. El decaimiento de pesos elegido para los tres modelos (5e-4) y el dropout de 0.4 para dos de ellos están en el borde de la malla, por lo que el óptimo podría estar fuera de ella. El F1 de validación supera al de prueba en 0.01 a 0.04, lo que indica una selección algo optimista sobre esas 360 imágenes.
3. **Presupuesto de épocas:** 16 de las 24 corridas llegaron al máximo de 100 épocas, con la mejor época cerca del final. No podemos descartar que un presupuesto mayor mejore los tres modelos.
4. **Tamaño del conjunto de prueba:** 361 imágenes, de las cuales 19 son de `trash`. Los intervalos por clase son amplios, y la prueba de McNemar solo detecta diferencias en algunas semillas (en 4 de 8 para ResNet-18 frente a MobileNetV2).
5. **Datos:** el ruido de etiqueta no se auditó, y todas las imágenes son de estudio, con un solo objeto sobre fondo blanco. Los resultados no se extienden directamente a escenas reales.
6. **Determinismo:** algunas operaciones de cuDNN y la precisión mixta introducen una variación residual pequeña entre ejecuciones de una misma semilla.

## 7.8 Trabajo futuro

- Repetir la comparación con modelos preentrenados bajo el mismo protocolo, para separar el efecto de la transferencia del efecto de la arquitectura.
- Incluir ruido, desenfoque y sustitución de fondo en el aumento de datos, y evaluar con imágenes de residuos en condiciones de operación.
- Ampliar la malla de hiperparámetros en decaimiento de pesos y dropout, y aumentar el presupuesto de épocas.
- Completar la auditoría de etiquetas, empezando por las 19 imágenes que fallan los tres modelos.
