# 6. Declaración a priori

En este capítulo reunimos las decisiones que fijamos antes de la búsqueda de hiperparámetros y de las corridas definitivas: métricas, umbrales, unidad de independencia, partición, protocolo de entrenamiento, regla de tamaño de muestra y análisis estadístico. Las registramos en un preregistro fechado el 8 de noviembre de 2026 a las 02:12 (`results/common/preregistro_2026-11-08.md`), que el código escribe una sola vez y no sobrescribe. Estas decisiones se mantienen en las etapas siguientes del proyecto: cualquier modelo posterior se evalúa con la misma partición, las mismas métricas y los mismos umbrales.

## 6.1 Problema e hipótesis

El proyecto es de **clasificación de imágenes** en seis clases de material (sección 1.2). La hipótesis nula global establece que las tres arquitecturas (CNN estilo VGG reducida, MobileNetV2 y ResNet-18), entrenadas desde cero con el mismo protocolo, tienen la misma distribución de F1 macro en prueba; la alternativa, que al menos una difiere. Por pares, la hipótesis nula establece que la diferencia media de F1 macro, pareada por semilla, es cero (sección 1.3).

## 6.2 Métricas

| Tipo | Métricas |
|---|---|
| Primaria | F1 macro en el conjunto de prueba |
| Secundarias | Accuracy, balanced accuracy, F1 ponderado y por clase, precisión y recall macro, AUC-ROC y AUC-PR macro (uno contra resto), coeficiente de correlación de Matthews (MCC) y error de calibración esperado (ECE) |
| Costo | Parámetros, operaciones por imagen (GMAC), tiempo de entrenamiento y de inferencia |

Elegimos el F1 macro porque asigna el mismo peso a las seis clases y no oculta un mal desempeño en `trash`, la clase minoritaria (sección 2.2). La accuracy se reporta para comparar con la literatura.

## 6.3 Umbrales de decisión

| Parámetro | Valor |
|---|---|
| Nivel de significancia | $\alpha = 0.05$ |
| Potencia objetivo | $1 - \beta = 0.80$ |
| Margen de relevancia práctica | $\delta = 0.02$ en F1 macro |
| Corrección por comparaciones múltiples | Holm, sobre los tres pares de modelos |

Una diferencia entre dos modelos es **relevante** si es significativa tras la corrección de Holm y su magnitud es al menos $\delta$. El modelo con mayor F1 macro medio es el modelo de referencia. Un modelo posterior debe superarlo en al menos $\max(\delta, \text{MDE})$, donde MDE es la diferencia mínima detectable del experimento con potencia de 0.80.

## 6.4 Unidad de independencia y unidad de réplica

- **Unidad de independencia:** el grupo de casi-duplicados. Las fotografías casi idénticas de un mismo objeto forman un grupo (2521 imágenes en 2512 grupos), y cada grupo se asigna completo a una sola partición. Fuera de los grupos, asumimos independencia entre fotografías (sección 1.6).
- **Unidad de réplica:** la semilla, que controla la inicialización de los pesos, el orden de los lotes, el aumento de datos y el dropout. Los tres modelos comparten partición y semillas, por lo que sus métricas están pareadas por semilla.

## 6.5 Partición

| Elemento | Valor |
|---|---|
| Archivo | `data/splits/trashnet_split_v2_s1234.csv` |
| Tamaños | 1800 entrenamiento, 360 validación y 361 prueba (≈ 71/14/14 %) |
| Esquema | Estratificado por clase y agrupado por casi-duplicados, sin las 6 imágenes con etiqueta contradictoria (sección 2.4) |
| Huella del conjunto de prueba | SHA-256 registrado en `data/SHA256SUMS.txt` y verificado antes de cada corrida |

Reglas de uso:

- La búsqueda de hiperparámetros, el early stopping y la selección de la mejor época usan solo el conjunto de validación.
- El conjunto de prueba se evalúa una vez por corrida, con el modelo de la mejor época.
- La normalización por canal y los pesos de clase se calculan solo con el conjunto de entrenamiento.

## 6.6 Protocolo de entrenamiento común

| Elemento | Decisión |
|---|---|
| Régimen | Desde cero (`weights=None`) para los tres modelos. La CNN no tiene pesos preentrenados, y preentrenar solo dos modelos confundiría el efecto de la arquitectura con el de la transferencia |
| Entrada | 224×224, redimensionamiento directo desde 512×384 con interpolación bicúbica; normalización con media y desviación estándar por canal del entrenamiento |
| Pérdida | Entropía cruzada ponderada por clase, $w_k = N/(K\,n_k)$ |
| Aumento de datos (solo entrenamiento) | `RandomResizedCrop(224, scale=(0.7, 1.0))`, volteos horizontal y vertical, `RandomRotation(15)`, `ColorJitter(0.2, 0.2, 0.2, 0.02)` |
| Optimización | AdamW, lote de 32, 3 épocas de calentamiento y decaimiento coseno hasta 1 % de la tasa inicial, recorte del gradiente en 5, precisión mixta |
| Duración | Máximo 100 épocas; early stopping con paciencia de 15 épocas sobre el F1 macro de validación |
| Búsqueda de hiperparámetros | Malla de 8 configuraciones, igual para los tres modelos: tasa de aprendizaje {1e-3, 3e-4} × decaimiento de pesos {1e-4, 5e-4} × dropout {0.2, 0.4}; semilla 100; máximo 50 épocas; selección por F1 macro de validación |

## 6.7 Tamaño de muestra

El número de semillas $R$ se calcula con un piloto de 5 semillas (0 a 4), ejecutado con la configuración elegida en la búsqueda de hiperparámetros. Para cada par de modelos,

$$R = \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \sigma_d^2}{\delta^2},$$

donde $\sigma_d$ es la desviación estándar de las diferencias pareadas de F1 macro de validación en el piloto. Tomamos el máximo de los tres pares, redondeado hacia arriba, y lo acotamos a $R_{\text{final}} = \min(\max(R, 6), 10)$. El mínimo de 6 permite aplicar pruebas por rangos y el máximo de 10 lo impone el tiempo de GPU disponible. Las corridas del piloto forman parte del experimento final porque usan la misma configuración.

## 6.8 Análisis estadístico planeado

| Nivel | Prueba |
|---|---|
| Global | Friedman, con las semillas como bloques y los modelos como tratamientos; $p$ exacto por permutación y $W$ de Kendall |
| Por pares | t pareada y Wilcoxon de rangos con signo, con Holm; diferencia media con IC 95 % (t y bootstrap); tamaños de efecto $d_z$ y correlación rango-biserial; Nemenyi y Conover como complemento |
| Por imagen | McNemar por par y por semilla (exacto si $b + c < 25$), con Holm dentro de cada semilla |
| AUC-ROC | Bootstrap estratificado y pareado sobre las imágenes de prueba (10 000 remuestreos) |
| Varianza | ANOVA de un factor sobre las corridas, con $\eta^2$ e ICC(1) |
| Robustez | F1 macro con ruido gaussiano ($\sigma$ = 0.05 y 0.10), desenfoque, brillo ±30 %, oclusión y fondo sustituido; comparación pareada de la caída con fondo sustituido frente a la caída media de las perturbaciones genéricas |

Los resultados de este análisis se presentan en el capítulo 5 y se resumen en el capítulo 7.
