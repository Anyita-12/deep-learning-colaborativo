# 8. Benchmark de diez modelos: problema, preguntas e hipótesis

Los capítulos 1 a 7 fijaron el problema, la partición, la métrica primaria, el margen de relevancia práctica y el modelo de referencia (ResNet-18 entrenada desde cero). En los capítulos 8 a 19 ampliamos la comparación a diez modelos: los tres modelos base y siete modelos publicados sobre TrashNet, seleccionados con la revisión sistemática del capítulo 10, que amplía la del capítulo 3. Todos se evalúan con la misma partición, las mismas métricas y los mismos umbrales. Ninguno de los elementos declarados en los capítulos 1, 2 y 6 se modifica.

## 8.1 Resumen estructurado

**Problema.** Clasificar imágenes de TrashNet en seis materiales; métrica primaria: F1 macro en prueba, con margen de relevancia $\delta = 0.02$.

**Método.** Seleccionamos siete modelos publicados de seis familias entre 62 estudios, con un puntaje explícito. Los entrenamos con un protocolo preregistrado: partición agrupada por casi-duplicados, la misma búsqueda de 8 configuraciones y 8 semillas por modelo. Los comparamos con los tres modelos base y dos baselines no profundos (Friedman, t pareada y Wilcoxon con Holm, McNemar, DeLong y equivalencia). Replicamos el paper guía (Celik, 2025) y evaluamos ocho variantes de ablación.

**Resultados.** RWCNet obtiene el mayor F1 macro ($0.945 \pm 0.010$) y ViT-B16 el segundo ($0.938 \pm 0.014$). Ambos superan a ResNet-18 por 0.082 y 0.075 ($p$ Holm $\leq 0.004$) y son equivalentes entre sí dentro de $\pm\delta$ (TOST $p = 0.024$). La accuracy publicada del paper guía (99.4 %) no se replica (su protocolo da $0.925 \pm 0.005$; elegir la época con la prueba la infla 2.8 puntos), aunque el mismo pipeline reproduce RWCNet y GFN. En el paper guía aporta el preentrenamiento (0.085) y en RWCNet su extractor principal (0.033) y la supervisión profunda (0.022); en ninguno la segunda red tiene un aporte significativo. Con ruido gaussiano, el F1 de las CNN cae entre 0.30 y 0.67, y el de ViT-B16, 0.04.

**Conclusión.** Los modelos preentrenados superan de forma significativa y relevante al modelo de referencia del E1. La fragilidad ante el ruido, compartida por las arquitecturas convolucionales, es la limitación que atacará el modelo propio.

## 8.2 Tipo de problema y ruta metodológica

El problema sigue siendo de **clasificación de imágenes** (tipo 7, familia F4 de la ruta metodológica). Seguimos la ruta F4a de clasificación de imágenes:

| Paso de la ruta F4a | Dónde se desarrolla |
|---|---|
| I1. Tarea y unidad de independencia | Secciones 1.2 y 6.4: grupo de casi-duplicados |
| I2. Duplicados y partición por unidad de independencia | Sección 2.4 y auditoría de fuga del capítulo 9 |
| I3. Normalización calculada solo en entrenamiento | Sección 6.6 y capítulo 11 (estadísticas de ImageNet para los modelos preentrenados) |
| I4. Aumento de datos solo en entrenamiento | Sección 6.6, sin cambios |
| I5. Baselines y 10 modelos con igual presupuesto | Capítulos 11 y 13 |
| I6. Misma política de inferencia | Sin *test-time augmentation* en ningún modelo |
| I7. Pruebas de ajuste, sobreajuste y comparación | Capítulos 13 y 14 |
| I8. Interpretabilidad y análisis de errores | Capítulo 15 |

La formulación matemática de la sección 1.2 se mantiene. Cada modelo define una función $f_\theta : \mathcal{X} \to \Delta^5$, y el riesgo esperado

$$R(f) = \mathbb{E}_{(x, y) \sim P}\big[\mathcal{L}(f(x), y)\big]$$

se estima con el riesgo empírico sobre el conjunto de prueba sellado, de 361 imágenes. El espacio de hipótesis $\mathcal{H}$ cambia en cada modelo; sus arquitecturas se describen en los capítulos 10 y 12. Tres modelos sustituyen la entropía cruzada ponderada por una pérdida propia, siempre con los mismos pesos de clase: FocusRCNet usa destilación de conocimiento, ResMsCapsule la pérdida de margen de las redes de cápsulas y RWCNet suma pérdidas de sus salidas auxiliares.

## 8.3 Elementos heredados del capítulo 6

| Elemento | Valor |
|---|---|
| Métrica primaria | F1 macro en el conjunto de prueba |
| Métricas secundarias | Accuracy, balanced accuracy, F1 ponderado y por clase, precisión y recall macro, AUC-ROC y AUC-PR macro, MCC, ECE; se añaden NLL y Brier |
| Margen de relevancia práctica | $\delta = 0.02$ en F1 macro |
| Umbral de mejora sobre el modelo de referencia | $\max(\delta, \text{MDE}) = 0.036$ (sección 7.2) |
| Nivel de significancia y potencia | $\alpha = 0.05$, $1 - \beta = 0.80$ |
| Unidad de independencia | Grupo de casi-duplicados (2521 imágenes en 2512 grupos) |
| Unidad de réplica | Semilla; 8 semillas (0 a 7) compartidas por todos los modelos |
| Partición | `trashnet_split_v2_s1234.csv`: 1800 / 360 / 361, huella del conjunto de prueba verificada antes de cada corrida |

## 8.4 Preguntas de investigación e hipótesis

Sea $F_m^{(s)}$ el F1 macro de prueba del modelo $m$ con la semilla $s$, y $D_{a,b}^{(s)} = F_a^{(s)} - F_b^{(s)}$ la diferencia pareada por semilla.

| RQ | Pregunta | Hipótesis | Prueba y criterio | Tipo |
|---|---|---|---|---|
| RQ1 | ¿Difieren los diez modelos en F1 macro de prueba? | $H_0$: los diez modelos tienen la misma distribución de $F_m^{(s)}$. $H_1$: al menos uno difiere | Friedman con $p$ por permutación (bloques = semillas) y corrección de Iman-Davenport; $W$ de Kendall; rechazo si $p < 0.05$ | Confirmatoria |
| RQ2 | ¿Supera algún modelo publicado al modelo de referencia del E1 (ResNet-18) por al menos el umbral de 0.036? | $H_0$: $\mathbb{E}[D_{m,\text{ResNet-18}}] = 0$. $H_1$: $\mathbb{E}[D_{m,\text{ResNet-18}}] \neq 0$ | t pareada con Holm sobre los 45 pares preregistrados, Wilcoxon y bootstrap como contraste; relevante si $p_{\text{Holm}} < 0.05$ y diferencia $\geq \delta$; mejora sobre el E1 si además es $\geq 0.036$ | Confirmatoria |
| RQ3 | ¿Se reproduce la accuracy publicada del paper guía (99.40 %) con su código, partición y métrica? | $H_0$: la accuracy reportada pertenece al IC 95 % de la réplica | Criterio fijado antes de ejecutar: éxito si 0.994 cae dentro del IC 95 % bootstrap de cada corrida | Confirmatoria |
| RQ4 | ¿Aportan la fusión multinivel (*hypercolumn*), la combinación de dos redes y el preentrenamiento del paper guía? | $H_0$: $\mathbb{E}[F_{\text{completo}} - F_{\text{variante}}] = 0$ para cada variante | t pareada con Holm sobre las 5 variantes; relevante si $\geq \delta$ | Exploratoria (no figura en el preregistro) |
| RQ5 | ¿Son los modelos con atención global más robustos al ruido gaussiano que las arquitecturas convolucionales? | $H_0$: la caída de F1 con ruido $\sigma = 0.05$ de ViT-B16 es igual a la de cada CNN | IC 95 % bootstrap por grupo de la caída y diferencia pareada por semilla | Exploratoria (las perturbaciones sí figuran en el preregistro) |

Las RQ confirmatorias tienen familias de comparaciones fijadas antes de las corridas definitivas: los 45 pares de modelos (RQ1 y RQ2) y las 10 corridas de la réplica (RQ3). Las exploratorias generan hipótesis para la etapa siguiente del proyecto y no se usan para seleccionar el modelo final. El capítulo 11 detalla el preregistro y la aclaración sobre la prueba de decisión por pares.

## 8.5 Organización

| Capítulo | Contenido |
|---|---|
| 9 | Ficha de datos, contrato de datos, EDA orientado a decisiones y auditoría de fuga |
| 10 | Revisión sistemática ampliada, selección de los siete modelos, análisis por modelo, búsqueda transversal de mecanismos y brecha |
| 11 | Preregistro, protocolo común, presupuesto de ajuste, fichas de modelo y hardware |
| 12 | Paper guía: descripción, réplica, limitaciones cuantificadas y ablación |
| 13 | Benchmark: tabla principal, ranking, inferencia estadística y eficiencia |
| 14 | Bondad de ajuste, sobreajuste, calibración e incertidumbre |
| 15 | Robustez, interpretabilidad, análisis de errores, techo de desempeño y ablación del mejor modelo |
| 16 | Discusión: causas de las diferencias, robustez frente a costo, techo de desempeño y amenazas a la validez |
| 17 | Reproducibilidad: entorno, lista de verificación y comandos |
| 18 | Conclusiones: respuesta a las RQ, modelo recomendado, alcance, brecha, hipótesis y trabajo futuro |
| 19 | Referencias en formato APA, con su versión BibTeX |
