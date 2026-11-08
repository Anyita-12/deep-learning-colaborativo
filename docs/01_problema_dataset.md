# 1. Definición del problema y conjunto de datos

En este capítulo planteamos el problema de clasificar imágenes de residuos sólidos según su material, lo formalizamos y declaramos, antes de entrenar, la pregunta de investigación, la hipótesis, las métricas y la unidad de independencia que rigen todo el trabajo. También justificamos el uso de TrashNet y describimos su estructura técnica.

## 1.1 Contexto y motivación

En 2016 se generaron en el mundo 2010 millones de toneladas de residuos sólidos municipales. Al menos el 33 % no se gestionó de forma ambientalmente segura y solo el 13.5 % se recicló (Kaza et al., 2018). Para 2050 se proyecta que la generación anual alcance 3800 millones de toneladas (United Nations Environment Programme [UNEP] y International Solid Waste Association [ISWA], 2024). El aprovechamiento de esos residuos depende de separarlos por material, y la separación manual es lenta, costosa y expone a los operarios a residuos peligrosos (Masand et al., 2021).

La clasificación automática de imágenes es el componente de percepción de varios sistemas que se han propuesto para resolver este problema:

- **Plantas de reciclaje:** robots que separan residuos a partir del reconocimiento de imágenes (Mao et al., 2021).
- **Contenedores inteligentes:** contenedores que clasifican el residuo al depositarlo y lo dirigen al compartimento correspondiente (Yang y Li, 2020). Feng et al. (2022) reportan 146 ms por reconocimiento en su prototipo.
- **Aplicaciones móviles y dispositivos de borde:** clasificadores ejecutados en teléfonos o en hardware de bajo costo que informan al usuario sobre la disposición correcta del residuo en la fuente (Echchidmi y Bouayad, 2026; Huang et al., 2021; Rabano et al., 2018; N.-S. Vo et al., 2024).
- **Redes de Internet de las cosas:** entrenamiento federado entre dispositivos que conserva la privacidad de los datos locales (Quan et al., 2024).
- **Gestión de emergencias:** clasificación de residuos en zonas afectadas por inundaciones (Pitakaso et al., 2024).

En el ámbito regional, Meléndez-Villadiego et al. (2026), de la Universidad del Norte, entrenaron variantes de ResNet con TrashNet combinado con imágenes propias, capturadas en un recinto que simula las condiciones de operación de un sistema de clasificación.

El desempeño de estos sistemas depende de cuánto se puede confiar en el clasificador y de qué evidencia respalda la elección de su arquitectura. Este trabajo aporta esa evidencia para los modelos base: los comparamos con un protocolo común, con varias réplicas y con pruebas estadísticas, sobre una partición que controla la fuga de información.

## 1.2 Definición formal del problema

El proyecto corresponde al tipo **7. Clasificación de imágenes**: cada observación es una fotografía de un objeto, y la tarea consiste en asignarle una de seis clases de material.

**Espacio de entrada.** Una imagen RGB redimensionada y normalizada por canal:

$$\mathcal{X} = \mathbb{R}^{3 \times 224 \times 224}.$$

**Espacio de salida.** Las seis clases de TrashNet:

$$\mathcal{Y} = \{1, \dots, 6\} \equiv \{\texttt{cardboard}, \texttt{glass}, \texttt{metal}, \texttt{paper}, \texttt{plastic}, \texttt{trash}\}.$$

**Función objetivo.** Buscamos una función $f_\theta : \mathcal{X} \to \Delta^{5}$, donde $\Delta^{5}$ es el conjunto de vectores de 6 probabilidades que suman 1. La red asigna a cada imagen una distribución sobre las clases mediante *softmax*,

$$p_\theta(y = k \mid x) = \frac{\exp\big(z_k(x;\theta)\big)}{\sum_{j=1}^{6} \exp\big(z_j(x;\theta)\big)},$$

donde $z(x;\theta) \in \mathbb{R}^6$ son las salidas de la última capa (*logits*). La clase predicha es $\hat y = \arg\max_k \, p_\theta(y = k \mid x)$.

**Función de pérdida.** Minimizamos la entropía cruzada ponderada por clase sobre el conjunto de entrenamiento $\{(x_i, y_i)\}_{i=1}^{N}$:

$$\mathcal{L}(\theta) = -\frac{\sum_{i=1}^{N} w_{y_i} \log p_\theta(y_i \mid x_i)}{\sum_{i=1}^{N} w_{y_i}}, \qquad w_k = \frac{N}{K\, n_k},$$

donde $K = 6$ y $n_k$ es el número de imágenes de la clase $k$ en entrenamiento. El peso $w_k$ compensa el desbalance: una imagen de `trash`, la clase menos frecuente, pesa unas cuatro veces más que una de `paper` (sección 2.2). La regularización por decaimiento de pesos se aplica en el optimizador (AdamW) y no forma parte de $\mathcal{L}$.

La métrica primaria, el F1 macro, no es diferenciable respecto a $\theta$ porque depende de $\arg\max$. Por eso entrenamos con $\mathcal{L}$, que sí lo es, y evaluamos con F1 macro.

## 1.3 Pregunta de investigación e hipótesis

**Pregunta.** Al entrenar desde cero con el mismo protocolo, sobre una partición de TrashNet controlada por duplicados, ¿difieren en F1 macro de prueba una CNN estilo VGG reducida, MobileNetV2 y ResNet-18, y alguna diferencia supera el margen de relevancia práctica $\delta$?

Elegimos el entrenamiento desde cero porque la CNN no tiene pesos preentrenados. Preentrenar solo dos de los tres modelos confundiría el efecto de la arquitectura con el de la transferencia desde ImageNet.

**Hipótesis operativa.** Sea $F_m^{(s)}$ el F1 macro de prueba del modelo $m$ con la semilla $s$.

- $H_0$ global: las tres arquitecturas tienen la misma distribución de $F_m^{(s)}$.
- $H_1$ global: al menos una arquitectura difiere.
- Por pares: $H_0: \; \mathbb{E}\big[F_a^{(s)} - F_b^{(s)}\big] = 0$, con la diferencia pareada por semilla.

Una diferencia se considera relevante solo si es estadísticamente significativa y su magnitud alcanza $\delta$. El modelo con mayor F1 macro medio define el modelo de referencia, y su desempeño fija el umbral que cualquier modelo posterior debe superar.

## 1.4 Decisiones declaradas antes del entrenamiento

| Decisión | Valor |
|---|---|
| Métrica primaria | F1 macro en prueba: $\mathrm{F1}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} \frac{2\,P_k R_k}{P_k + R_k}$, con $P_k$ y $R_k$ la precisión y el recall de la clase $k$ |
| Métricas secundarias | Accuracy, balanced accuracy, F1 ponderado y por clase, precisión y recall macro, AUC-ROC y AUC-PR macro (uno contra resto), MCC y error de calibración esperado (ECE) |
| Margen de relevancia práctica | $\delta = 0.02$ en F1 macro |
| Nivel de significancia y potencia | $\alpha = 0.05$; potencia objetivo 0.80 |
| Unidad de independencia | El grupo de casi-duplicados: las fotografías casi idénticas de un mismo objeto forman un grupo, y cada grupo se asigna completo a una sola partición |
| Unidad de réplica | La semilla, que controla la inicialización, el orden de los lotes, el aumento de datos y el dropout; los tres modelos comparten semillas, por lo que las métricas están pareadas |
| Partición | Estratificada por clase y agrupada por casi-duplicados, 1800/360/361 imágenes (entrenamiento/validación/prueba), fija para todos los experimentos |

Elegimos el F1 macro porque asigna el mismo peso a las seis clases. Con el desbalance de TrashNet, la accuracy puede ser alta aunque el modelo falle en `trash`. Conservamos la accuracy como métrica secundaria porque es la que reporta la mayoría de los estudios (capítulo 3). El valor de $\delta$ corresponde a una mejora de dos puntos porcentuales en F1 macro, del orden de la variación esperada entre semillas. Estas decisiones se mantienen sin cambios en las etapas siguientes del proyecto.

## 1.5 Justificación del conjunto de datos

**Relevancia.** TrashNet clasifica residuos en las fracciones que separa un sistema de reciclaje: cartón, vidrio, metal, papel, plástico y rechazo (`trash`). Las clases corresponden a decisiones reales de aprovechamiento, y las confusiones entre ellas (vidrio y plástico, papel y cartón) son las mismas que dificultan la separación manual.

**Procedencia.** Yang y Thung (2016) construyeron el conjunto de datos en el curso CS229 de la Universidad de Stanford. Fotografiaron cada objeto sobre una cartulina blanca, con luz solar o de interior, usando tres teléfonos (iPhone 7 Plus, iPhone 5S e iPhone SE). Las fotografías originales (~3.5 GB) se redujeron a 512×384 píxeles con el script `resize.py` del repositorio, que además rota las imágenes verticales. En su informe, una SVM con descriptores SIFT alcanzó 63 % de accuracy y una CNN 22 %; tras cambiar la inicialización de pesos, la CNN llegó a cerca de 75 % (Thung, 2017).

**Disponibilidad y licencia.** El conjunto de datos es público en GitHub (Thung, 2017) y en Hugging Face (`garythung/trashnet`). El repositorio se distribuye bajo licencia MIT, que permite usar, copiar, modificar y redistribuir su contenido si se conserva el aviso de copyright.

**Uso previo en la literatura.** TrashNet es el conjunto de referencia para clasificar imágenes de residuos desde 2018, lo que permite comparar nuestros resultados con estudios previos:

- Aral et al. (2018) obtuvieron 95 % de accuracy con DenseNet121 ajustado, y Rabano et al. (2018) 87.2 % con MobileNet, que luego llevaron a una aplicación Android.
- Ruiz et al. (2019) compararon VGG, Inception y ResNet, con 88.6 % para un modelo Inception-ResNet; A. H. Vo et al. (2019) alcanzaron 94 % con una variante de ResNeXt.
- Los estudios posteriores reportan valores entre 94 % y 99.6 % con modelos preentrenados, mecanismos de atención u optimización de hiperparámetros (Feng et al., 2022; Huang et al., 2021; Mao et al., 2021; Yang y Li, 2020).

En nuestra revisión (capítulo 3), la mediana de accuracy de los 59 estudios incluidos es cercana a 96 %, pero solo 13 aplican pruebas estadísticas y 44 combinan TrashNet con otros conjuntos de datos. Además, las fotografías contienen un único objeto sobre fondo blanco. Abdu y Noor (2022) señalan que ese fondo limita la generalización a imágenes reales, y Mao et al. (2022) encontraron que un detector entrenado con TrashNet no sirve para escenas con varios objetos. Por eso medimos en el análisis exploratorio si el fondo y la iluminación predicen la clase (sección 2.7) y evaluamos los modelos con el fondo sustituido (sección 5.3).

## 1.6 Descripción técnica

| Aspecto | Descripción |
|---|---|
| Tamaño | 2527 imágenes en la versión publicada; 2521 tras eliminar 6 imágenes con etiqueta contradictoria (sección 2.3) |
| Estructura | Una carpeta por clase; el nombre del archivo combina la clase y un índice (`glass/glass12.jpg`) |
| Formato | JPEG, RGB de 8 bits por canal, 512×384 píxeles (proporción 4:3) en todas las imágenes |
| Dimensionalidad | $3 \times 384 \times 512 = 589\,824$ valores por imagen original; $3 \times 224 \times 224 = 150\,528$ tras el redimensionamiento |
| Tipo de dato y unidad de observación | Imagen fotográfica de un único objeto de residuo |
| Variable objetivo | Clase de material, con seis categorías nominales (tabla siguiente) |
| Etiquetado | Manual, por los autores, al fotografiar cada objeto |
| Dependencia entre observaciones | Casi-duplicados del mismo objeto y efectos de sesión de captura (fondo, iluminación, teléfono) |
| Versión | `dataset-resized.zip`, versión de 512×384 píxeles publicada por los autores (`garythung/trashnet`) |
| Suma de verificación | SHA-256 del archivo comprimido y del conjunto de prueba (abajo), registradas en `data/SHA256SUMS.txt` y verificadas antes de cada corrida |

```text
c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c  data/raw/trashnet/dataset-resized.zip
4e4651dc5c8bbbfcb53ca1d6ed761112c01b2b364e5a12105bd19de2360cbd87  test_set:data/splits/trashnet_split_v2_s1234.csv
```

**Distribución de clases.**

| Clase | Versión publicada | Tras el análisis exploratorio |
|---|---:|---:|
| `cardboard` | 403 | 403 |
| `glass` | 501 | 498 |
| `metal` | 410 | 409 |
| `paper` | 594 | 594 |
| `plastic` | 482 | 480 |
| `trash` | 137 | 137 |
| **Total** | **2527** | **2521** |

La razón de desbalance entre la clase más frecuente y la menos frecuente es 4.34 (`paper` frente a `trash`), y `trash` representa el 5.4 % del total (sección 2.2).

**Ruido de etiqueta esperado.** Cada imagen tiene una sola etiqueta asignada por los autores, sin revisión independiente. Esperamos ruido en tres situaciones: objetos de material mixto (un envase de cartón con plástico), pares de clases visualmente parecidas (vidrio y plástico transparentes, papel y cartón delgado) y la clase `trash`, que agrupa objetos heterogéneos que no pertenecen a las demás. El análisis exploratorio encontró 3 pares de imágenes idénticas con etiquetas distintas, que eliminamos (sección 2.3). La auditoría por muestreo del ruido restante quedó diseñada y se declara como limitación (sección 2.8).

**Estructura de dependencia.** TrashNet no incluye metadatos de objeto, sesión o fecha de captura, por lo que la dependencia se infiere de las imágenes:

- **Casi-duplicados:** algunos objetos se fotografiaron dos o más veces con un cambio leve de encuadre. Si una copia queda en entrenamiento y otra en prueba, la métrica de prueba sobreestima el desempeño. Agrupamos estas imágenes con un hash perceptual (2521 imágenes en 2512 grupos) y asignamos cada grupo completo a una partición (sección 2.4).
- **Efectos de sesión:** si cada clase se fotografió en sesiones distintas, el fondo y la iluminación pueden asociarse con la clase. El análisis exploratorio confirmó esa asociación (sección 2.7).

Fuera de los grupos de casi-duplicados, asumimos independencia entre fotografías. Este supuesto condiciona la partición y las pruebas estadísticas de los capítulos 4 y 5.

## Referencias

- Abdu, H. y Noor, M. H. M. (2022). Domestic trash classification with transfer learning using VGG16. En *2022 IEEE 12th International Conference on Control System, Computing and Engineering (ICCSCE)* (pp. 137–141). IEEE. https://doi.org/10.1109/ICCSCE54767.2022.9935653
- Aral, R. A., Keskin, Ş. R., Kaya, M. y Hacıömeroğlu, M. (2018). Classification of TrashNet dataset based on deep learning models. En *2018 IEEE International Conference on Big Data (Big Data)* (pp. 2058–2062). IEEE. https://doi.org/10.1109/BigData.2018.8622212
- Echchidmi, M. y Bouayad, A. (2026). Compact waste image classification with multi-student CNNs and edge-oriented model selection. *Frontiers in Artificial Intelligence*, *9*, Artículo 1804734. https://doi.org/10.3389/frai.2026.1804734
- Feng, Z., Yang, J., Chen, L., Chen, Z. y Li, L. (2022). An intelligent waste-sorting and recycling device based on improved EfficientNet. *International Journal of Environmental Research and Public Health*, *19*(23), Artículo 15987. https://doi.org/10.3390/ijerph192315987
- Huang, K., Lei, H., Jiao, Z. y Zhong, Z. (2021). Recycling waste classification using vision transformer on portable device. *Sustainability*, *13*(21), Artículo 11572. https://doi.org/10.3390/su132111572
- Kaza, S., Yao, L. C., Bhada-Tata, P. y Van Woerden, F. (2018). *What a waste 2.0: A global snapshot of solid waste management to 2050*. World Bank. https://doi.org/10.1596/978-1-4648-1329-0
- Mao, W.-L., Chen, W.-C., Fathurrahman, H. I. K. y Lin, Y.-H. (2022). Deep learning networks for real-time regional domestic waste detection. *Journal of Cleaner Production*, *344*, Artículo 131096. https://doi.org/10.1016/j.jclepro.2022.131096
- Mao, W.-L., Chen, W.-C., Wang, C.-T. y Lin, Y.-H. (2021). Recycling waste classification using optimized convolutional neural network. *Resources, Conservation and Recycling*, *164*, Artículo 105132. https://doi.org/10.1016/j.resconrec.2020.105132
- Masand, A., Chauhan, S., Jangid, M., Kumar, R. y Roy, S. (2021). ScrapNet: An efficient approach to trash classification. *IEEE Access*, *9*, 130947–130958. https://doi.org/10.1109/ACCESS.2021.3111230
- Meléndez-Villadiego, M., Huerta, M. K., Manzur-Urueta, J., Sierra González, O., Oliveros-Corredor, V., Viloria-Núñez, C. y Rueda-Delgado, D. (2026). Recycle AId: Training an artificial intelligence model based on computer vision for solid waste classification. *IEEE Access*, *14*, 35384–35395. https://doi.org/10.1109/ACCESS.2026.3662988
- Pitakaso, R., Srichok, T., Khonjun, S., Golinska-Dawson, P., Sethanan, K., Nanthasamroeng, N., Gonwirat, S., Luesak, P. y Boonmee, C. (2024). Optimization-driven artificial intelligence-enhanced municipal waste classification system for disaster waste management. *Engineering Applications of Artificial Intelligence*, *133*, Artículo 108614. https://doi.org/10.1016/j.engappai.2024.108614
- Quan, M. K., Nguyen, D. C., Nguyen, V.-D., Wijayasundara, M., Setunge, S. y Pathirana, P. N. (2024). Toward privacy-preserving waste classification in the Internet of Things. *IEEE Internet of Things Journal*, *11*(14), 24814–24830. https://doi.org/10.1109/JIOT.2024.3386727
- Rabano, S. L., Cabatuan, M. K., Sybingco, E., Dadios, E. P. y Calilung, E. J. (2018). Common garbage classification using MobileNet. En *2018 IEEE 10th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment and Management (HNICEM)* (pp. 1–4). IEEE. https://doi.org/10.1109/HNICEM.2018.8666300
- Ruiz, V., Sánchez, Á., Vélez, J. F. y Raducanu, B. (2019). Automatic image-based waste classification. En J. M. Ferrández Vicente, J. R. Álvarez-Sánchez, F. de la Paz López, J. Toledo Moreo y H. Adeli (Eds.), *From bioinspired systems and biomedical applications to machine learning* (pp. 422–431). Springer. https://doi.org/10.1007/978-3-030-19651-6_41
- Thung, G. (2017). *trashnet* [Conjunto de datos y código fuente]. GitHub. https://github.com/garythung/trashnet
- United Nations Environment Programme y International Solid Waste Association. (2024). *Global waste management outlook 2024: Beyond an age of waste – Turning rubbish into a resource*. United Nations Environment Programme. https://www.unep.org/resources/global-waste-management-outlook-2024
- Vo, A. H., Hoang Son, L., Vo, M. T. y Le, T. (2019). A novel framework for trash classification using deep transfer learning. *IEEE Access*, *7*, 178631–178639. https://doi.org/10.1109/ACCESS.2019.2959033
- Vo, N.-S., Nguyen, N.-T.-X., Le, G.-P., Nguyen, L.-T.-N. y Pham, H.-A. (2024). An efficient model on AIoT devices for trash classification applications. *SN Computer Science*, *5*, Artículo 630. https://doi.org/10.1007/s42979-024-02988-x
- Yang, M. y Thung, G. (2016). *Classification of trash for recyclability status* [Informe de proyecto, CS229: Machine Learning]. Stanford University. https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf
- Yang, Z. y Li, D. (2020). WasNet: A neural network-based garbage collection management system. *IEEE Access*, *8*, 103984–103993. https://doi.org/10.1109/ACCESS.2020.2999678
