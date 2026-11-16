# Sección 3.1: estrategia de búsqueda de la revisión sistemática

Este documento reúne todo lo que pide la sección 3.1 del enunciado:

1. cadenas de búsqueda con fecha y número de resultados;
2. bases de datos;
3. periodo;
4. criterios de inclusión y exclusión;
5. diagrama PRISMA;
6. hoja de extracción y acuerdo entre revisores;
7. valoración de la calidad metodológica;
8. búsqueda transversal de mecanismos y tabla de mecanismos.

La revisión tiene **dos búsquedas**, y el informe debe presentarlas así:

1. **Búsqueda original (7-11-2026)**, la que produjo el corpus.
   - Se hizo en Scopus, IEEE Xplore, Web of Science y arXiv, con la cadena "TrashNet".
   - La documentación completa (bitácora, 168 registros con su decisión, PRISMA, calidad y síntesis) está en `datos/seleccion_PRISMA_TrashNet_busqueda_original.xlsx`.
   - **Es el PRISMA principal.**
2. **Búsqueda ampliada y transversal (16-11-2026).**
   - Cadenas con sinónimos sobre el problema (P1–P3), ejecutadas en arXiv, OpenAlex e IEEE Xplore, que sirven para medir qué deja fuera la restricción a "TrashNet".
   - Las cadenas transversales de mecanismos (T1–T6) que pide el enunciado.
   - Es reproducible con los scripts de `scripts/`, y las respuestas crudas de las APIs están en `respuestas_crudas/`.

> **Estado: la sección 3.1 está completa.**
> - **Bases:** Scopus, IEEE Xplore y Web of Science (búsqueda original), más arXiv como complementaria.
> - **Cadenas registradas:** todas, con fecha y conteo.
> - **PRISMA:** completo.
> - **Cuartiles SJR:** listos para los 7 del benchmark.
> - **Kappa entre los dos revisores humanos:** 0,857, ya calculado.

---

## 1. Contexto: cómo se construyó el corpus

- **Búsqueda original (7-11-2026, entre las 16:25 y las 17:15).** Primero se probaron cuatro cadenas exploratorias amplias en Scopus:

  | Cadena exploratoria | Resultados | Motivo para descartarla |
  |---|---|---|
  | 1 | 1407 | Demasiado amplia: incluía residuos de construcción y otros temas |
  | 2 | 982 | Aún ruidosa (por ejemplo, ruteo de recolección hospitalaria) |
  | 3 | 559 | Incluía espectroscopía, no imágenes |
  | 4 | 477 | Inmanejable en el tiempo disponible |

  Después se fijó como definitiva la cadena **"TrashNet"**, para garantizar la comparabilidad con el dataset del proyecto. Se ejecutó en Scopus, IEEE Xplore, Web of Science y arXiv.
  - Se identificaron **301 registros** y quedaron **168 únicos**.
  - El cribado por título y resumen lo hizo Claude de forma preliminar, siguiendo la "Guía_criterios" del Excel. Dos revisores humanos lo verificaron en una muestra común de 25 registros (sección 6.3).
  - La lectura a texto completo se hizo para los 66 registros de **prioridad Alta**.
  - Quedaron **59 estudios incluidos**. Son los PDFs del corpus de Zotero, con sus IDs R001–R168.
- **Búsqueda ampliada (16-11-2026).** Mide cuánto se pierde por usar solo "TrashNet":
  - con sinónimos, P1 da 3326 registros en OpenAlex y 1839 en IEEE Xplore, frente a 242 y 77 con "TrashNet";
  - aun así, **recupera 60 de los 63 registros del corpus (95 %)**.

  La restricción a "TrashNet" se declara como limitación, ya anotada en la bitácora original: pueden quedar fuera estudios con otros datasets. La lista de candidatos de la búsqueda ampliada queda para una extensión futura.

## 2. Bases de datos y periodo

| Base | Rol | Búsqueda original (7-11-2026) | Búsqueda ampliada y transversal (16-11-2026) |
|---|---|---|---|
| **Scopus** | Principal | `TITLE-ABS-KEY("TrashNet")`, 2018–2026, artículo o ponencia, inglés: **161** | No se repitió: la sesión disponible solo tenía acceso "Preview" |
| **IEEE Xplore** | Principal | Título, resumen o palabras clave de autor = "TrashNet", 2018–2026: **76** | P1–P3 y T1–T6 en el campo "All Metadata", 2018–2026, con captura de cada cadena |
| **Web of Science** | Principal | `TS=("TrashNet")`, 2018–2026, artículo o ponencia, inglés: **61 → 60** | — |
| **arXiv** | Complementaria (como pide el enunciado) | `all=TrashNet`, 2018–2026: **4** | API oficial, búsqueda en el resumen |
| **OpenAlex** | Auxiliar: permite descargar los registros completos para el cribado automático | — | API, filtro `title_and_abstract.search` |

- **Periodo:** del 1 de enero de 2018 al 31 de diciembre de 2026, unos 8 años. Se justifica porque TrashNet se publicó en 2016–2017 y la literatura de Deep Learning sobre este problema empieza en 2018.
- **Trabajos fundacionales anteriores:** se admiten solo como fuentes primarias de mecanismos (sección 8): hypercolumns, 2015; destilación, 2015; supervisión profunda, 2015.
- **Por qué OpenAlex en la búsqueda ampliada:** las APIs de Scopus e IEEE requieren una clave institucional. OpenAlex es abierta y permite descargar y guardar los registros completos, así que sobre ella se hacen el cribado automático y la deduplicación. IEEE Xplore aporta los conteos en una base de la lista del enunciado.

## 3. Cadenas de búsqueda, fecha y número de resultados

Las cadenas usan comillas para frases exactas, `OR` entre sinónimos y `AND` entre conceptos. OpenAlex aplica stemming, así que no se usan comodines. La consulta exacta enviada a cada API está en `datos/registro_cadenas.csv`.

### 3.0 Búsqueda original (definitiva, 7-11-2026)

| Base | Hora | Cadena exacta | Filtros | Antes de filtros | Final | Exportación |
|---|---|---|---|---|---|---|
| Scopus | 16:29 | `TITLE-ABS-KEY ( "TrashNet" ) AND PUBYEAR > 2017 AND PUBYEAR < 2027 AND ( DOCTYPE ( ar ) OR DOCTYPE ( cp ) ) AND LANGUAGE ( english )` | En la cadena | No registrado | 161 | `scopus_cadenaA_2026-11-07.csv` |
| IEEE Xplore | 16:43 | `(("Document Title":"TrashNet") OR ("Abstract":"TrashNet") OR ("Author Keywords":"TrashNet"))` | Años 2018–2026 | 76 | 76 (69 ponencias y 7 revistas) | `ieee_cadenaA_2026-11-07.csv` |
| Web of Science | 16:54 | `TS=("TrashNet") AND PY=(2018-2026) AND DT=("Article" OR "Proceedings Paper") AND LA=("English")` | En la consulta | 61 | 60 | `wos_cadenaA_2026-11-07` (Parte 1 y 2) |
| arXiv | 17:15 | `all=TrashNet` (búsqueda avanzada) | 2018-01-01 a 2026-12-31 | 4 | 4 | Capturas transcritas |

Las cadenas exploratorias de Scopus (16:25 a 16:30: 1407, 982, 559 y 477 resultados) están, con su texto exacto, en la hoja "Bitácora_búsquedas" del Excel. Se declaran en la metodología.

### 3.1 Búsqueda ampliada sobre el problema (16-11-2026)

| ID | Cadena | Concepto que cubre |
|---|---|---|
| P1 | `("waste" OR "garbage" OR "trash" OR "litter" OR "recyclable") AND ("classification" OR "sorting" OR "recognition") AND ("deep learning" OR "convolutional neural network" OR "CNN" OR "vision transformer" OR "neural network") AND ("image" OR "images" OR "vision")` | Cadena general, con sinónimos del dominio, la tarea y el método |
| P2 | `"TrashNet"` | Trabajos sobre el dataset del proyecto |
| P3 | `("waste" OR "garbage" OR "trash") AND ("transfer learning" OR "fine-tuning" OR "pretrained" OR "pre-trained") AND ("image classification")` | Transferencia desde ImageNet, el régimen del E2 |

| ID | Base | Fecha y hora de ejecución (UTC) | N.º de resultados |
|---|---|---|---|
| P1 | arXiv | 2026-11-16 19:21:55 | 88 |
| P1 | OpenAlex | 2026-11-16 19:21:59 | 3326 |
| P2 | arXiv | 2026-11-16 19:22:33 | 4 |
| P2 | OpenAlex | 2026-11-16 19:22:37 | 242 |
| P3 | arXiv | 2026-11-16 19:22:40 | 7 |
| P3 | OpenAlex | 2026-11-16 19:22:44 | 150 |
| P1 | IEEE Xplore | 2026-11-16 19:48:30 | 1839 |
| P2 | IEEE Xplore | 2026-11-16 19:48:41 | 77 |
| P3 | IEEE Xplore | 2026-11-16 19:48:52 | 208 |

### 3.2 Búsqueda transversal de mecanismos

Cada cadena combina, como exige el enunciado:
- (i) la familia del mecanismo;
- (ii) una modalidad de origen distinta del problema;
- (iii) "transfer" o "adaptation".

| ID | Mecanismo (modelo del Top 10) | Cadena |
|---|---|---|
| T1 | Fusión multinivel (Celik-HC) | `("hypercolumn" OR "multi-level feature fusion" OR "multi-layer feature fusion" OR "multi-scale feature fusion") AND ("medical image" OR "remote sensing" OR "speech" OR "time series") AND ("transfer" OR "adaptation")` |
| T2 | Destilación (FocusRCNet) | `("knowledge distillation") AND ("speech" OR "natural language" OR "time series" OR "sensor" OR "EEG") AND ("transfer" OR "adaptation")` |
| T3 | Robustez al ruido (limitación del paper guía) | `("noise robustness" OR "robustness to noise" OR "corruption robustness" OR "gaussian noise") AND ("convolutional neural network" OR "vision transformer") AND ("speech" OR "audio" OR "medical image" OR "remote sensing") AND ("adaptation" OR "transfer")` |
| T3b | Robustez al ruido, versión ampliada | `("noise robustness" OR "robustness to noise" OR "corruption robustness" OR "noise augmentation" OR "noisy images") AND ("deep learning" OR "neural network") AND ("speech" OR "audio" OR "medical image" OR "remote sensing") AND ("adaptation" OR "transfer")` |
| T4 | Cápsulas (ResMsCapsule) | `("capsule network" OR "capsule networks") AND ("text" OR "signal" OR "EEG" OR "time series" OR "medical image") AND ("transfer" OR "adaptation")` |
| T5 | Supervisión profunda (RWCNet) | `("deep supervision" OR "auxiliary loss" OR "auxiliary classifier") AND ("segmentation" OR "speech" OR "natural language") AND ("transfer" OR "adaptation")` |
| T6 | Atención de canal y espacial (GFN, FocusRCNet) | `("channel attention" OR "spatial attention" OR "CBAM") AND ("time series" OR "signal" OR "speech" OR "text") AND ("transfer" OR "adaptation")` |

| ID | arXiv: fecha y hora (UTC) | arXiv: N | OpenAlex: fecha y hora (UTC) | OpenAlex: N | IEEE Xplore: fecha y hora (UTC) | IEEE Xplore: N |
|---|---|---|---|---|---|---|
| T1 | 2026-11-16 19:22:45 | 21 | 2026-11-16 19:22:50 | 89 | 2026-11-16 19:49:34 | 274 |
| T2 | 2026-11-16 19:22:51 | 275 | 2026-11-16 19:22:56 | 789 | 2026-11-16 19:49:48 | 678 |
| T3 | 2026-11-16 19:22:58 | 3 | 2026-11-16 19:23:02 | 7 | 2026-11-16 19:50:20 | 180 |
| T3b | 2026-11-16 19:30:08 | 15 | 2026-11-16 19:30:12 | 28 | 2026-11-16 19:50:34 | 95 |
| T4 | 2026-11-16 19:23:02 | 10 | 2026-11-16 19:23:06 | 83 | 2026-11-16 19:50:46 | 161 |
| T5 | 2026-11-16 19:23:07 | 63 | 2026-11-16 19:23:11 | 137 | 2026-11-16 19:50:59 | 219 |
| T6 | 2026-11-16 19:23:12 | 91 | 2026-11-16 19:23:17 | 447 | 2026-11-16 19:51:12 | 1100 |

IEEE Xplore busca también en las palabras clave indexadas (campo "All Metadata"), por eso da más resultados que OpenAlex en las cadenas transversales.

- **T3 devolvió muy pocos registros** (3 y 7) por exigir 4 conceptos a la vez. Por eso se registró una versión ampliada, T3b, que cambia "CNN o ViT" por "deep learning o neural network". Se reportan **las dos**, como pide el principio de registrar cada cadena ejecutada.
- **T2 y T6** se revisaron solo en sus registros más citados: 200 de OpenAlex y 100 de arXiv.

## 4. Criterios de inclusión y exclusión

**Inclusión** (todos deben cumplirse):

| Código | Criterio |
|---|---|
| I1 | Estudio primario (artículo de revista, congreso o preprint) |
| I2 | Dominio: residuos sólidos domésticos o municipales reciclables (basura, reciclables, TrashNet y similares) |
| I3 | Datos: imágenes (RGB) |
| I4 | Tarea: clasificación o reconocimiento del tipo de residuo |
| I5 | Método: Deep Learning |
| I6 | Periodo 2018–2026 |

**Exclusión en título y resumen** (se registra la primera regla que falla):

| Código | Razón |
|---|---|
| X0 | Tipo de documento no elegible: tesis, dataset, software, editorial, libro, resumen de congreso, fe de erratas o retractación |
| X1 | Revisión, encuesta o estudio bibliométrico (no primario) |
| X2 | Fuera del dominio: "waste" en otro sentido, residuos de alimentos, electrónicos, de construcción, médicos o aguas residuales |
| X3 | No usa imágenes: sensores, espectros, texto o series |
| X4 | Tarea distinta de la clasificación: solo detección, segmentación, conteo o predicción |
| X5 | No usa Deep Learning |

**Criterios a texto completo** (los de calidad de la sección 3.1):
- métricas reportadas;
- detalle suficiente para reproducir (arquitectura, partición y entrenamiento) o código;
- calidad de la revista o el congreso.

Los dos primeros se registran en la hoja de extracción (sección 6). La calidad de la revista se mide con el "2-year mean citedness" de OpenAlex (equivalente al factor de impacto) para todos los estudios, y con el **cuartil SJR** para los 7 del benchmark (consultado en scimagojr.com el 16-11-2026, en el año de publicación; `datos/cuartiles_sjr_top7.csv`).

**Cómo se aplicó el cribado:**
- **Búsqueda original:**
  - Cribado preliminar de título y resumen hecho por Claude con la "Guía_criterios" del Excel, sobre los 168 registros.
  - **Verificación humana:** dos revisores (H1 y H2) decidieron de forma independiente la muestra común de 25 registros. Las 2 discrepancias se resolvieron como se explica en la sección 6.3. Una de ellas (R132) pasó de Excluir a Dudoso, así que quedan 24 exclusiones: 23 sin relación con la clasificación de imágenes de residuos y 1 que no usa Deep Learning.
  - Lectura a texto completo de los registros de prioridad Alta, con los criterios de métricas, línea base y protocolo.
- **Búsqueda ampliada:** cribado automático con reglas de expresiones regulares sobre el título y el resumen (`scripts/cribado.py`, códigos X0–X5). Los 538 registros sin resumen se cribaron solo por título, con criterio conservador.

## 5. Diagrama PRISMA 2020 (búsqueda original)

```
IDENTIFICACIÓN (7-11-2026, cadena "TrashNet", 2018-2026)
  Registros identificados: 301
    · Scopus 161 · IEEE Xplore 76 · Web of Science 60 · arXiv 4 (complementaria)
  Duplicados eliminados (DOI o título normalizado): 133
            │
CRIBADO     ▼
  Registros cribados por título y resumen: 168 (todos con resumen)
  Excluidos: 24  (selección de Claude, verificada por 2 revisores humanos
                  en una muestra común de 25: κ = 0,857)
    · Sin relación directa con clasificación de imágenes de residuos ... 23
    · No usa deep learning .............................................. 1
            │
ELEGIBILIDAD▼
  Reportes buscados a texto completo: 144 (113 Incluir + 31 Dudoso)
    · Fuera del corte de prioridad (Media 38 y Baja 40, no leídos) ..... 78 (*)
    · No recuperados (sin PDF; R021, R028, R086, R092) .................. 4
  Reportes evaluados a texto completo: 62 (prioridad Alta)
  Excluidos a texto completo: 3
    · Sin relación directa con clasificación de imágenes (detección YOLOv3; R007) ... 1
    · Sin modelo de referencia o línea base (R029, R032) ............................. 2
            │
INCLUSIÓN   ▼
  Estudios incluidos en la síntesis: 59
  Seleccionados para el benchmark (sección 3.2): 7, más los 3 modelos base del E1
```

(*) **Corte de prioridad, que hay que declarar:** prioridad Alta significa un artículo de revista marcado Incluir, o un registro con 10 o más citas. Con 144 candidatos no hubo tiempo de leerlos todos. Los 78 de prioridad Media o Baja quedan identificados en el Excel.

Todos los números se pueden verificar en la hoja "Registros" del Excel (columnas "Decisión final T/R", "Prioridad texto completo", "Decisión TC" y "Motivo exclusión TC"). La hoja "PRISMA" los recalcula sola al abrir el archivo en Excel.

### 5.1 Flujo de la búsqueda ampliada (verificación de cobertura, 16-11-2026)

| Etapa | n |
|---|---|
| Identificados en OpenAlex (P1 3326 + P2 242 + P3 150) | 3718 |
| Identificados en arXiv (P1 88 + P2 4 + P3 7) | 99 |
| Solo conteo en IEEE Xplore (P1 1839 + P2 77 + P3 208) | 2124 |
| Duplicados eliminados | 383 |
| Cribados por título y resumen | 3434 |
| Excluidos: X0 tipo de documento 198; X1 revisión 146; X2 fuera de dominio 1278; X3 sin imágenes 2; X4 otra tarea 241; X5 sin DL 11 | 1876 |
| Pasan el cribado | 1558 |
| De ellos, ya están en el corpus original | 59 |

**Lectura:**
- La cadena "TrashNet" capta el núcleo de la literatura comparable: el 95 % del corpus aparece también en la búsqueda ampliada.
- Hay unos 1500 trabajos de clasificación de residuos con otros datasets que no se revisaron. Es una limitación declarada.
- El único registro del corpus que las reglas automáticas excluyen (X4, detección) es el mismo R007 que el equipo excluyó a texto completo. Eso muestra que los criterios son coherentes entre las dos búsquedas.

## 6. Hoja de extracción estructurada y acuerdo entre revisores

### 6.1 Hoja de extracción

Archivo: `datos/hoja_extraccion.csv`, con 63 filas, una por PDF del corpus de Zotero. La columna `decision_TC_equipo` indica cuáles son los 59 incluidos, los 3 excluidos a texto completo y el que no tiene PDF. La columna `id_registro_original` enlaza cada fila con el Excel. El Excel original también trae su propia hoja de calidad ("Calidad") y de síntesis ("Tabla_síntesis"), hechas en el E1. La hoja de extracción las amplía con los campos que pide el E2.

**Campos fijos:**

| Grupo | Campos |
|---|---|
| Identificación | autores, año, título, DOI, revista o congreso, tipo |
| Impacto | citas (OpenAlex), 2-year mean citedness de la revista |
| Arquitectura y datos | familias de arquitectura, datasets, partición (proporciones), k-fold, conjunto de validación |
| Resultados | métricas, accuracy del resumen |
| Rigor | repeticiones o semillas, pruebas estadísticas, código público, baselines ajustados por igual, control de duplicados o fuga |
| Calidad | riesgo de fuga, repeticiones, equidad de baselines |
| Trazabilidad | etiqueta del equipo, seleccionado para el Top 7, verificado a mano |

**Método de extracción:**
- Los 63 PDFs se extrajeron con un script sobre el texto completo de los PDFs (`scripts/extraccion.py`, con patrones explícitos).
- **Los 7 papers del benchmark se verificaron a mano** (columna `verificado_manual = Sí`). En el caso de Celik-HC, la verificación se hizo contra su código.
- Las filas automáticas son orientativas: un patrón puede confundir una mención con un uso (por ejemplo, que un paper cite TACO no implica que lo use).

**Síntesis de los 59 incluidos:**

| Campo | Resultado |
|---|---|
| Año | 2018: 2; 2019: 2; 2020: 4; 2021: 5; 2022: 8; 2023: 3; 2024: 11; **2025: 12; 2026: 12** (el 59 % es de 2024 a 2026) |
| Tipo | 52 artículos de revista y 7 de congreso |
| Dataset | Todos usan TrashNet (criterio de búsqueda; la extracción automática lo detecta en 57); 14 usan además un dataset propio |
| Familias (automático) | CNN en casi todos; transferencia desde ImageNet en 24; atención en 8; metaheurísticas en 8; Transformer en 7; ensambles en 6; híbridos DL+ML en 4; GNN, aprendizaje por refuerzo y LSTM en 2 cada uno; destilación y aprendizaje federado en 1 cada uno |
| Conjunto de validación separado | Solo **24 de 59** lo declaran |
| k-fold | 19 de 59 |
| Repeticiones o semillas | Solo **9 de 59** reportan más de una corrida |
| Pruebas estadísticas | Solo **9 de 59** usan alguna (t, Wilcoxon, McNemar, ANOVA o "p < 0,05") |
| Código público | **2 de 59** oficial (Celik y Shi/MbXception); 3 parcial, de terceros o a petición |
| Baselines ajustados por igual | 12 de 59 lo declaran |

**Mensaje para el informe:** la literatura sobre TrashNet casi nunca reporta varianza entre semillas, pruebas estadísticas ni un conjunto de validación independiente. Eso justifica el protocolo del E2 (8 semillas, partición con auditoría y pruebas pareadas) y explica parte de la brecha entre los valores reportados y los replicados.

### 6.2 Los 7 del benchmark (verificados a mano)

| Modelo | Paper | Dataset | Partición | Validación separada | Repeticiones | Pruebas estadísticas | Código | Accuracy reportada | Citas (OpenAlex) | 2-yr mean citedness de la revista | Cuartil SJR (año de publicación) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Celik-HC (guía) | Celik 2025, *The Visual Computer* | TrashNet; Household_Garbage (Kaggle, 12 clases) | 80/20 aleatoria | **No: la época se elige con el test** | Una | No | Sí, oficial | 99,40 % | 15 | 4,1 | Q2 |
| MbXception | Shi et al. 2020, *IEEE Access* | TrashNet | No reportada en el texto | No declarada | Una | No | Sí, oficial | 94,34 % | 64 | 7,4 | Q1 |
| ViT-B16 | Huang K. et al. 2021, *Sustainability* | TrashNet | 70/13/17 (oficial) | Sí | Una | No | No | 96,98 % | 68 | 7,1 | Q1 en Geography, Planning and Development; Q2 en las demás categorías |
| GFN | Wang Z. et al. 2025, *Electronics* | TrashNet; Garbage Dataset (Kaggle) | 80/10/10 | Sí | Una | No | No | 94,21 % | 20 | 4,9 | Q2 |
| FocusRCNet | Zheng D. et al. 2023, *VCIBA* | TrashNet | 70/30 train/val | **La validación hace de test** | Una | No | No | ~92 % | 21 | 9,9 | Q2 en Computer Vision (Q1 en 2025) |
| ResMsCapsule | Huang L. et al. 2023, *ESPR* | TrashNet (**4 clases**) | 80/20 | No declarada | Media y DE por variante (sin número de corridas) | No | No | 91,41 % | 10 | 5,0 | Q1 en Pollution; Q2 en Environmental Chemistry |
| RWCNet | Hossen et al. 2024, *IEEE Access* | TrashNet | 70/20/10 con 5 pliegues | Sí | 5 pliegues | No | No | 95,01 % | 127 | 7,4 | Q1 |

**Lectura:** cuatro de los siete se publicaron en revistas Q1. El paper guía (Q2) es el de mayor accuracy reportada y el que más se aleja en la réplica.

### 6.3 Acuerdo entre revisores (kappa de Cohen) en el cribado

**Búsqueda original (Excel), revisores humanos:**
- Dos revisores (H1 y H2) decidieron de forma independiente la **muestra común de 25 registros**: Incluir, Excluir o Dudoso. Están en las columnas "Decisión humana (T/R)" y "decision humana 2" del Excel.

| Comparación (n = 25) | Acuerdo observado | κ de Cohen (3 categorías) | κ de Cohen (pasa o se excluye) |
|---|---|---|---|
| **H1 frente a H2** | 0,92 | **0,857** | **1,000** |
| H1 frente a Claude | 0,92 | 0,860 | 0,896 |
| H2 frente a Claude | 0,96 | 0,925 | 0,896 |

- **Interpretación:** acuerdo casi perfecto entre los dos revisores humanos (Landis y Koch). Coinciden en los 25 registros en la decisión que importa para el PRISMA: si pasa o no a texto completo. La única diferencia entre ellos es Incluir frente a Dudoso en 2 registros. La selección preliminar de Claude coincide con ambos revisores en el mismo nivel.
- **Discrepancias y su resolución.** Regla: decide la mayoría de los tres revisores (H1, H2 y Claude); si no hay mayoría, Dudoso, que pasa a texto completo.

| Registro | H1 | H2 | Claude | Decisión final | Efecto |
|---|---|---|---|---|---|
| R100 *Trash Image Classification Using Autoencoder* | Dudoso | Incluir | Incluir | **Incluir** | Ninguno: ya pasaba, con prioridad Media |
| R132 *Real Time Detection and Categorization of Waterborne Trash…* | Dudoso | Incluir | **Excluir** | **Dudoso** | Pasa a texto completo, pero con prioridad Baja (ponencia con menos de 10 citas), así que queda fuera del corte. Los humanos notaron que el resumen describe clasificación con CNN, no detección. |

- En el Excel quedaron:
  - las dos decisiones finales escritas sobre la fórmula (como indica su hoja LÉEME);
  - una columna "Acuerdo humanos (H1 vs H2)";
  - los kappas y el texto de "Quién hizo la selección" en la hoja "PRISMA".

**Búsqueda ampliada (kappa ya calculado):**
- **Revisor 1:** las reglas automáticas de la sección 4.
- **Revisor 2:** cribado manual independiente de título y resumen, hecho por Claude con los mismos criterios, sin ver la decisión de las reglas.
- **Calibración** (`datos/kappa_muestra_calibracion.csv`): 60 registros, semilla 2026.
  - La primera versión de las reglas dio κ = 0,26, porque aceptaba "waste" en cualquier sentido (por ejemplo, "wasted time").
  - Las reglas se corrigieron con esta muestra, así que **no se usa para el kappa reportado**.
- **Validación** (`datos/kappa_muestra_validacion.csv`): 60 registros nuevos, semilla 777.
  - Acuerdo observado: **0,833**.
  - **κ de Cohen: 0,669** (acuerdo sustancial según Landis y Koch).
  - Sensibilidad de las reglas: 0,93. Precisión: 0,77. Las reglas son conservadoras.
- **Declararlo así en el informe:** "segundo revisor asistido por IA".

## 7. Valoración de la calidad metodológica

**Criterios** (columnas `riesgo_fuga`, `repeticiones` y `equidad_baselines` de la hoja):

| Dimensión | Niveles |
|---|---|
| Riesgo de fuga | **Alto:** no hay validación separada, o el test selecciona el modelo. **Medio:** hay validación o k-fold, pero sin control de duplicados ni normalización solo con train. **Bajo:** validación separada y control de duplicados o fuga. |
| Repeticiones | Una corrida, o varias corridas o semillas |
| Equidad de baselines | Declara que los baselines se entrenaron con el mismo protocolo y ajuste, o no lo declara |

**Resultados en los 59 incluidos:**
- **Riesgo de fuga:** alto en 25 y medio en 34. **Ninguno** alcanza riesgo bajo: ninguno controla los casi duplicados de TrashNet. En el E1 se encontraron grupos de casi duplicados en TrashNet, por eso la partición v2 los agrupa.
- **Repeticiones:** 50 de 59 reportan una sola corrida.
- **Equidad de baselines:** no declarada en 47 de 59.
- **Puntaje de calidad (0 a 4)** = riesgo de fuga (bajo 2, medio 1, alto 0) + repeticiones (1 si hay varias) + equidad (1 si se declara): 19 estudios con 0, 29 con 1, 7 con 2 y 4 con 3. **Ninguno llega a 4.**

**Los 7 del benchmark:**

| Modelo | Riesgo de fuga | Repeticiones | Equidad de baselines | Comentario |
|---|---|---|---|---|
| Celik-HC | **Alto** | Una | No declarada | La selección de la época con el test está **confirmada por la réplica: +2,8 puntos de accuracy, IC 95 % [1,4; 4,2]** |
| MbXception | Alto | Una | No declarada | No describe la partición |
| ViT-B16 | Medio | Una | No declarada | Compara con cifras publicadas de otros trabajos |
| GFN | Medio | Una | No declarada | |
| FocusRCNet | **Alto** | Una | No declarada | Sin test independiente: el mismo 30 % selecciona y evalúa |
| ResMsCapsule | Alto | Varias (sin detallar) | Declarada | Usa 4 clases, no comparable con 6 |
| RWCNet | Medio | 5 pliegues | No declarada | Normaliza con la media y la DE de todo el dataset (fuga menor) |

## 8. Búsqueda transversal: tabla de mecanismos candidatos para el E3

Archivo: `datos/tabla_mecanismos.csv`.
- Las fuentes primarias se verificaron contra OpenAlex o arXiv el 16-11-2026: título, año y DOI o ID de arXiv.
- Los ejemplos de transferencia salen de los registros de las cadenas T1–T6 (`datos/transversal/`).
- La transferibilidad se valora frente a la evidencia propia del benchmark.

| ID | Mecanismo | Dónde aparece en el Top 10 | Modalidad de origen | Fuente primaria | Ejemplos de transferencia a otras modalidades (de la búsqueda) | Propiedad estructural compartida con TrashNet | Transferibilidad preliminar |
|---|---|---|---|---|---|---|---|
| M1 | Destilación de conocimiento | FocusRCNet | Compresión de modelos; lenguaje y voz | Hinton et al. 2015, arXiv:1503.02531 | DistilBERT (10.48550/arxiv.1910.01108); TinyBERT (10.18653/v1/2020.findings-emnlp.372); KD + curriculum para keyword spotting **robusto al ruido** (10.1109/access.2023.3314191) | Dataset pequeño, se busca un modelo ligero y existe un maestro robusto (ViT-B16) | **Alta**: destilar la robustez de ViT-B16 hacia una CNN ligera |
| M2 | Entrenamiento y aumentación con ruido o corrupciones | Ninguno (es la limitación) | Voz y audio; imagen médica | Hendrycks y Dietterich 2019, arXiv:1903.12261 | Robustez al ruido en keyword spotting (10.1109/access.2023.3280477); transferencia a ruido no visto (10.1109/icassp40776.2020.9054543); "Pre-trained Under Noise" en fracturas (arXiv:2507.09731) | El mismo ruido gaussiano de sensor que hace colapsar al paper guía | **Alta**: la intervención más directa |
| M3 | Supervisión profunda o salidas auxiliares | RWCNet | Segmentación médica y teledetección | Lee et al. 2015, arXiv:1409.5185; Szegedy et al. 2015, 10.1109/cvpr.2015.7298594 | Detección de cambios en teledetección (10.1016/j.isprsjprs.2020.06.003); vasos retinianos (10.1016/j.ijmedinf.2019.03.015); adaptación entre modalidades médicas (10.1109/tmi.2020.2972701) | Regulariza con pocos datos | **Alta**: está en el mejor modelo |
| M4 | Self-attention o Transformer | ViT-B16, GFN | Lenguaje | Vaswani et al. 2017, arXiv:1706.03762; Dosovitskiy et al. 2021, arXiv:2010.11929 | Traslado de NLP a visión | Contexto global y menor dependencia de texturas de alta frecuencia | **Alta**: el único robusto al ruido (F1 0,894 con σ = 0,05) |
| M5 | Adaptación de dominio adversarial | Ninguno | Teledetección, EEG y fallas | Ganin y Lempitsky 2015, arXiv:1409.7495 | Teledetección agrícola (10.3390/rs14246298); DA-CapsNet en EEG (10.1016/j.knosys.2023.111137); CBAM con DA parcial (10.1016/j.engappai.2023.106674) | Cambio de dominio: fondo blanco de TrashNet frente al fondo real | Media |
| M6 | Atención de canal y espacial (SE, CBAM) y PPM | GFN, FocusRCNet | Visión, luego series y señales | Hu et al. 2018, 10.1109/cvpr.2018.00745; Woo et al. 2018, 10.1007/978-3-030-01234-2_1; Zhao et al. 2017, 10.1109/cvpr.2017.660 | Desgaste de herramientas (10.1016/j.jmapro.2022.11.017); EEG (10.1109/taffc.2024.3394436) | Separar el objeto del fondo | Media |
| M7 | Redes de cápsulas | ResMsCapsule | Dígitos, luego texto, EEG y fallas | Sabour et al. 2017, arXiv:1710.09829 | Texto (10.18653/v1/d18-1350); fallas multidominio (10.1109/tim.2020.3042300) | Relaciones parte-todo en objetos deformados | Media (mala calibración y frágil al ruido) |
| M8 | Fusión multinivel (hypercolumn) | Celik-HC | Segmentación y localización fina | Hariharan et al. 2015, 10.1109/cvpr.2015.7298642 | Teledetección (10.1016/j.inffus.2018.09.006); tumores cerebrales (10.1109/access.2024.3480271); deslizamientos (10.1007/s10346-022-01847-1) | Texturas finas de materiales | **Baja**: la ablación muestra que no aporta (+0,005, n.s.) |
| M9 | Gradient boosting sobre embeddings | Celik-HC | Datos tabulares | Chen y Guestrin 2016, 10.1145/2939672.2939785 | Sin cadena propia | Clasificador final | **Baja**: empeora frente a softmax (−1,2 a −1,6 puntos) |
| M10 | Convoluciones separables y bloques invertidos | MbXception, MobileNetV2 | Visión eficiente | Chollet 2017, 10.1109/cvpr.2017.195; Sandler et al. 2018, arXiv:1801.04381 | Base de las redes ligeras | Bajo costo en dispositivos de borde | Media (como alumno en M1) |
| M11 | Conexiones residuales | ResNet-18, GFN, ResMsCapsule | Visión | He et al. 2016, 10.1109/cvpr.2016.90 | Ubicuas | Optimización de redes profundas | Baja (ya están en casi todos los modelos) |

**Lectura para el E3:**
- La limitación cuantificada del paper guía es la **fragilidad ante el ruido** (F1 de 0,914 a 0,255 con σ = 0,05). Es compartida por las CNN del benchmark, no por ViT-B16.
- Los mecanismos con mayor transferibilidad son **M1 + M2 + M4**: destilar un maestro Transformer robusto hacia un alumno ligero entrenado con aumentación de ruido.
- La cadena T3b muestra que esa combinación ya funciona en **voz**: hay destilación robusta al ruido en keyword spotting.
- Esta tabla no compromete la elección final, como dice el enunciado.

## 9. Pendientes

No queda nada obligatorio. En el informe hay que declarar como limitaciones:
- la restricción de la búsqueda a "TrashNet" (sección 5.1);
- el corte de prioridad a texto completo (78 no leídos);
- la selección asistida por IA, verificada por dos revisores humanos con κ = 0,857;
- la lectura a texto completo asistida por IA, con los 7 del benchmark verificados a mano.

**Opcional:** repetir en Scopus las cadenas P1–P3 y T1–T6 con una sesión institucional.

## 10. Archivos

| Archivo | Contenido |
|---|---|
| `datos/registro_cadenas.csv` | Cada cadena por base: consulta exacta, fecha y hora en UTC, N de resultados y registros descargados |
| `datos/registro_cadenas_manual_IEEE_Scopus.csv` | Plantilla para la ejecución manual |
| `datos/cribado_principal.csv` | Los 3434 registros únicos, con su fuente, cadena, DOI, revista, citas, decisión y si están en el corpus |
| `datos/prisma_conteos.json` | Todos los números del diagrama PRISMA |
| `datos/hoja_extraccion.csv` | Hoja de extracción y calidad de los 63 registros del corpus |
| `datos/top7_verificado.json` | Valores verificados a mano de los 7 del benchmark |
| `datos/seleccion_PRISMA_TrashNet_busqueda_original.xlsx` | **Búsqueda original:** bitácora con cadenas, fechas y conteos; 168 registros con las decisiones de Claude y de los dos revisores, y las discrepancias resueltas; PRISMA con los kappas; calidad; síntesis; frecuencia de modelos |
| `datos/cuartiles_sjr_top7.csv` | Cuartiles SJR de las revistas de los 7 del benchmark |
| `capturas/` | Captura de cada cadena en IEEE Xplore |
| `datos/kappa_muestra_*.csv` | Muestras de calibración y validación, con la decisión de las reglas y la manual |
| `datos/tabla_mecanismos.csv` | Tabla de mecanismos candidatos |
| `datos/transversal/` | Registros de las cadenas T1–T6 |
| `respuestas_crudas/` | Respuestas originales de las APIs (XML de arXiv y JSON de OpenAlex) |
| `scripts/` | Los scripts del proceso, en orden de uso: `busqueda.py` (cadenas), `corpus.py` (lectura del RDF de Zotero y datos de OpenAlex), `cribado.py` (deduplicación, cribado y cruce con el corpus) y `extraccion.py` (hoja de extracción) |

Para regenerar todo, ejecutar los scripts en ese orden. Los conteos pueden cambiar si se ejecutan otro día, porque las bases crecen; por eso se guardan las respuestas crudas con su fecha.
