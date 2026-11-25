# Protocolo v4: opción B (segundo intento autorizado)

**Estado de este documento.** Se congela en dos momentos, cada uno con su tag de Git y su fecha:

1. `e3-protocolo-v4-piloto`: antes del piloto. Fija los candidatos, el piloto (solo validación), la regla de elección y todo el análisis confirmatorio.
2. `e3-protocolo-v4`: después del piloto y antes de la primera corrida confirmatoria. Solo añade el candidato elegido (sección 9) y el registro del piloto en `design_log.md`. Ninguna otra regla puede cambiar entre los dos tags.

Los protocolos v1 a v3 y todos los resultados de la versión A (NAGFuse) quedan **intactos y se siguen reportando**. Este documento no los sustituye.

| Campo | Valor (completar antes del tag `e3-protocolo-v4-piloto`) |
|---|---|
| Fecha y hora del tag del piloto | `2026-11-24 18:30 (America/Bogota)` |
| Autorización del docente | Dr. Lihki Rubio, 2026-11-24, correo. Tras conocer los resultados de la versión A, indicó que revisáramos los criterios de la guía, recomendó explorar combinaciones de modelos o metodologías y dejó a criterio del equipo la estrategia de contar con planes alternativos (A, B, C). |
| Plazo de entrega acordado | `2026-11-27 18:00` |
| Responsable de la ejecución | Anyely Sanjuan Benavides (Colab y Drive del equipo) |

## 1. Motivo

La versión A cumplió R1 y R2 pero no R3: en F1 macro limpio supera por más de δ a 9 de los 12 rivales, pero empata con RWCNet (−0.0004) y ViT-B16 (+0.0002), y frente a GFN la ventaja (+0.0125) es menor que δ. Ese resultado se reportó tal cual (capítulos 27 y 30). El docente autorizó un segundo intento (opción B) con declaración completa de lo ocurrido.

## 2. Exposición declarada

- **El conjunto de prueba ya se usó** para evaluar la versión A. Los resultados de A en prueba eran conocidos al diseñar B.
- Para limitar el sesgo: (a) B **no** se ajusta con el conjunto de prueba ni con los conjuntos adicionales; (b) los candidatos se fijan aquí, antes de entrenar, con un argumento a priori; (c) la elección entre candidatos usa **solo validación** (piloto, sección 4); (d) la decisión confirmatoria usa **α_B = 0.025** en lugar de 0.05 (Bonferroni por dos intentos sobre la misma familia y el mismo conjunto de prueba).
- Aun así, B es un análisis decidido después de ver resultados de prueba. El informe lo dice explícitamente y deja al docente la valoración de R3 bajo esta autorización.

## 3. Qué cambia y qué no

**Cambia solo la red de cada rama.** El mecanismo es idéntico al de A: estimador MAD sobre HH₁ de Haar, contracción suave con umbral universal en la rama convolucional, mezcla en el espacio de probabilidades con π₀ aprendido, compuerta analítica g(σ̂) con ρ = 2 y s_lo = 0.014535 (el mismo archivo `nagfuse_noise_stats.json`), pérdida NLL de la mezcla + 0.5·(CE de cada rama). Los Teoremas 1 y 2 no dependen de las redes de las ramas y siguen valiendo; las pruebas unitarias verifican el Teorema 2 en cada candidato.

**No cambia:** partición v2, preprocesamiento (224 px por redimensionado directo), aumentos, normalización, pesos de clase, optimizador y planificador, número máximo de épocas, parada temprana, AMP, fuente de pesos preentrenados (ImageNet-1K de torchvision, la misma de los rivales), métricas, perturbaciones, conjuntos adicionales, rivales (congelados; no se reentrena ninguno), familia de contrastes, pruebas y δ.

**Hiperparámetros de B:** los mismos que eligió la búsqueda de la versión A (intento 1: lr 1e-4, weight decay 1e-4, dropout 0.4). B no recibe búsqueda propia. Es un presupuesto de ajuste **menor** que el de los rivales (8 intentos cada uno), es decir, en contra del modelo propio; se declara como desviación.

## 4. Candidatos y argumento a priori

La bitácora de desarrollo de A (filas 1 a 6 de `design_log.md`) mostró que fusionar ViT-B16 con cualquier CNN, incluso una más fuerte que EfficientNet-B0 (DenseNet201), no mejora el F1 de validación en más de 0.002. Por eso el cambio con más probabilidad de aportar en limpio es la **rama robusta** (la de mayor peso con ruido y la mitad de la mezcla en limpio), no solo la convolucional. Entre las redes de torchvision con pesos ImageNet-1K y entrada nativa de 224 px se eligen dos más precisas en ImageNet que las ramas de A (Swin-S y ConvNeXt-T). B1 y B2 cuestan menos que A; B3 cuesta más y se incluye para aislar el efecto de la rama convolucional:

| Candidato | Rama robusta | Rama convolucional (con contracción) | Parámetros aprox. | Top-1 ImageNet de las ramas |
|---|---|---|---|---|
| A (referencia) | ViT-B/16 | EfficientNet-B0 | 89.8 M | 81.1 % / 77.7 % |
| **B1** | Swin-S | ConvNeXt-T | ~77 M | 83.2 % / 82.5 % |
| **B2** | Swin-S | EfficientNet-B0 | ~53 M | 83.2 % / 77.7 % |
| **B3** | ViT-B/16 | ConvNeXt-T | ~114 M | 81.1 % / 82.5 % |

B2 y B3 separan el efecto de cada rama. Los tres se implementan en `e3/nagfuse.py` (`VARIANT_KW`).

## 5. Piloto (solo validación)

- Cada candidato se entrena **una vez** con el mismo procedimiento del intento 1 de la búsqueda de A: semilla 100, lr 1e-4, weight decay 1e-4, dropout 0.4, 30 épocas como máximo, misma parada temprana. Así el F1 de validación es directamente comparable con el de A en ese intento: **0.9688**.
- El piloto **no evalúa en prueba** (etapa `hpo` del código: solo validación).
- **Regla de elección (fija):** se elige el candidato con mayor F1 macro de validación; si dos difieren en menos de 0.002, el de menos parámetros. **Se lanza la fase confirmatoria solo si el elegido supera 0.9688.** Si ninguno lo supera, no se lanza y se reporta el piloto como resultado de la opción B.
- Una sola semilla tiene un error de alrededor de ±0.005 en validación; la regla lo asume y se declara.
- Nd de la bitácora pasa de 7 a 10 (los tres candidatos del piloto).

## 6. Fase confirmatoria

- Modelo `NAGFuse-B` = candidato elegido. Semillas 0 a 19, en orden, pareadas con las de los rivales.
- **Regla de corte:** se analizan las semillas terminadas al acabar la cola o en la fecha de la sección 9, lo que ocurra primero. **Mínimo 12 semillas** (0 a 11); con menos no hay análisis confirmatorio y se reporta como exploratorio.
- Evaluación: la misma de A (prueba, perturbaciones, MC dropout, calibración, tres conjuntos adicionales y análisis del mecanismo).

## 7. Análisis confirmatorio (fijado en `entregable3/analisis/analisis_b.py`)

- **H1-B (decide R3):** NAGFuse-B frente a cada uno de los 12 rivales en F1 macro limpio. Tres pruebas (t pareada, Wilcoxon exacta, bootstrap jerárquico con 10 000 réplicas y semilla 2026), Holm con m = 12 por prueba y decisión por el máximo p ajustado. **Supera** = p máx. < α_B = 0.025, diferencia media > δ = 0.02 y límite inferior del IC 95 % > 0. R3 se cumple solo si supera a los 12.
- Se informa también el veredicto con α = 0.05, como dato secundario.
- **Secundarios, con las mismas reglas que en A:** H1 con MCC y con semillas 0–9 (sensibilidad), H2-B (ruido σ = 0.05), H3-B (TOST frente al mejor rival), conjuntos adicionales, Friedman + Iman-Davenport + Nemenyi con B en lugar de A.
- **Exploratorios:** B frente a A; compuerta apagada frente a encendida con ruido 0.05 en B.

## 8. Reporte

- Se reporta **cualquier resultado** de B, también si es peor que A.
- El libro conserva A como resultado del protocolo v1–v3 y añade B en un capítulo propio, con este documento, el piloto y la exposición de la sección 2.
- La tabla maestra incluye a NAGFuse-B como una fila más; los contrastes de A no cambian.

## 9. Elección tras el piloto (completar antes del tag `e3-protocolo-v4`)

| Campo | Valor |
|---|---|
| F1 val B1 / B2 / B3 | 0.9801 / 0.9851 / 0.9703 |
| Elegido y motivo según la regla | NAGFuse-B2: mayor F1 de validación; la diferencia con B1 (0.0050) supera el umbral de empate de 0.002. |
| ¿Supera 0.9688? ¿Se lanza la fase confirmatoria? | Sí (0.9851 > 0.9688). Se lanza. |
| Fecha y hora del tag `e3-protocolo-v4` | 2026-11-24 20:45 (America/Bogota) |
| Fecha y hora de corte de la fase confirmatoria | 2026-11-25 06:45 (America/Bogota) |
