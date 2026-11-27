# 31. Reproducibilidad, autoría y uso de herramientas

## 31.1 Cómo reproducir

| Paso | Comando o archivo | Tiempo |
|---|---|---|
| Entorno de entrenamiento | `entregable3/requirements-train.txt` (PyTorch 2.14.1, pandas 3.1.1) | — |
| Pruebas | `cd entregable3 && python -m pytest -q tests` | < 1 min (CPU) |
| Entrenamiento completo | `entregable3/notebooks/E3_entrenamiento_colab.ipynb` con `REF = "e3-protocolo-v3"` | ≈ 19 h en A100 |
| Opción B: piloto y fase confirmatoria | `entregable3/notebooks/E3_entrenamiento_B_colab.ipynb` con `FASE = "piloto"` y luego `FASE = "confirmatoria"` | 0.4 h en A100 (piloto); 7.9 h en A100 y T4 (20 semillas) |
| Análisis desde los resultados versionados | `cd entregable3/analisis && python cargar.py && python confirmatorio.py && python secundario.py && python conforme.py && python tablas_y_figuras.py && python datos_dashboard.py && python congelar.py`; para la opción B, `python analisis_b.py` después de `cargar.py` | ≈ 3 min (CPU) |
| Dashboard | `cd dashboard && pip install -r requirements.txt && python app.py` | — |
| Contenedor | `docker build -t nagfuse . && docker run -p 8050:8050 nagfuse` | — |

**Determinismo.** `cudnn.deterministic = True`, `benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Dos operaciones no tienen implementación determinista en CUDA (la mediana con índices y el *backward* de la atención con Flash Attention); se aceptan con advertencia. Por eso la reproducción de una corrida puede diferir en el último decimal.

**Resultados versionados.** Las 232 corridas de la versión A y de los rivales, y las 23 de la opción B (3 de piloto y 20 confirmatorias), están sin pesos en `results/`, `entregable2/results/` y `entregable3/results/`, con los hashes de los capítulos 21 y 32. Los pesos de NAGFuse-B no están en el repositorio ni en Zenodo. Los pesos de NAGFuse (semillas 0–2) y de cada rival (semilla 8), y los modelos exportados, están fuera del repositorio por su tamaño: Zenodo, DOI en `CITATION.cff`. Herramienta Render: https://nagfuse-trashnet.onrender.com
**Prueba de humo del despliegue (anexo).** Ejecutada el 23-11-2026 a las 22:44 (hora de Bogotá) con `python dashboard/tests/prueba_humo.py https://nagfuse-trashnet.onrender.com`. La primera parte construye cada página y ejecuta cada callback en el equipo local; la segunda consulta cada ruta del servicio público en Render.

```text
Prueba de humo · 2026-11-23 22:44
página /                                      OK              6 ms
página /eda                                   OK           1589 ms
página /modelos                               OK            360 ms
página /ficha                                 OK              0 ms
tabla y curva de ruido                        OK             59 ms
curvas de entrenamiento                       OK            112 ms
diagnóstico CNN                               OK            167 ms
diagnóstico Celik-HC                          OK             93 ms
diagnóstico FocusRCNet                        OK             87 ms
diagnóstico GFN                               OK             87 ms
diagnóstico MbXception                        OK            148 ms
diagnóstico MobileNetV2                       OK            102 ms
diagnóstico NAGFuse                           OK             97 ms
diagnóstico RWCNet                            OK            100 ms
diagnóstico ResMsCapsule                      OK            100 ms
diagnóstico ResNet-18                         OK             98 ms
diagnóstico ViT-B16                           OK            122 ms
explorador de imagen de prueba                OK             34 ms
predicción en vivo NAGFuse (ONNX, CPU)        OK           2263 ms
predicción en vivo + oclusión                 OK          16039 ms
GET https://nagfuse-trashnet.onrender.com/                       200       289 ms
GET https://nagfuse-trashnet.onrender.com/eda                    200       243 ms
GET https://nagfuse-trashnet.onrender.com/modelos                200       251 ms
GET https://nagfuse-trashnet.onrender.com/ficha                  200       261 ms
GET https://nagfuse-trashnet.onrender.com/_dash-layout           200       248 ms
```

**Lectura.** Todas las páginas y callbacks respondieron sin error y las cinco rutas públicas devolvieron HTTP 200 en unos 250 ms. La predicción en vivo se ejecutó en el equipo local con el modelo ONNX en CPU (2.3 s por imagen; 16 s con el mapa de oclusión). En el plan gratuito de Render no hay memoria para cargar el modelo de 360 MB (límite de 512 MB), por lo que allí la aplicación muestra el aviso correspondiente y el resto de las funciones opera con resultados precalculados.


**Reproducción independiente.** Pendiente: un tercero debe ejecutar el análisis desde los resultados versionados y reportar la discrepancia máxima frente a `tabla_maestra.csv`.

## 31.2 Artefactos del repositorio

`LICENSE` (MIT), `CITATION.cff`, `entregable3/MODEL_CARD.md`, `entregable3/DATA_STATEMENT.md`, `Dockerfile` e integración continua (`.github/workflows/e3-tests.yml`).

## 31.3 Declaración de autoría

| Integrante | Contribuciones |
|---|---|
| Anyely Sanjuan Benavides | Me hago responsable de este trabajo |
| Juan David Díaz Fernández | Me hago responsable de este trabajo |

## 31.4 Declaración de uso de herramientas de inteligencia artificial

Durante el Entregable 3 se usó un asistente de IA generativa (Claude, de Anthropic) para:

- redactar y revisar código (entrenamiento reanudable, análisis estadístico, exportación y dashboard);
- proponer la estructura de la demostración;
- buscar literatura;
- redactar borradores de los capítulos;
- en la opción B (capítulo 32): implementar los candidatos y el cuaderno de entrenamiento en Colab, redactar `analisis_b.py`, guiar paso a paso la ejecución en Colab y en GitHub Desktop, revisar la consistencia de los resultados y redactar el borrador del capítulo 32.

Las decisiones de diseño, el preregistro, la elección de la opción B, la ejecución de los experimentos y la revisión final del texto y del código son responsabilidad de los autores. Toda cifra del informe sale de código versionado y ejecutable. 

## 31.5 Referencias del Entregable 3

- Angelopoulos, A. N., y Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning, 16*(4), 494–591.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.
- Donoho, D. L. (1995). De-noising by soft-thresholding. *IEEE Transactions on Information Theory, 41*(3), 613–627.
- Donoho, D. L., y Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika, 81*(3), 425–455.
- Guo, C., Pleiss, G., Sun, Y., y Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Hendrycks, D., y Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., y Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation, 3*(1), 79–87.
- Kim, B. J., Kawano, M., Iwasawa, Y., y Matsuo, Y. (2025). Unlocking noise-resistant vision: Key architectural secrets for robust models. arXiv:2509.20939.
- Kunwar, S. (2026). Garbage Dataset v2. Kaggle, `sumn2u/garbage-classification-v2`.
- Li, Q., Shen, L., Guo, S., y Lai, Z. (2020). Wavelet integrated CNNs for noise-robust image classification. *CVPR*.
- Massart, P. (1990). The tight constant in the Dvoretzky–Kiefer–Wolfowitz inequality. *Annals of Probability, 18*(3), 1269–1283.
- Nadeau, C., y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics, 15*(6), 657–680.
- Single, S., Iranmanesh, S., y Raad, R. (2023). RealWaste: A novel real-life data set for landfill waste classification using deep learning. *Information, 14*(12), 633.
- Yang, M., y Thung, G. (2016). Classification of trash for recyclability status. CS229 Project Report, Stanford University.
- Yudin, D., et al. (2024). WaRP: Waste recycling plant dataset. Kaggle, `parohod/warp-waste-recycling-plant-dataset`.
