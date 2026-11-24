# 31. Reproducibilidad, autoría y uso de herramientas

## 31.1 Cómo reproducir

| Paso | Comando o archivo | Tiempo |
|---|---|---|
| Entorno de entrenamiento | `entregable3/requirements-train.txt` (PyTorch 2.14.1, pandas 3.1.1) | — |
| Pruebas | `cd entregable3 && python -m pytest -q tests` | < 1 min (CPU) |
| Entrenamiento completo | `entregable3/notebooks/E3_entrenamiento_colab.ipynb` con `REF = "e3-protocolo-v3"` | ≈ 19 h en A100 |
| Análisis desde los resultados versionados | `cd entregable3/analisis && python cargar.py && python confirmatorio.py && python secundario.py && python conforme.py && python tablas_y_figuras.py && python datos_dashboard.py && python congelar.py` | ≈ 3 min (CPU) |
| Dashboard | `cd dashboard && pip install -r requirements.txt && python app.py` | — |
| Contenedor | `docker build -t nagfuse . && docker run -p 8050:8050 nagfuse` | — |

**Determinismo.** `cudnn.deterministic = True`, `benchmark = False`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Dos operaciones no tienen implementación determinista en CUDA (la mediana con índices y el *backward* de la atención con Flash Attention); se aceptan con advertencia. Por eso la reproducción de una corrida puede diferir en el último decimal.

**Resultados versionados.** Las 232 corridas (sin pesos) están en `results/`, `entregable2/results/` y `entregable3/results/`, con los hashes del capítulo 21. Los pesos de NAGFuse (semillas 0–2) y de cada rival (semilla 8), y los modelos exportados, están fuera del repositorio por su tamaño: Zenodo, DOI en `CITATION.cff`.

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
- redactar borradores de los capítulos.

Las decisiones de diseño, el preregistro, la ejecución de los experimentos y la revisión final del texto y del código son responsabilidad de los autores. Toda cifra del informe sale de código versionado y ejecutable. 

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
