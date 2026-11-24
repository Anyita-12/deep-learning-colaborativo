# Análisis del Entregable 3

Lee los resultados versionados (`results/`, `entregable2/results/`, `entregable3/results/`) y produce la tabla maestra única, los contrastes y las figuras del libro y del dashboard. Solo CPU (numpy, pandas, scipy, scikit-learn, matplotlib); unos 3 minutos.

| Paso | Script | Salida |
|---|---|---|
| 1 | `cargar.py` | `maestra.pkl` (una fila por modelo y semilla), `preds.pkl` |
| 2 | `confirmatorio.py` | H1 (F1 y MCC, semillas 0–9), H2: t, Wilcoxon exacta, bootstrap jerárquico, Holm, Cliff, $d_z$, Nadeau–Bengio |
| 3 | `secundario.py` | H3 (TOST), ablación, P1–P5, conjuntos adicionales, Friedman e Iman–Davenport, Nemenyi |
| 4 | `conforme.py` | Predicción conforme dividida por clase |
| 5 | `tablas_y_figuras.py` | `entregable3/resultados/*.csv` y `docs/images/e3/*.png` |
| 6 | `datos_dashboard.py` | `dashboard/data/` (copia idéntica de los CSV y salidas por imagen) |
| 7 | `congelar.py` | SHA-256 de los directorios congelados (capítulo 21) |

`estad.py` contiene las pruebas. El bootstrap usa semilla 2026 y 10 000 réplicas; regenerar desde cero da los mismos archivos byte a byte.
