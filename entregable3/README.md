# Entregable 3: NAGFuse y evaluación ampliada

Este directorio contiene el modelo propio (NAGFuse), el protocolo preregistrado y el sistema de entrenamiento reanudable para Colab Pro. Reutiliza sin modificar el código congelado del E1 (`src/`) y del E2 (`entregable2/src/`): las semillas nuevas de los rivales usan exactamente el mismo código y los mismos hiperparámetros elegidos.

```
entregable3/
├── e3/
│   ├── nagfuse.py        modelo propio: Haar ortonormal, estimador MAD, contracción, compuerta y mezcla
│   ├── theory.py         Teoremas 1 y 2 y su verificación numérica (numpy)
│   ├── cfg.py            protocolo en código: semillas, cola de tareas, conjuntos adicionales
│   ├── extra_data.py     descarga, mapeo de clases, casi duplicados y caché de los conjuntos adicionales
│   ├── sync.py           persistencia en Drive (copias atómicas, espejo de checkpoints, reservas)
│   ├── worker.py         ejecuta una tarea en un proceso aislado (E1, E2 o NAGFuse)
│   └── orchestrator.py   cola con prioridades, reintentos, reanudación y estado
├── tests/                pytest: teoría (sin GPU) y modelo (CPU, sin pesos preentrenados)
├── notebooks/E3_entrenamiento_colab.ipynb
├── experiments/          estandar.md, protocolo.md, design_log.md, procedencia.md, demostracion.md
└── requirements-train.txt
```

## Garantías ante cortes

| Evento | Qué se pierde | Cómo se retoma |
|---|---|---|
| Error de Python en una corrida | Nada | El orquestador reintenta una vez desde el checkpoint local de la última época |
| Se cierra el navegador, se va la luz o el internet en casa | Nada mientras Colab mantenga la sesión | El entrenamiento corre en la nube; al volver, se ve el registro |
| Colab cierra la máquina virtual | La corrida en curso (o los últimos 25 min si es larga) | Abrir el cuaderno y *Ejecutar todas*: se restaura todo desde Drive |
| Una corrida ya terminada | Nunca | Sus archivos se copian a Drive al terminar, de forma atómica |

Cada corrida termina con los archivos del E2 (`config`, `history`, `val_predictions`, `test_predictions`, `metrics_test`, `calibration`, `robustness`, `gradcam`, `timing`, `uncertainty_mc_dropout`) y además `extra_eval.pkl` (conjuntos adicionales), `nagfuse_analisis.pkl` (compuerta, ρ y ramas) y `E3_HOOK.json`.

## Uso

Ver el cuaderno `notebooks/E3_entrenamiento_colab.ipynb`. Comandos equivalentes:

```bash
cd entregable3
python -m pytest -q tests
python -m e3.orchestrator smoke                 # prueba de humo en carpetas _smoke
python -m e3.orchestrator setup && python -m e3.orchestrator prepare
python -m e3.orchestrator run --worker 0
python -m e3.orchestrator status
```
