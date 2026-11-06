# Proyecto Final de Deep Learning - Díaz J., Sanjuan A. 

Entregables encadenados E1–E3. Dataset de trabajo: TrashNet (clasificación de imágenes).

## Estructura
```
data/        datos (NO versionados; ver "Datos")
models/      checkpoints (NO versionados)
notebooks/   exploración (EDA)
src/         código reutilizable (device.py, datos, modelos, entrenamiento)
configs/     YAML de experimentos
results/     tablas pequeñas (csv/json) por corrida
reports/     Jupyter Book
docs/        DECISIONES.md (declaración a priori, uso de IA)
tests/       pruebas mínimas (shapes, no-leakage, semillas)
```

## Instalación
Python recomendado: **3.12** (TensorFlow 2.21 admite 3.10–3.13 *[verificar en PyPI]*).

### Windows, solo CPU
```powershell
git clone <URL> C:\dev\dl-proyecto ; cd C:\dev\dl-proyecto
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/device.py
```
Si PowerShell bloquea el script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### GPU NVIDIA (requiere WSL2/Ubuntu o Linux)
TensorFlow ≥2.11 **no** usa GPU en Windows nativo.
```bash
# En Ubuntu (WSL2), con el repo clonado en ~/dl/ (NO en /mnt/c)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt
python src/device.py     # debe mostrar "device": "GPU"
```

## Datos
Descargar TrashNet y colocarlo en `data/raw/` *(instrucciones exactas: [PENDIENTE])*. Tras crear la partición se guarda `data/SHA256SUMS.txt` del test congelado.

## Reproducibilidad
- Cada corrida registra el dict de `setup_device()` (versiones, hardware, semilla).
- CPU y GPU **no** dan resultados idénticos: las corridas finales de los 3 modelos se hacen en **la misma máquina**; los tiempos no son comparables entre máquinas.
- Test se evalúa una vez por corrida; transformaciones se ajustan solo con train.

## Flujo de trabajo en equipo
Ramas cortas `feature/<tarea>` → PR revisado por la otra persona → merge a `main`. Commits pequeños. No se sube `data/`, `models/` ni `.venv/`.
