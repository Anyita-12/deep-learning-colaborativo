"""SHA-256 de los directorios congelados (rutas relativas y contenido, en orden). Ejecutar desde cualquier lugar."""
import glob, hashlib, os
from cargar import REPO

def h(d):
    m = hashlib.sha256(); cwd = os.getcwd(); os.chdir(REPO)
    for f in sorted(glob.glob(d + "/**/*", recursive=True)):
        if os.path.isfile(f) and "__pycache__" not in f:
            m.update(f.encode()); m.update(open(f, "rb").read())
    os.chdir(cwd); return m.hexdigest()

if __name__ == "__main__":
    for d in ["results/summary", "entregable2/results/summary", "entregable3/results", "entregable3/resultados"]:
        print(f"| `{d}` | `{h(d)}` |")
