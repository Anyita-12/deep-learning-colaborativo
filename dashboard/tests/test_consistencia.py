"""El dashboard y el libro usan la misma tabla maestra; las cifras del libro coinciden con ella."""
import hashlib, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "entregable3" / "resultados"
DASH = ROOT / "dashboard" / "data"


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def test_mismos_csv_en_libro_y_dashboard():
    for f in RES.glob("*.csv"):
        assert sha(f) == sha(DASH / f.name), f"{f.name} difiere entre entregable3/resultados y dashboard/data"


def test_cifras_del_capitulo_28_salen_de_la_tabla_maestra():
    m = pd.read_csv(DASH / "tabla_maestra.csv").set_index("modelo")
    txt = (ROOT / "docs" / "28_e3_evaluacion_ampliada.md").read_text(encoding="utf-8")
    fila = re.search(r"\| \*\*NAGFuse\*\* \|(.+)\|", txt).group(1)
    nums = [float(x) for x in re.findall(r"\d\.\d{3}", fila)]
    esperado = [m.loc["NAGFuse", c] for c in ["F1 limpio", "F1 ruido 0.05", "F1 ruido 0.10", "F1 desenfoque",
                                              "F1 oclusión", "F1 fondo", "ECE", "NLL"]]
    assert [round(e, 3) for e in esperado] == nums


def test_cifras_del_capitulo_27_salen_de_los_contrastes():
    h1 = pd.read_csv(DASH / "contrastes_H1_f1.csv").set_index("rival")
    txt = (ROOT / "docs" / "27_e3_protocolo_validacion.md").read_text(encoding="utf-8")
    for rival in ["RWCNet", "ViT-B16", "GFN", "Celik-HC"]:
        assert f"| {rival} | {int(h1.loc[rival, 'R'])} | {h1.loc[rival, 'media_dif']:+.4f} |" in txt
