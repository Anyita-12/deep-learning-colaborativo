"""Registros de la búsqueda ampliada que mencionan TrashNet y no están en el corpus de la síntesis.

Uso:
    python entregable2/revision_sistematica/scripts/fuera_corpus.py <lista_de_fuentes_Scopus.xlsx>

1. Toma de datos/cribado_principal.csv los registros que pasan el cribado, aparecen con la cadena P2
   ("TrashNet") y no están en el corpus.
2. Recupera de respuestas_crudas/ el título, el resumen (índice invertido de OpenAlex) y la fuente.
3. Clasifica la fuente como indexada en Scopus si su ISSN figura en la lista de fuentes de Scopus con
   cobertura en el año de publicación, o, en los congresos, si el nombre central de la serie aparece completo
   en el título de unas actas del mismo año (o coincide con el acrónimo). La lista es la que publica
   Elsevier (ext_list_Oct_2026.xlsx, hoja de actas hasta junio de 2026); no se versiona por su tamaño.
4. Aplica al título y al resumen las reglas de la brecha de la sección 10.9, con los mismos patrones de
   extraccion.py: cuatro o más familias, más de una corrida o semilla y pruebas estadísticas.
Guarda datos/trashnet_fuera_corpus.csv.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

# Patrones de scripts/extraccion.py (mismas reglas que la hoja de extracción)
FAMILIAS = [
    ("Transformer", r"\b(vision transformer|vit\b|swin|transformer)"),
    ("Cápsulas", r"\bcapsule"),
    ("GNN", r"\bgraph (neural|convolutional)"),
    ("Recurrente (LSTM/GRU)", r"\b(lstm|gru|recurrent)"),
    ("Detector (YOLO/R-CNN)", r"\b(yolo\w*|faster r-?cnn|mask r-?cnn|ssd\b)"),
    ("Destilación", r"\bknowledge distillation|\bdistill"),
    ("Ensamble", r"\bensembl"),
    ("Híbrido DL+ML (SVM/XGBoost/ELM/RF)", r"\b(xgboost|svm|support vector|extreme learning machine|elm\b|random forest|lightgbm|catboost)"),
    ("Metaheurística", r"\b(metaheuristic|meta-heuristic|genetic algorithm|whale|hummingbird|coot|capuchin|particle swarm|pso\b|grey wolf|optimization algorithm)"),
    ("Atención (SE/CBAM/SimAM)", r"\b(attention|squeeze-and-excitation|cbam|simam)"),
    ("Aprendizaje por refuerzo", r"\breinforcement learning"),
    ("Federado", r"\bfederated"),
    ("CNN preentrenada (transfer)", r"\b(transfer learning|pre-?trained|fine-?tun\w*|imagenet)"),
    ("CNN", r"\b(cnn|convolutional|resnet|densenet|mobilenet|efficientnet|vgg|xception|inception|alexnet|shufflenet|nasnet)"),
]
RE_REPS = re.compile(r"\b(random seeds?|different seeds|independent (runs|trials|experiments)|repeated (\w+ )?(times|runs)|"
                     r"(\d+|five|ten|three) (independent )?(runs|trials|repetitions)|averaged over (\d+|five|ten) |mean\s*(±|\+/-)|\d\s*±\s*\d)", re.I)
RE_STATS = re.compile(r"\b(t-?test|wilcoxon|anova|friedman|mcnemar|kruskal|nemenyi|mann-whitney|statistically significant|"
                      r"statistical significance|p\s*[<=]\s*0\.0\d)", re.I)


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def issn(s) -> str:
    return re.sub(r"[^0-9X]", "", str(s or "").upper())


def resumen(inv) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def cubre(cobertura, anio) -> bool:
    for a, b in re.findall(r"(\d{4})\s*-\s*(\d{4}|ongoing|present)?", str(cobertura or "")):
        fin = 9999 if not b or not b.isdigit() else int(b)
        if int(a) <= anio <= fin:
            return True
    return False


def leer_scopus(path: Path):
    wb = openpyxl.load_workbook(path, read_only=True)
    seriales = {}
    for hoja, ci, ce, cc in [("Scopus Sources", 2, 3, 5), ("Serial Conf. Proc. with Profile", 2, 3, 5)]:
        ws = next(w for w in wb.worksheets if w.title.startswith(hoja))
        for r in ws.iter_rows(min_row=2, values_only=True):
            for k in (r[ci], r[ce]):
                if k:
                    seriales.setdefault(issn(k), []).append(r[cc])
    ws = next(w for w in wb.worksheets if w.title.startswith("All Conf. Proceedings"))
    actas = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[3]:
            continue
        anio = int(str(r[3])[:4])
        for t in (r[0], r[1]):
            if t:
                actas.setdefault(anio, set()).add(" " + sin_vacias(t) + " ")
    return seriales, actas


VACIAS = {"ieee", "the", "of", "on", "and", "in", "for", "proceedings", "annual"}


def quitar_numeros(s: str) -> str:
    return " ".join(t for t in s.split() if not re.fullmatch(r"\d+(st|nd|rd|th)?", t))


def sin_vacias(s) -> str:
    return " ".join(t for t in quitar_numeros(norm(s)).split() if t not in VACIAS)


def nucleo(nombre: str) -> str:
    """Nombre de la serie del congreso sin acrónimo, años, ordinales ni palabras vacías."""
    return sin_vacias(re.sub(r"\([^)]*\)", " ", str(nombre or "")))


def en_actas(nombre: str, anio: int, actas: dict) -> bool:
    """Coincide si el nombre central aparece completo y seguido en el título de unas actas del mismo año,
    o si coincide el 90 % de sus palabras y además el acrónimo entre paréntesis."""
    nuc = nucleo(nombre)
    if len(nuc.split()) < 3:
        return False
    acr = re.findall(r"\(([^)]*)\)", str(nombre or ""))
    acr = re.sub(r"[^a-z0-9]", "", acr[-1].lower()) if acr else ""
    for titulo in actas.get(anio, ()):
        if f" {nuc} " in titulo:
            return True
        tokens = set(titulo.split())
        if acr and len(acr) >= 3 and acr in titulo.replace(" ", "") and \
                sum(t in tokens for t in nuc.split()) / len(nuc.split()) >= 0.9:
            return True
    return False


def main(lista: Path):
    c = pd.read_csv(BASE / "datos" / "cribado_principal.csv", encoding="utf-8-sig")
    x = c[(c.decision == "INCLUIDO") & c.cadenas.str.contains("P2") & ~c.en_corpus].copy()
    crudos = {}
    for f in glob.glob(str(BASE / "respuestas_crudas" / "P*_openalex_p*.json")):
        for w in json.load(open(f, encoding="utf-8"))["results"]:
            d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            crudos.setdefault(d or norm(w.get("display_name")), w)
    seriales, actas = leer_scopus(lista)
    filas = []
    for _, r in x.iterrows():
        w = crudos.get(str(r.doi).lower()) if isinstance(r.doi, str) else None
        w = w or crudos.get(norm(r.titulo), {})
        src = (w.get("primary_location") or {}).get("source") or {}
        anio = int(r.anio)
        issns = [issn(i) for i in (src.get("issn") or [])] + ([issn(src.get("issn_l"))] if src.get("issn_l") else [])
        por_issn = any(cubre(cov, anio) for i in issns for cov in seriales.get(i, []))
        es_arxiv = "arxiv" in str(r.fuente).lower() or str(r.venue).lower() == "arxiv"
        tipo_fuente = src.get("type") or ("repository" if es_arxiv else "sin fuente")
        por_actas = (not por_issn and tipo_fuente in ("conference", "book series", "sin fuente")
                     and en_actas(src.get("display_name") or r.venue, anio, actas))
        if tipo_fuente == "repository":
            indexada = "Repositorio de preprints"
        elif tipo_fuente == "sin fuente" and not por_actas:
            indexada = "Sin fuente identificada"
        elif por_issn or por_actas:
            indexada = "Indexada en Scopus"
        else:
            indexada = "No consta en Scopus"
        texto = f"{r.titulo} {resumen(w.get('abstract_inverted_index'))}".lower()
        fams = [n for n, rx in FAMILIAS if re.search(rx, texto)]
        reps, stats = bool(RE_REPS.search(texto)), bool(RE_STATS.search(texto))
        filas.append({"doi": r.doi, "anio": anio, "titulo": r.titulo, "fuente": src.get("display_name") or r.venue,
                      "tipo_documento": w.get("type"), "tipo_fuente": tipo_fuente, "issn": "; ".join(sorted(set(issns))),
                      "indexacion": indexada, "via": "ISSN" if por_issn else ("actas" if por_actas else ""),
                      "tiene_resumen": bool(w.get("abstract_inverted_index")), "familias": "; ".join(fams),
                      "n_familias": len(fams), "cuatro_familias": len(fams) >= 4, "varias_corridas": reps,
                      "pruebas_estadisticas": stats, "cumple_tres": len(fams) >= 4 and reps and stats})
    out = pd.DataFrame(filas)
    out.to_csv(BASE / "datos" / "trashnet_fuera_corpus.csv", index=False, encoding="utf-8-sig")
    print(len(out), "registros")
    print(pd.crosstab(out.tipo_fuente, out.indexacion, margins=True))
    print("con resumen:", int(out.tiene_resumen.sum()))
    print(out[["cuatro_familias", "varias_corridas", "pruebas_estadisticas", "cumple_tres"]].sum())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
