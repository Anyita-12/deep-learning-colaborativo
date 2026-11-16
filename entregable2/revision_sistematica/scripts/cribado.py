import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

BUSQ, CORPUS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

RE_WASTE = re.compile(r"\b(waste|garbage|trash|litter|recycl\w*|rubbish|refuse sorting|municipal solid)\b", re.I)
RE_IMAGE = re.compile(r"\b(image\w*|visual|vision|camera\w*|photo\w*|picture\w*|video\w*)\b", re.I)
RE_TASK = re.compile(r"\b(classif\w*|sort\w*|recogni\w*|identif\w*|categori\w*)\b", re.I)
RE_DL = re.compile(r"\b(deep|convolutional|cnns?|neural|transformer\w*|vit|resnet\w*|mobilenet\w*|efficientnet\w*|"
                   r"densenet\w*|vgg\w*|inception\w*|xception|yolo\w*|capsule\w*|attention|transfer learning|federated)\b", re.I)
RE_NONIMAGE = re.compile(r"\b(sensors?|spectr\w*|time series|forecast\w*|tabular|questionnaires?|acoustic|sound|audio|"
                         r"textual|generation rate|amount of waste|gis|routing|scheduling)\b", re.I)
RE_IMPLICA_IMAGEN = re.compile(r"\b(trashnet|taco|convolutional|cnns?|vit|yolo\w*|pixels?)\b", re.I)
W = r"(?:waste|garbage|trash|litter|rubbish|recyclables?|recycling|recyclable \w+)"
T = r"(?:classif\w*|sort\w*|recogni\w*|identif\w*|categori\w*|segregat\w*|separat\w*)"
RE_WASTE_FRASE = re.compile(rf"\b{W}\s+(?:\w+\s+)?(?:{T}|images?|dataset\w*|bins?|items?|objects?|materials?|types?|categor\w*)\b"
                            rf"|\b{T}\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+)?{W}\b|\btrashnet\b", re.I)
RE_WASTE_NO = re.compile(r"\b(food waste|e-waste|electronic waste|weee|wastewater|waste water|construction and demolition|"
                         r"battery|batteries|incinerat\w*)\b", re.I)
RE_DETECCION = re.compile(r"\b(detect\w*|segment\w*|count\w*|localiz\w*|tracking)\b", re.I)
RE_REVIEW = re.compile(r"\b(review|survey|overview|bibliometric|state of the art)\b", re.I)
TIPOS_OK = {"article", "preprint", "conference-paper", "book-chapter", "review", None, "letter"}


def norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def decide(r):
    texto = f"{r['titulo']} {r.get('resumen') or ''}"
    if r.get("tipo") not in TIPOS_OK:
        return "X0 tipo de documento no elegible (libro, capítulo, tesis, dataset u otro)"
    if r.get("tipo") == "review" or RE_REVIEW.search(r["titulo"]):
        return "X1 revisión o encuesta, no estudio primario"
    if not (RE_WASTE.search(r["titulo"]) or RE_WASTE_FRASE.search(texto)):
        return "X2 fuera del dominio de residuos"
    if RE_WASTE_NO.search(r["titulo"]):
        return "X2 fuera del dominio de residuos"
    if RE_DETECCION.search(r["titulo"]) and not RE_TASK.search(r["titulo"]):
        return "X4 tarea distinta de clasificación (detección, segmentación, conteo, predicción)"
    solo_titulo = not r.get("resumen")
    if solo_titulo:
        return "INCLUIDO" if RE_TASK.search(texto) else "X4 tarea distinta de clasificación (detección, segmentación, conteo, predicción)"
    if not RE_IMAGE.search(texto) and not RE_IMPLICA_IMAGEN.search(texto) and RE_NONIMAGE.search(texto):
        return "X3 no usa imágenes (sensores, espectros, texto o series)"
    if not RE_TASK.search(texto):
        return "X4 tarea distinta de clasificación (detección, segmentación, conteo, predicción)"
    if not RE_DL.search(texto):
        return "X5 no usa Deep Learning"
    return "INCLUIDO"


def main():
    registros = []
    for cid in ("P1", "P2", "P3"):
        for r in json.loads((BUSQ / f"{cid}_registros.json").read_text(encoding="utf-8")):
            r["cadena"] = cid
            registros.append(r)
    identificados = Counter(r["fuente"] for r in registros)
    unicos, vistos = [], {}
    for r in registros:
        claves = [k for k in (("doi", r["doi"].lower()) if r["doi"] else None, ("t", norm(r["titulo"]))) if k]
        hit = next((vistos[k] for k in claves if k in vistos), None)
        if hit is not None:
            hit["cadenas"].add(r["cadena"])
            hit["fuentes"].add(r["fuente"])
            if not hit.get("resumen") and r.get("resumen"):
                hit["resumen"] = r["resumen"]
            continue
        r = dict(r, cadenas={r["cadena"]}, fuentes={r["fuente"]})
        unicos.append(r)
        for k in claves:
            vistos[k] = r
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    for c in corpus:
        hit = next((r for r in unicos if (c["doi"] and r["doi"].lower() == c["doi"]) or norm(r["titulo"]) == norm(c["titulo"])), None)
        if hit and not hit.get("resumen"):
            hit["resumen"] = c["resumen"]
    for r in unicos:
        r["decision"] = decide(r)
    idx_doi = {r["doi"].lower(): r for r in unicos if r["doi"]}
    idx_t = {norm(r["titulo"]): r for r in unicos}
    for c in corpus:
        hit = idx_doi.get(c["doi"]) or idx_t.get(norm(c["titulo"]))
        c["recuperado_por_busqueda"] = bool(hit)
        c["decision_cribado_automatico"] = hit["decision"] if hit else ""
        c["cadenas"] = ",".join(sorted(hit["cadenas"])) if hit else ""
        if hit:
            hit["en_corpus"] = True
    with open(OUT / "cribado_principal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "cadenas", "anio", "doi", "titulo", "venue", "citas", "decision", "en_corpus"])
        for r in unicos:
            w.writerow([";".join(sorted(r["fuentes"])), ",".join(sorted(r["cadenas"])), r["anio"], r["doi"],
                        r["titulo"], r["venue"], r["citas"], r["decision"], bool(r.get("en_corpus"))])
    (OUT / "corpus_cruce.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    dec = Counter(r["decision"] for r in unicos)
    resumen = {
        "identificados_por_base": dict(identificados),
        "identificados_total": len(registros),
        "duplicados_eliminados": len(registros) - len(unicos),
        "cribados": len(unicos),
        "sin_resumen": sum(1 for r in unicos if not r.get("resumen")),
        "decisiones": dict(sorted(dec.items())),
        "corpus_total": len(corpus),
        "corpus_recuperado": sum(c["recuperado_por_busqueda"] for c in corpus),
        "corpus_recuperado_e_incluido": sum(c["decision_cribado_automatico"] == "INCLUIDO" for c in corpus),
        "corpus_no_recuperado": [c["titulo"] for c in corpus if not c["recuperado_por_busqueda"]],
        "corpus_recuperado_pero_excluido": [(c["titulo"], c["decision_cribado_automatico"]) for c in corpus
                                            if c["recuperado_por_busqueda"] and c["decision_cribado_automatico"] != "INCLUIDO"],
    }
    (OUT / "prisma_principal.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
