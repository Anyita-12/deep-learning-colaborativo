import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "crudo").mkdir(exist_ok=True)
UA = "Mozilla/5.0 (academic systematic review script)"
DESDE, HASTA = "2018-01-01", "2026-12-31"

WASTE = ['"waste"', '"garbage"', '"trash"', '"litter"', '"recyclable"']
CADENAS = {
    "P1": {
        "grupos": [WASTE,
                   ['"classification"', '"sorting"', '"recognition"'],
                   ['"deep learning"', '"convolutional neural network"', '"CNN"', '"vision transformer"', '"neural network"'],
                   ['"image"', '"images"', '"vision"']],
        "tipo": "principal"},
    "P2": {"grupos": [['"TrashNet"']], "tipo": "principal"},
    "P3": {
        "grupos": [['"waste"', '"garbage"', '"trash"'],
                   ['"transfer learning"', '"fine-tuning"', '"pretrained"', '"pre-trained"'],
                   ['"image classification"']],
        "tipo": "principal"},
    "T1": {
        "grupos": [['"hypercolumn"', '"multi-level feature fusion"', '"multi-layer feature fusion"', '"multi-scale feature fusion"'],
                   ['"medical image"', '"remote sensing"', '"speech"', '"time series"'],
                   ['"transfer"', '"adaptation"']],
        "tipo": "transversal"},
    "T2": {
        "grupos": [['"knowledge distillation"'],
                   ['"speech"', '"natural language"', '"time series"', '"sensor"', '"EEG"'],
                   ['"transfer"', '"adaptation"']],
        "tipo": "transversal"},
    "T3": {
        "grupos": [['"noise robustness"', '"robustness to noise"', '"corruption robustness"', '"gaussian noise"'],
                   ['"convolutional neural network"', '"vision transformer"'],
                   ['"speech"', '"audio"', '"medical image"', '"remote sensing"'],
                   ['"adaptation"', '"transfer"']],
        "tipo": "transversal"},
    "T3b": {
        "grupos": [['"noise robustness"', '"robustness to noise"', '"corruption robustness"', '"noise augmentation"', '"noisy images"'],
                   ['"deep learning"', '"neural network"'],
                   ['"speech"', '"audio"', '"medical image"', '"remote sensing"'],
                   ['"adaptation"', '"transfer"']],
        "tipo": "transversal"},
    "T4": {
        "grupos": [['"capsule network"', '"capsule networks"'],
                   ['"text"', '"signal"', '"EEG"', '"time series"', '"medical image"'],
                   ['"transfer"', '"adaptation"']],
        "tipo": "transversal"},
    "T5": {
        "grupos": [['"deep supervision"', '"auxiliary loss"', '"auxiliary classifier"'],
                   ['"segmentation"', '"speech"', '"natural language"'],
                   ['"transfer"', '"adaptation"']],
        "tipo": "transversal"},
    "T6": {
        "grupos": [['"channel attention"', '"spatial attention"', '"CBAM"'],
                   ['"time series"', '"signal"', '"speech"', '"text"'],
                   ['"transfer"', '"adaptation"']],
        "tipo": "transversal"},
}


def generica(grupos):
    return " AND ".join("(" + " OR ".join(g) + ")" if len(g) > 1 else g[0] for g in grupos)


def arxiv_query(grupos):
    def campo(t):
        return f'abs:{t}'
    partes = []
    for g in grupos:
        terms = [campo(t) for t in g]
        partes.append("(" + " OR ".join(terms) + ")" if len(terms) > 1 else terms[0])
    return " AND ".join(partes)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def ahora():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_arxiv(cid, grupos, n=100):
    q = arxiv_query(grupos)
    fecha = f"submittedDate:[{DESDE.replace('-', '')}0000 TO {HASTA.replace('-', '')}2359]"
    full = f"({q}) AND {fecha}"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": full, "start": 0, "max_results": n, "sortBy": "relevance"})
    t = ahora()
    raw = get(url)
    (OUT / "crudo" / f"{cid}_arxiv.xml").write_bytes(raw)
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    root = ET.fromstring(raw)
    total = int(root.find("o:totalResults", ns).text)
    recs = []
    for e in root.findall("a:entry", ns):
        recs.append({"fuente": "arXiv", "id": e.find("a:id", ns).text.strip(),
                     "titulo": " ".join(e.find("a:title", ns).text.split()),
                     "anio": e.find("a:published", ns).text[:4],
                     "doi": "", "citas": "", "venue": "arXiv",
                     "resumen": " ".join(e.find("a:summary", ns).text.split())})
    time.sleep(3.5)
    return full, t, total, recs


def oa_abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def run_openalex(cid, grupos, paginas=1):
    q = generica(grupos)
    filtro = f"title_and_abstract.search:{q},from_publication_date:{DESDE},to_publication_date:{HASTA}"
    recs, total, t = [], None, ahora()
    for p in range(paginas):
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": filtro, "per_page": 200, "page": p + 1, "sort": "cited_by_count:desc",
             "select": "id,doi,display_name,publication_year,cited_by_count,primary_location,abstract_inverted_index,type"})
        raw = get(url)
        (OUT / "crudo" / f"{cid}_openalex_p{p}.json").write_bytes(raw)
        d = json.loads(raw)
        total = d["meta"]["count"]
        for w in d["results"]:
            src = ((w.get("primary_location") or {}).get("source") or {})
            recs.append({"fuente": "OpenAlex", "id": w["id"], "titulo": w.get("display_name") or "",
                         "anio": w.get("publication_year"), "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                         "citas": w.get("cited_by_count"), "venue": src.get("display_name") or "",
                         "tipo": w.get("type"), "resumen": oa_abstract(w.get("abstract_inverted_index"))})
        if len(recs) >= total or not d["results"]:
            break
        time.sleep(1)
    return filtro, t, total, recs


def main():
    paginas = {"P1": 20, "P2": 2, "P3": 5}
    registro = []
    solo = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CADENAS)
    for cid, c in CADENAS.items():
        if cid not in solo:
            continue
        g = c["grupos"]
        qa, ta, na, ra = run_arxiv(cid, g)
        qo, to, no, ro = run_openalex(cid, g, paginas.get(cid, 1))
        registro.append({"id": cid, "tipo": c["tipo"], "base": "arXiv (API)", "cadena_generica": generica(g),
                         "consulta_exacta": qa, "fecha_hora": ta, "n_resultados": na, "registros_descargados": len(ra)})
        registro.append({"id": cid, "tipo": c["tipo"], "base": "OpenAlex (API)", "cadena_generica": generica(g),
                         "consulta_exacta": qo, "fecha_hora": to, "n_resultados": no, "registros_descargados": len(ro)})
        with open(OUT / f"{cid}_registros.json", "w", encoding="utf-8") as f:
            json.dump(ra + ro, f, ensure_ascii=False)
        print(cid, "arXiv", na, "OpenAlex", no, "descargados", len(ra) + len(ro), flush=True)
    with open(OUT / ("registro_cadenas.csv" if len(solo) == len(CADENAS) else "registro_cadenas_" + "_".join(solo) + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registro[0]))
        w.writeheader()
        w.writerows(registro)


if __name__ == "__main__":
    main()
