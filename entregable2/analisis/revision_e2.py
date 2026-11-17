"""Cribado del E2 sobre los 62 estudios leídos a texto completo en la revisión del capítulo 3.

Paso 1 (requiere el texto de los PDF, que no se versiona por derechos de autor):
    python entregable2/analisis/revision_e2.py marcar <corpus_text.json>
  cuenta en el texto completo las menciones de TrashNet y de los rasgos que condicionan la elegibilidad
  del E2 y guarda results/revision/indicadores_texto.csv.
Paso 2:
    python entregable2/analisis/revision_e2.py puntuar
  une los indicadores con los metadatos de Zotero (DOI, año, revista), la tabla de síntesis del
  capítulo 3 y las citas de OpenAlex, aplica las reglas de elegibilidad en orden y calcula el puntaje
  de priorización. Guarda results/revision/cribado_e2.csv.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from comun import REPO, RES

REV = RES / "revision"
REV.mkdir(exist_ok=True)
NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "z": "http://www.zotero.org/namespaces/export#",
      "dcterms": "http://purl.org/dc/terms/", "bib": "http://purl.org/net/biblio#",
      "dc": "http://purl.org/dc/elements/1.1/", "link": "http://purl.org/rss/1.0/modules/link/"}

PATRONES = {
    "trashnet": r"trashnet|trash\s*net",
    "seis_clases": r"\b(six|6)[\s-]*(classes|categories|class|types)\b",
    "cuatro_clases": r"\b(four|4)[\s-]*(classes|categories)\b",
    "metaheuristica": r"hummingbird|whale|beluga|coot bird|salp|capuchin|genetic algorithm|particle swarm|grey wolf|"
                      r"arithmetic optimi[sz]ation|differential evolution|metaheuristic|manta ray",
    "refuerzo": r"reinforcement learning",
    "gan": r"generative adversarial",
    "federado": r"federated",
    "deteccion": r"\byolo|faster r-?cnn|object detection",
    "ensamble": r"ensemble",
    "estadistica": r"t-test|wilcoxon|p-value|p ?< ?0\.0|significance test|anova|mcnemar|friedman",
    "ruido_corrupcion": r"gaussian noise|noise robustness|robustness to noise|image corruption|corrupted images|adversarial noise",
    "repeticiones": r"random seeds?|different seeds|independent runs|repeated (\d+|five|ten) times|(five|ten|\d+) (independent )?runs",
}


def marcar(path_json: str):
    corpus = json.load(open(path_json, encoding="utf-8"))
    rows = []
    for zid, d in corpus.items():
        t = d["text"].lower()
        r = {"id_adjunto": int(zid), "paginas": d["pages"]}
        r.update({k: len(re.findall(p, t)) for k, p in PATRONES.items()})
        rows.append(r)
    pd.DataFrame(rows).sort_values("id_adjunto").to_csv(REV / "indicadores_texto.csv", index=False)
    print(len(rows), "estudios marcados")


def metadatos_zotero(rdf_path) -> pd.DataFrame:
    root = ET.parse(rdf_path).getroot()
    items, last = [], None
    for el in root:
        tag = el.tag.split("}")[1]
        if tag in ("Attachment",):
            continue
        ids = [i.text for i in el.iter(f"{{{NS['dc']}}}identifier") if i.text and i.text.startswith("DOI")]
        if tag == "Journal":
            if last is not None and not last["doi"] and ids:
                last["doi"] = ids[0][4:].strip().lower()
            continue
        title = el.find("dc:title", NS)
        att = el.find("link:link", NS)
        if title is None or att is None:
            continue
        subj = [s.text for s in el.findall("dc:subject", NS)]
        journal = el.find("dcterms:isPartOf/bib:Journal/dc:title", NS)
        last = {"id_adjunto": int(att.get(f"{{{NS['rdf']}}}resource").split("_")[1]), "titulo": title.text,
                "anio": int((el.findtext("dc:date", default="0", namespaces=NS) or "0")[:4]),
                "doi": ids[0][4:].strip().lower() if ids else "",
                "id_cap3": next((s for s in subj if s and re.fullmatch(r"R\d{3}", s)), ""),
                "decision_cap3": next((s for s in subj if s in ("Incluir", "Excluir", "Dudoso")), ""),
                "revista": journal.text if journal is not None else "",
                "resumen": " ".join((el.findtext("dcterms:abstract", default="", namespaces=NS) or "").split())}
        items.append(last)
    return pd.DataFrame(items)


def citas_openalex(dois: list[str]) -> dict:
    out = {}
    dois = [d for d in dois if d]
    for i in range(0, len(dois), 40):
        chunk = dois[i:i + 40]
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": "doi:" + "|".join(chunk), "per-page": 50, "select": "doi,cited_by_count"})
        req = urllib.request.Request(url, headers={"User-Agent": "trashnet-benchmark-review/1.0 (academic use)"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        for w in d["results"]:
            out[w["doi"].replace("https://doi.org/", "").lower()] = w["cited_by_count"]
    return out


# Familia del modelo propuesto en cada estudio seleccionado (para la diversidad)
FAMILIA = {"Celik-HC (paper guía)": "Híbrido DL + ML (fusión multinivel + XGBoost)",
           "MbXception": "CNN multirrama", "ViT-B16": "Transformer (ajuste fino)",
           "GFN": "Híbrido CNN-Transformer con atención", "FocusRCNet": "CNN ligera con destilación",
           "ResMsCapsule": "Red de cápsulas", "RWCNet": "Fusión de CNN preentrenadas con salidas auxiliares"}


def puntuar():
    corpus = pd.read_csv(REV / "corpus_62_papers.csv")
    ind = pd.read_csv(REV / "indicadores_texto.csv").drop(columns="paginas")
    meta = metadatos_zotero(REV / "corpus_62_papers_zotero.rdf")
    sint = pd.read_csv(REPO / "results" / "estado_arte" / "tabla_sintesis.csv", encoding="utf-8-sig")
    sint.columns = ["ID", "Referencia", "Problema", "Datasets", "Familias", "Metricas", "acc_trashnet"]
    df = corpus.rename(columns={"id_zotero": "id_adjunto"}).merge(ind, on="id_adjunto", how="left") \
        .merge(meta, on="id_adjunto", how="left").merge(sint[["ID", "Datasets", "acc_trashnet"]],
                                                         left_on="id_cap3", right_on="ID", how="left")
    df["anio"] = df["autores_anio_titulo"].str.extract(r" - (\d{4}) - ").astype(int)
    # Decisiones a texto completo y calidad del capítulo 3 (hojas exportadas de seleccion_PRISMA_TrashNet.xlsx)
    reg = pd.read_csv(REV / "prisma_e1" / "registros.csv")
    reg["doi"] = reg["DOI"].astype(str).str.lower().str.strip()
    cal = pd.read_csv(REV / "prisma_e1" / "calidad.csv").rename(columns={"ID (de Registros)": "ID"})
    reg = reg.merge(cal[["ID", "Usa pruebas estadísticas (Sí/No)", "Reporta dispersión (Sí/No)"]], on="ID", how="left")
    reg = reg.rename(columns={"ID": "id_registro", "Decisión TC (texto completo)": "decision_tc",
                              "Motivo exclusión TC": "motivo_tc", "Usa pruebas estadísticas (Sí/No)": "prueba_estadistica_cap3",
                              "Reporta dispersión (Sí/No)": "dispersion_cap3"})
    df = df.merge(reg[["doi", "id_registro", "decision_tc", "motivo_tc", "prueba_estadistica_cap3", "dispersion_cap3"]]
                  .drop_duplicates("doi"), on="doi", how="left")
    cit = citas_openalex(df["doi"].fillna("").tolist())
    df["citas_openalex"] = df["doi"].map(cit)
    df["fecha_citas"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Reglas de elegibilidad del E2, aplicadas en orden (la primera que se cumple es el motivo)
    def motivo(r):
        if r.decision_tc == "Excluir":
            return f"E0: excluido a texto completo en el capítulo 3 ({r.motivo_tc})"
        if r.trashnet < 3:
            return "E1: TrashNet no es el conjunto principal de evaluación"
        if r.cuatro_clases >= 2 and r.seis_clases == 0:
            return "E2: no evalúa las seis clases de TrashNet"
        contrib = f"{r.titulo} {r.resumen}".lower()  # la contribución se juzga por el título y el resumen
        if re.search(PATRONES["metaheuristica"] + "|" + PATRONES["refuerzo"] + "|" + PATRONES["gan"] + "|"
                     + PATRONES["federado"] + r"|optimi[sz]ation algorithm|swarm", contrib):
            return "E3: depende de metaheurísticas, aprendizaje por refuerzo, GAN o aprendizaje federado"
        if re.search(PATRONES["deteccion"], contrib):
            return "E4: detección de objetos, no clasificación"
        if re.search(r"ensemble", contrib):
            return "E5: ensamble de varias redes (presupuesto de cómputo desigual)"
        return ""
    df["motivo_exclusion_e2"] = df.apply(motivo, axis=1)
    df["elegible_e2"] = df["motivo_exclusion_e2"] == ""

    # Puntaje de priorización (pesos iguales, cada componente en [0, 1])
    edad = (2026 - df["anio"] + 1).clip(lower=1)
    df["citas_por_anio"] = df["citas_openalex"].fillna(0) / edad
    el = df["elegible_e2"]
    df["s_citas"] = np.nan
    df.loc[el, "s_citas"] = df.loc[el, "citas_por_anio"].rank(pct=True)
    df["s_recencia"] = (df["anio"] - 2018) / (2026 - 2018)
    acc = pd.to_numeric(df["acc_trashnet"], errors="coerce")
    df["s_desempeno"] = ((acc - 80) / 20).clip(0, 1).fillna(0.5)
    df["s_implementacion"] = np.where(df["codigo_propio"].notna(), 1.0, 0.5)
    df["puntaje"] = df[["s_citas", "s_recencia", "s_desempeno", "s_implementacion"]].mean(axis=1)
    df.loc[~el, "puntaje"] = np.nan
    df["rango_puntaje"] = df["puntaje"].rank(ascending=False, method="min")
    df["familia_seleccion"] = df["seleccionado_top7"].map(FAMILIA)
    cols = ["id_adjunto", "id_cap3", "autores_anio_titulo", "anio", "doi", "revista", "paginas", "trashnet",
            "seis_clases", "cuatro_clases", "metaheuristica", "refuerzo", "gan", "federado", "deteccion", "ensamble",
            "estadistica", "ruido_corrupcion", "repeticiones", "id_registro", "decision_tc", "prueba_estadistica_cap3",
            "dispersion_cap3", "codigo_propio", "nota_codigo", "acc_trashnet", "citas_openalex", "fecha_citas",
            "citas_por_anio", "motivo_exclusion_e2", "elegible_e2", "resumen", "s_citas", "s_recencia", "s_desempeno",
            "s_implementacion", "puntaje", "rango_puntaje", "seleccionado_top7", "familia_seleccion"]
    df[cols].sort_values(["elegible_e2", "puntaje"], ascending=[False, False]).to_csv(REV / "cribado_e2.csv", index=False)
    print(df["motivo_exclusion_e2"].replace("", "elegible").value_counts())
    print(df.loc[df.seleccionado_top7.notna(), ["seleccionado_top7", "elegible_e2", "motivo_exclusion_e2", "puntaje",
                                                 "rango_puntaje", "citas_openalex"]])


if __name__ == "__main__":
    if sys.argv[1] == "marcar":
        marcar(sys.argv[2])
    else:
        puntuar()
