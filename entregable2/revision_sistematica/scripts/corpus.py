import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RDF, CSV_CORPUS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
UA = "Mozilla/5.0 (academic systematic review script)"
NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "z": "http://www.zotero.org/namespaces/export#",
      "dcterms": "http://purl.org/dc/terms/", "bib": "http://purl.org/net/biblio#",
      "foaf": "http://xmlns.com/foaf/0.1/", "link": "http://purl.org/rss/1.0/modules/link/",
      "dc": "http://purl.org/dc/elements/1.1/"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def parse_rdf():
    root = ET.parse(RDF).getroot()
    journals = {}
    for j in root.findall("bib:Journal", NS):
        about = j.get(f"{{{NS['rdf']}}}about")
        ids = [i.text for i in j.findall("dc:identifier", NS)]
        journals[about] = {"venue": (j.findtext("dc:title", "", NS)),
                           "doi": next((i[4:] for i in ids if i.startswith("DOI ")), "")}
    items = []
    for el in list(root):
        if el.find("bib:authors", NS) is None:
            continue
        part = el.find("dcterms:isPartOf", NS)
        venue, doi = "", ""
        if part is not None:
            res = part.get(f"{{{NS['rdf']}}}resource")
            if res in journals:
                venue, doi = journals[res]["venue"], journals[res]["doi"]
            else:
                j = part.find("bib:Journal", NS)
                if j is not None:
                    venue = j.findtext("dc:title", "", NS)
                    doi = next((i.text[4:] for i in j.findall("dc:identifier", NS) if i.text.startswith("DOI ")), "")
        conf = el.find("bib:presentedAt/bib:Conference/dc:title", NS)
        if conf is not None and not venue:
            venue = conf.text
        tags = [s.text for s in el.findall("dc:subject", NS)]
        att = el.find("link:link", NS)
        items.append({
            "titulo": el.findtext("dc:title", "", NS),
            "anio": el.findtext("dc:date", "", NS)[:4],
            "autores": [p.findtext("foaf:surname", "", NS) for p in el.iter(f"{{{NS['foaf']}}}Person")],
            "venue": venue, "doi": doi.lower(),
            "tipo": el.findtext("z:itemType", "", NS),
            "etiquetas": tags, "resumen": el.findtext("dcterms:abstract", "", NS),
            "att": att.get(f"{{{NS['rdf']}}}resource") if att is not None else ""})
    rdf = f"{{{NS['rdf']}}}"
    atts = {a.get(rdf + "about"): a.find("z:path", NS).get(rdf + "resource")
            for a in root.findall("z:Attachment", NS) if a.find("z:path", NS) is not None}
    for it in items:
        path = atts.get(it["att"], "")
        m = re.search(r"files/(\d+)/", path)
        it["id_zotero"] = m.group(1) if m else ""
    return items


def enrich(items):
    dois = [i["doi"] for i in items if i["doi"]]
    found = {}
    for k in range(0, len(dois), 40):
        chunk = dois[k:k + 40]
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": "doi:" + "|".join(chunk), "per_page": 50,
             "select": "id,doi,cited_by_count,publication_year,type,primary_location"})
        for w in get(url)["results"]:
            found[(w.get("doi") or "").replace("https://doi.org/", "").lower()] = w
        time.sleep(1)
    sources = {}
    for w in found.values():
        src = ((w.get("primary_location") or {}).get("source") or {})
        if src.get("id"):
            sources[src["id"].split("/")[-1]] = None
    ids = list(sources)
    for k in range(0, len(ids), 40):
        url = "https://api.openalex.org/sources?" + urllib.parse.urlencode(
            {"filter": "openalex:" + "|".join(ids[k:k + 40]), "per_page": 50,
             "select": "id,display_name,summary_stats,type,host_organization_name"})
        for s in get(url)["results"]:
            sources[s["id"].split("/")[-1]] = s
        time.sleep(1)
    for it in items:
        w = found.get(it["doi"])
        it["openalex_id"] = w["id"] if w else ""
        it["citas"] = w["cited_by_count"] if w else None
        src = ((w or {}).get("primary_location") or {}).get("source") or {}
        s = sources.get((src.get("id") or "").split("/")[-1]) or {}
        st = s.get("summary_stats") or {}
        it["venue_openalex"] = s.get("display_name", "")
        it["venue_2yr_mean_citedness"] = st.get("2yr_mean_citedness")
        it["venue_h_index"] = st.get("h_index")
        it["editorial"] = s.get("host_organization_name", "")
    return items


if __name__ == "__main__":
    items = enrich(parse_rdf())
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), sum(1 for i in items if i["openalex_id"]), "con OpenAlex")
