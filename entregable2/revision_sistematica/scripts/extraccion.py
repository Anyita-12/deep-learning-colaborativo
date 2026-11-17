import csv
import json
import re
import sys
from pathlib import Path

CORPUS, MD, CODIGO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])

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
DATASETS = [
    ("TrashNet", r"trashnet|yang and thung|thung"),
    ("TACO", r"\btaco\b"),
    ("TrashBox", r"trashbox"),
    ("Garbage Classification (Kaggle, 12 clases)", r"garbage[ _-]?classification.{0,60}(kaggle|12 classes|twelve)|mostafaabla"),
    ("Waste Classification data (Kaggle, O/R)", r"waste classification data|techsash|sashaank|organic and recyclable"),
    ("Huawei / garbage China", r"huawei"),
    ("Propio / recolectado", r"(self-?(built|made|collected|constructed)|our own dataset|we (collected|captured|built)|own dataset|collected by the authors|taken by the authors)"),
]
METRICAS = [("Accuracy", r"\baccuracy"), ("Precisión", r"\bprecision"), ("Recall", r"\brecall|sensitivity"),
            ("F1", r"\bf1|f-?score|f-measure"), ("AUC", r"\bauc\b|roc curve"), ("MCC", r"\bmcc\b|matthews"),
            ("Kappa", r"\bkappa")]
RE_RATIO = re.compile(r"\b(\d{2})\s*(?:%|percent)?\s*[:/]\s*(\d{2})(?:\s*[:/]\s*(\d{2}))?\b")
RE_RATIO_SIMPLE = re.compile(r"\b([5-9])\s*:\s*([1-5])(?:\s*:\s*([1-5]))?\b(?=[^.]{0,60}(?:train|test|valid|split|divid|ratio))")
RE_PCT = re.compile(r"\b(\d{2})\s*%\s*(?:of the (?:data(?:set)?|images) )?(?:(?:is|was|were|are) )?(?:used |allocated |assigned )?(?:for |as |to )?(?:the )?(train\w*|test\w*|valid\w*)")
RE_KFOLD = re.compile(r"\b(\d+|five|ten)[- ]fold|cross[- ]validation", re.I)
RE_VAL = re.compile(r"\bvalidation (set|data|split|subset)|\bval(idation)? set|train(ing)?[, /-]+validation", re.I)
RE_REPS = re.compile(r"\b(random seeds?|different seeds|independent (runs|trials|experiments)|repeated (\w+ )?(times|runs)|"
                     r"(\d+|five|ten|three) (independent )?(runs|trials|repetitions)|averaged over (\d+|five|ten) |mean\s*(±|\+/-)|\d\s*±\s*\d)", re.I)
RE_STATS = re.compile(r"\b(t-?test|wilcoxon|anova|friedman|mcnemar|kruskal|nemenyi|mann-whitney|statistically significant|"
                      r"statistical significance|p\s*[<=]\s*0\.0\d)", re.I)
RE_DEDUP = re.compile(r"\b(duplicat\w*|near-duplicate|data leakage)", re.I)
RE_BASE_JUSTO = re.compile(r"(same (hyperparameters?|training (settings|setup|protocol|conditions)|experimental (settings|setup|conditions))|"
                           r"identical (settings|conditions|hyperparameters)|under the same|all (models|networks) (were|are) trained (with|using|under))", re.I)
RE_ACC = re.compile(r"accuracy[^.%]{0,60}?(\d{2}(?:\.\d+)?)\s*%", re.I)


def texto(idz):
    p = MD / f"{idz}.md"
    if not p.exists():
        return ""
    t = p.read_text(encoding="utf-8")
    corte = re.search(r"\n#+\s*(references|bibliography|referencias)\b|\nreferences\n", t, re.I)
    return t[:corte.start()] if corte else t


def main():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    codigo = {r["id_zotero"]: r for r in csv.DictReader(open(CODIGO, encoding="utf-8-sig"))}
    filas = []
    for c in corpus:
        t = texto(c["id_zotero"])
        low = t.lower()
        cab = f"{c['titulo']} {c['resumen']}".lower()
        fams = [n for n, rx in FAMILIAS if re.search(rx, cab)] or [n for n, rx in FAMILIAS if re.search(rx, low[:15000])]
        dsets = [n for n, rx in DATASETS if re.search(rx, low)]
        flat = re.sub(r"\s+", " ", low)
        ratios = set()
        for m in RE_RATIO.finditer(flat):
            v = [int(x) for x in m.groups() if x]
            if 99 <= sum(v) <= 101 and v[0] >= 50:
                ratios.add("/".join(map(str, v)))
        for m in RE_RATIO_SIMPLE.finditer(flat):
            v = [int(x) for x in m.groups() if x]
            if sum(v) == 10:
                ratios.add("/".join(str(x * 10) for x in v))
        for m in RE_PCT.finditer(flat):
            ratios.add(f"{ {'tr': 'train', 'te': 'test', 'va': 'val'}[m.group(2)[:2]]} {m.group(1)}%")
        ratios = sorted(ratios)[:6]
        kfold = bool(RE_KFOLD.search(low))
        val = bool(RE_VAL.search(low))
        reps = bool(RE_REPS.search(low))
        stats = sorted({m.group(0).lower() for m in RE_STATS.finditer(low)})
        dedup = bool(RE_DEDUP.search(low))
        justo = bool(RE_BASE_JUSTO.search(low))
        acc = [float(m.group(1)) for m in RE_ACC.finditer(f"{c['titulo']} {c['resumen']}") if 50 <= float(m.group(1)) <= 100]
        cod = codigo.get(c["id_zotero"], {})
        if not t:
            fuga, nrep, equidad = "No evaluable (sin texto completo)", "No evaluable", "No evaluable"
        else:
            if val and dedup:
                fuga = "Bajo"
            elif val or kfold:
                fuga = "Medio"
            else:
                fuga = "Alto"
            nrep = "Varias (declaradas)" if reps else "Una (no declara repeticiones)"
            equidad = "Declarada" if justo else "No declarada"
        filas.append({
            "id_zotero": c["id_zotero"], "anio": c["anio"], "autores": (c["autores"][0] + (" et al." if len(c["autores"]) > 1 else "")) if c["autores"] else "",
            "titulo": c["titulo"], "doi": c["doi"], "venue": c["venue"] or c["venue_openalex"], "tipo": c["tipo"],
            "citas_openalex": c["citas"], "venue_2yr_mean_citedness": round(c["venue_2yr_mean_citedness"], 2) if c["venue_2yr_mean_citedness"] else "",
            "arquitectura_familias": "; ".join(fams), "datasets": "; ".join(dsets) or "No identificado",
            "particion_ratios": "; ".join(ratios) or "No identificada", "k_fold": "Sí" if kfold else "No",
            "conjunto_validacion": "Sí" if val else "No declarado", "metricas": "; ".join(n for n, rx in METRICAS if re.search(rx, low)),
            "accuracy_en_resumen": max(acc) if acc else "", "repeticiones_semillas": "Sí" if reps else "No declaradas",
            "pruebas_estadisticas": "; ".join(stats) if stats else "No",
            "codigo_publico": "Sí, oficial" if cod.get("codigo_propio") else ("Parcial / terceros / a petición" if cod.get("nota_codigo") else "No"),
            "baselines_ajustados_igual": "Declarado" if justo else "No declarado",
            "control_duplicados_fuga": "Menciona" if dedup else "No menciona",
            "riesgo_fuga": fuga, "repeticiones": nrep, "equidad_baselines": equidad,
            "etiqueta_equipo": "; ".join(e for e in c["etiquetas"] if not e.startswith("R")),
            "seleccionado_top7": cod.get("seleccionado_top7", ""),
        })
    verificados = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8")) if len(sys.argv) > 5 else {}
    for fila in filas:
        v = verificados.get(fila["id_zotero"])
        fila["verificado_manual"] = "Sí" if v else "No (extracción automática)"
        if v:
            fila.update(v)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(len(filas))


if __name__ == "__main__":
    main()
