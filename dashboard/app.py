"""Dashboard del proyecto TrashNet (E3): Contexto, EDA y Modelos.

Todas las cifras se leen de data/tabla_maestra.csv y de los CSV de contrastes, los mismos que usa el libro
(entregable3/resultados). La inferencia en vivo usa solo CPU y el modelo exportado a ONNX; si el archivo no está
disponible (por ejemplo, en un plan con poca memoria), la página de Modelos sigue funcionando con las salidas
precalculadas de las 361 imágenes de prueba.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CFG = json.loads((DATA / "config_modelo.json").read_text())
CL = CFG["clases"]
MODEL_PATH = Path(os.environ.get("NAGFUSE_ONNX", HERE / "modelos" / "nagfuse_seed00.onnx"))

# ------------------------------------------------------------------ datos (una sola fuente)
M = pd.read_csv(DATA / "tabla_maestra.csv")
RUN = pd.read_csv(DATA / "por_corrida.csv")
H1 = pd.read_csv(DATA / "contrastes_H1_f1.csv")
H2 = pd.read_csv(DATA / "contrastes_H2_ruido005.csv")
H3 = pd.read_csv(DATA / "H3_no_inferioridad.csv")
ABL = pd.read_csv(DATA / "ablacion.csv")
EXT = pd.read_csv(DATA / "contrastes_extra.csv")
CONF = pd.read_csv(DATA / "conforme.csv")
EDA = pd.read_csv(DATA / "eda_conteos.csv")
IMG = pd.read_csv(DATA / "nagfuse_s00_por_imagen.csv.gz")
PRED = pd.read_csv(DATA / "predicciones.csv")
MODELOS = [m for m in M.modelo if not m.startswith("NAGFuse/")]
CURV = pd.read_csv(DATA / "curvas.csv")                  # historial por época (datos_dashboard_diagnostico.py)
PROBS = pd.read_csv(DATA / "probs_prueba.csv.gz")        # probabilidades en prueba, una semilla por modelo
COSTO = pd.read_csv(DATA / "costo.csv").set_index("modelo")
FICHA = (DATA / "MODEL_CARD.md").read_text(encoding="utf-8")
DIAG = sorted(PROBS.modelo.unique())
FAMILIA = {"NAGFuse": "Original: fusión ViT + CNN con compuerta de ruido", "ViT-B16": "Transformer (preentrenado)",
           "RWCNet": "CNN residual (preentrenada)", "GFN": "Fusión de CNN (preentrenada)",
           "Celik-HC": "CNN híbrida (preentrenada)", "ResMsCapsule": "Cápsulas multiescala",
           "MbXception": "CNN separable (preentrenada)", "FocusRCNet": "CNN ligera con atención",
           "NAGFuse-B": "Original, opción B (protocolo v4): mismo mecanismo, otras ramas",
           "CNN": "Baseline: CNN desde cero", "MobileNetV2": "Baseline: MobileNetV2 desde cero",
           "ResNet-18": "Baseline: ResNet-18 desde cero"}
PALETA = ["#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#d68910", "#16a085", "#7f8c8d"]

COLORES = {"NAGFuse": "#c0392b", "NAGFuse-B": "#e67e22", "ViT-B16": "#2e86c1", "RWCNet": "#27ae60", "GFN": "#8e44ad", "Celik-HC": "#d68910"}
COND = {"limpio": "Limpio", "ruido_0.05": "Ruido σ = 0.05", "ruido_0.10": "Ruido σ = 0.10", "desenfoque": "Desenfoque",
        "brillo_-30": "Brillo −30 %", "brillo_+30": "Brillo +30 %", "oclusion": "Oclusión",
        "fondo_sustituido": "Fondo sustituido"}


def fila(modelo: str, col: str) -> float:
    return float(M.loc[M.modelo == modelo, col].iloc[0])


def tarjeta(titulo: str, valor: str, nota: str = "") -> html.Div:
    return html.Div([html.Div(titulo, className="t"), html.Div(valor, className="v"), html.Div(nota, className="n")],
                    className="card")


def layout_fig(fig: go.Figure, titulo: str, x: str, y: str, h: int = 380) -> go.Figure:
    fig.update_layout(title=titulo, xaxis_title=x, yaxis_title=y, height=h, template="plotly_white",
                      margin=dict(l=60, r=20, t=50, b=50), legend=dict(font=dict(size=10)))
    return fig


# ------------------------------------------------------------------ página 1: contexto
def pagina_contexto() -> html.Div:
    nag, vit, rwc = "NAGFuse", "ViT-B16", "RWCNet"
    h3 = H3.iloc[0]
    veredictos = [
        ("H1. Superar a los 12 rivales en F1 limpio", f"No: supera a {int(H1.supera.sum())} de 12; empata con RWCNet y ViT-B16"),
        ("H2. Superarlos con ruido σ = 0.05", f"Frente a las {int(H2.supera.sum())} CNN, sí; frente a ViT-B16, no (+{H2.set_index('rival').loc['ViT-B16','media_dif']:.3f}, n.s.)"),
        ("H3. No inferior al mejor rival", f"Sí: diferencia {h3.dif:+.4f} con RWCNet, TOST p = {h3.p_tost:.1e}"),
    ] + [(r.prediccion, r.resultado) for r in PRED.itertuples() if r.prediccion.startswith("P") and "semillas" not in r.prediccion]
    return html.Div([
        html.H2("Clasificación de residuos con TrashNet"),
        html.P("Seis clases de residuos reciclables fotografiados sobre fondo blanco. El objetivo del Entregable 3 es un "
               "modelo propio, NAGFuse, que conserve el desempeño del mejor modelo y no se degrade con ruido, "
               "comparado con la familia completa del E2 bajo un protocolo preregistrado."),
        html.Img(src="/assets/trashnet_clases.jpg", style={"maxWidth": "100%", "maxHeight": "220px"}),
        html.Div([
            tarjeta("F1 macro limpio · NAGFuse", f"{fila(nag, 'F1 limpio'):.3f}",
                    f"IC 95 % [{fila(nag, 'F1 limpio IC_lo'):.3f}, {fila(nag, 'F1 limpio IC_hi'):.3f}] · 20 semillas"),
            tarjeta("F1 con ruido σ = 0.05", f"{fila(nag, 'F1 ruido 0.05'):.3f}",
                    f"RWCNet {fila(rwc, 'F1 ruido 0.05'):.3f} · ViT-B16 {fila(vit, 'F1 ruido 0.05'):.3f}"),
            tarjeta("ECE (calibración)", f"{fila(nag, 'ECE'):.3f}", f"ViT-B16 {fila(vit, 'ECE'):.3f}"),
            tarjeta("Corridas analizadas", f"{len(RUN)}", "E1 + E2 + E3, misma partición de prueba"),
        ], className="cards"),
        html.H3("Qué se preguntó y qué se encontró"),
        dash_table.DataTable(data=[{"Pregunta": a, "Resultado": b} for a, b in veredictos],
                             style_cell={"textAlign": "left", "whiteSpace": "normal", "fontSize": 13, "padding": "6px"},
                             style_header={"fontWeight": "bold"}),
        html.H3("Cómo funciona NAGFuse"),
        html.Ol([
            html.Li("Estima el nivel de ruido σ̂ de cada imagen con la mediana de los coeficientes diagonales de Haar."),
            html.Li("Limpia la entrada de la rama convolucional (EfficientNet-B0) con contracción de ondícula."),
            html.Li("Mezcla las probabilidades de ViT-B/16 y de EfficientNet-B0 con un peso π que una compuerta sin "
                    "parámetros entrenables fija a partir de σ̂: con ruido fuerte, π = 1 y responde solo ViT."),
        ]),
    ])


# ------------------------------------------------------------------ página 2: EDA
def pagina_eda() -> html.Div:
    f1 = go.Figure()
    for sp in ["train", "val", "test"]:
        d = EDA[EDA.split == sp].set_index("label").reindex(CL)
        f1.add_bar(x=CL, y=d.n, name={"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}[sp])
    layout_fig(f1, "Imágenes por clase y partición (TrashNet, partición v2)", "Clase", "Número de imágenes")
    f1.update_layout(barmode="group")
    f2 = go.Figure()
    for c, col in [("limpio", "#2e86c1"), ("ruido_0.05", "#e67e22"), ("ruido_0.10", "#c0392b")]:
        f2.add_histogram(x=IMG[IMG.cond == c].sigma, name=COND[c], nbinsx=60, marker_color=col, opacity=0.7)
    f2.add_vline(x=CFG["s_lo"], line_dash="dash", annotation_text="s_lo")
    f2.add_vline(x=CFG["rho"] * CFG["s_lo"], line_dash="dot", annotation_text="ρ·s_lo")
    layout_fig(f2, "Nivel de ruido estimado σ̂ en las 361 imágenes de prueba", "σ̂ (escala de píxel [0, 1])",
               "Número de imágenes")
    f2.update_layout(barmode="overlay")
    extra = pd.DataFrame([
        {"Conjunto": "RealWaste", "Origen": "Vertedero real (UCI 908)", "Imágenes": 2640, "Clases": 6, "Casi duplicados retirados": 387},
        {"Conjunto": "Garbage v2", "Origen": "Celular, web y comunidad (Kaggle)", "Imágenes": 3000, "Clases": 6, "Casi duplicados retirados": 7220},
        {"Conjunto": "WaRP-C", "Origen": "Planta de reciclaje (Kaggle)", "Imágenes": 863, "Clases": 4, "Casi duplicados retirados": 1},
    ])
    return html.Div([
        html.H2("Exploración de los datos"),
        html.P("2521 imágenes agrupadas por casi duplicados en 1800 de entrenamiento, 360 de validación y 361 de prueba. "
               "La clase trash es la menos frecuente; por eso la métrica primaria es el F1 macro."),
        dcc.Graph(figure=f1),
        html.P("En las imágenes limpias σ̂ es casi siempre 0, porque el fondo blanco está saturado. Con ruido, el "
               "estimador recupera el σ añadido (0.05 y 0.10) y supera ρ·s_lo, de modo que la compuerta se abre."),
        dcc.Graph(figure=f2),
        html.H3("Conjuntos adicionales (evaluados sin reentrenar)"),
        dash_table.DataTable(data=extra.to_dict("records"), style_cell={"textAlign": "left", "fontSize": 13}),
    ])


# ------------------------------------------------------------------ página 3: modelos
def fig_forest(t: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure()
    col = np.where(t.supera, "#c0392b", np.where(t.mejor_sig, "#e67e22", "#7f8c8d"))
    fig.add_trace(go.Scatter(x=t.media_dif, y=t.rival, mode="markers", marker=dict(color=col, size=9),
                             error_x=dict(type="data", symmetric=False, array=t.ic_hi - t.media_dif,
                                          arrayminus=t.media_dif - t.ic_lo),
                             customdata=np.stack([t.R, t.p_max_holm], 1),
                             hovertemplate="%{y}: %{x:+.4f}<br>R = %{customdata[0]}<br>p máx. Holm = %{customdata[1]:.4f}"))
    fig.add_vline(x=0, line_color="black"); fig.add_vline(x=0.02, line_dash="dash", annotation_text="δ")
    return layout_fig(fig, titulo, "Diferencia NAGFuse − rival (IC 95 %)", "", 420)


def fig_ruido(sel: list[str]) -> go.Figure:
    fig = go.Figure(); xs = ["limpio", "σ = 0.05", "σ = 0.10"]
    for m in sel:
        cols = ["F1 limpio", "F1 ruido 0.05", "F1 ruido 0.10"]
        y = [fila(m, c) for c in cols]; lo = [fila(m, c + " IC_lo") for c in cols]; hi = [fila(m, c + " IC_hi") for c in cols]
        fig.add_trace(go.Scatter(x=xs, y=y, name=m, mode="lines+markers", line=dict(color=COLORES.get(m)),
                                 error_y=dict(type="data", symmetric=False, array=np.subtract(hi, y), arrayminus=np.subtract(y, lo))))
    return layout_fig(fig, "F1 macro frente al ruido (media e IC 95 %)", "Condición", "F1 macro", 360)


def tabla_metricas(sel: list[str]) -> list[dict]:
    cols = ["F1 limpio", "F1 ruido 0.05", "F1 ruido 0.10", "F1 desenfoque", "F1 oclusión", "F1 fondo", "ECE",
            "F1 RealWaste", "F1 Garbage v2", "F1 WaRP-C"]
    out = []
    for m in sel:
        r = {"Modelo": m, "R": int(fila(m, "R"))}
        for c in cols:
            v, lo, hi = fila(m, c), fila(m, c + " IC_lo"), fila(m, c + " IC_hi")
            r[c] = "—" if np.isnan(v) else (f"{v:.3f}" if np.isnan(lo) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
        out.append(r)
    return out


def tabla_lideres() -> list[dict]:
    de = RUN.groupby("model").f1.std()
    out = []
    for m in MODELOS:
        c = COSTO.loc[m] if m in COSTO.index else None
        num = lambda v, k=1: None if v is None or pd.isna(v) else round(float(v), k)
        out.append({"Modelo": m, "Familia": FAMILIA.get(m, "—"), "Semillas": int(fila(m, "R")),
                    "F1 macro (media)": round(fila(m, "F1 limpio"), 4), "DE": num(de.get(m), 4),
                    "F1 ruido 0.05": round(fila(m, "F1 ruido 0.05"), 4), "ECE": round(fila(m, "ECE"), 4),
                    "Parámetros (M)": num(None if c is None else c.params / 1e6),
                    "GFLOPs": num(None if c is None else c.gflops, 2),
                    "Latencia lote 1 (ms)": num(None if c is None else c.lat_bs1_ms)})
    return out


def color(i: int, m: str) -> str:
    return COLORES.get(m, PALETA[i % len(PALETA)])


def rgba(h: str, a: float) -> str:
    return f"rgba({int(h[1:3], 16)}, {int(h[3:5], 16)}, {int(h[5:7], 16)}, {a})"


def fig_curvas(sel: list[str], met: str) -> go.Figure:
    """Media ± DE entre semillas, por época; solo épocas a las que llegó al menos la mitad de las semillas."""
    fig = go.Figure()
    for i, m in enumerate(sel):
        g = CURV[CURV.modelo == m].groupby("epoca"); c = color(i, m)
        for parte, estilo in [("train", "dash"), ("val", "solid")]:
            col = f"{parte}_{met}"; n = g[col].count(); ok = n >= max(2, n.max() / 2)
            mu, sd = g[col].mean()[ok], g[col].std().fillna(0)[ok]; x = list(mu.index)
            fig.add_trace(go.Scatter(x=x + x[::-1], y=list(mu + sd) + list((mu - sd)[::-1]), fill="toself",
                                     fillcolor=rgba(c, 0.12), line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=mu, name=f"{m} · {'entrenamiento' if parte == 'train' else 'validación'}",
                                     line=dict(color=c, dash=estilo)))
        mejor = CURV[CURV.modelo == m].groupby("semilla").mejor_epoca.first().median()
        if not pd.isna(mejor):
            fig.add_vline(x=float(mejor), line_color=c, line_dash="dot")
    eje = "Pérdida (entropía cruzada)" if met == "loss" else "F1 macro"
    return layout_fig(fig, f"{eje} por época (media ± DE entre semillas)", "Época", eje, 380)


def roc_pr(y: np.ndarray, s: np.ndarray):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp / y.sum()]; fpr = np.r_[0, fp / (len(y) - y.sum())]
    prec = tp / np.arange(1, len(y) + 1); rec = tp / y.sum()
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)); ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    return fpr, tpr, auc, rec, prec, ap


def figs_diagnostico(m: str) -> list[go.Figure]:
    d = PROBS[PROBS.modelo == m]; P = d[[f"p_{k}" for k in range(6)]].to_numpy(); y = d.y.to_numpy()
    yp = P.argmax(1); sem = int(d.semilla.iloc[0])
    cm = np.zeros((6, 6)); np.add.at(cm, (y, yp), 1); cmn = cm / cm.sum(1, keepdims=True)
    f1 = go.Figure(go.Heatmap(z=cmn, x=CL, y=CL, colorscale="Blues", zmin=0, zmax=1, text=cm.astype(int),
                              texttemplate="%{text}", hovertemplate="real %{y} → predicha %{x}: %{z:.2f}<extra></extra>"))
    f1.update_yaxes(autorange="reversed")
    layout_fig(f1, f"Matriz de confusión normalizada por fila ({m}, semilla {sem})", "Clase predicha", "Clase real", 400)
    f2, f3 = go.Figure(), go.Figure()
    for k in range(6):
        fpr, tpr, auc, rec, prec, ap = roc_pr((y == k).astype(float), P[:, k])
        f2.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{CL[k]} (AUC {auc:.3f})", line=dict(color=PALETA[k % 7])))
        f3.add_trace(go.Scatter(x=rec, y=prec, name=f"{CL[k]} (AP {ap:.3f})", line=dict(color=PALETA[k % 7])))
    f2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
    layout_fig(f2, "Curvas ROC por clase (uno contra el resto)", "Tasa de falsos positivos", "Tasa de verdaderos positivos", 380)
    layout_fig(f3, "Curvas precisión-exhaustividad por clase", "Exhaustividad (recall)", "Precisión", 380)
    conf, ok = P.max(1), (yp == y).astype(float); b = np.minimum((conf * 15).astype(int), 14)
    cnt = np.bincount(b, minlength=15); acc = np.bincount(b, ok, 15) / np.maximum(cnt, 1)
    cmed = np.bincount(b, conf, 15) / np.maximum(cnt, 1); ece = float(np.sum(cnt / len(y) * np.abs(acc - cmed)))
    v = cnt > 0; f4 = go.Figure()
    f4.add_bar(x=cmed[v], y=acc[v], width=1 / 15 * 0.9, name="Exactitud por intervalo", marker_color="#2e86c1",
               customdata=cnt[v], hovertemplate="confianza %{x:.2f}<br>exactitud %{y:.2f}<br>n = %{customdata}<extra></extra>")
    f4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Calibración perfecta", line=dict(dash="dash", color="gray")))
    layout_fig(f4, f"Diagrama de confiabilidad (15 intervalos, ECE = {ece:.3f})", "Confianza media", "Exactitud", 380)
    return [f1, f2, f3, f4]


def fig_pareto(eje: str, titulo: str) -> go.Figure:
    d = pd.DataFrame([{"m": m, "x": COSTO.loc[m, eje] if m in COSTO.index else np.nan, "y": fila(m, "F1 limpio")}
                      for m in MODELOS]).dropna().sort_values("x")
    front, mejor = [], -1.0
    for r in d.itertuples():
        if r.y > mejor:
            front.append(r); mejor = r.y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.x, y=d.y, mode="markers+text", text=d.m, textposition="top center",
                             marker=dict(size=10, color=["#c0392b" if m == "NAGFuse" else "#7f8c8d" for m in d.m]),
                             name="Modelos", hovertemplate="%{text}: F1 %{y:.3f}, %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r.x for r in front], y=[r.y for r in front], mode="lines", name="Frontera de Pareto",
                             line=dict(dash="dash", color="#27ae60")))
    fig.update_xaxes(type="log")
    return layout_fig(fig, titulo, titulo.split(" frente a ")[-1], "F1 macro limpio", 400)


def fig_global() -> go.Figure:
    """Interpretabilidad global del mecanismo: peso medio de la rama ViT y F1 de cada rama por condición."""
    from_cond = [c for c in COND if c in set(IMG.cond)]
    def f1m(y, yp):
        f = [];
        for k in range(6):
            tp = np.sum((yp == k) & (y == k)); f.append(2 * tp / max(np.sum(yp == k) + np.sum(y == k), 1))
        return float(np.mean(f))
    filas = []
    for c in from_cond:
        d = IMG[IMG.cond == c]; y = d.y.to_numpy()
        pv = d[[f"pv_{k}" for k in range(6)]].to_numpy().argmax(1); pc = d[[f"pc_{k}" for k in range(6)]].to_numpy().argmax(1)
        filas.append((COND[c], d.pi.mean(), f1m(y, d.pred.to_numpy()), f1m(y, pv), f1m(y, pc)))
    t = pd.DataFrame(filas, columns=["cond", "pi", "nag", "vit", "cnn"]); fig = go.Figure()
    fig.add_bar(x=t.cond, y=t.nag, name="NAGFuse (mezcla)", marker_color="#c0392b")
    fig.add_bar(x=t.cond, y=t.vit, name="Rama ViT-B/16 sola", marker_color="#2e86c1")
    fig.add_bar(x=t.cond, y=t.cnn, name="Rama EfficientNet-B0 sola", marker_color="#27ae60")
    fig.add_trace(go.Scatter(x=t.cond, y=t.pi, name="π medio (peso de ViT)", mode="lines+markers",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    return layout_fig(fig, "Interpretabilidad global: qué rama decide en cada condición (semilla 0, 361 imágenes)",
                      "Condición", "F1 macro / π", 420)


def tabla_hipotesis() -> list[dict]:
    out = []
    for nombre, t in [("H1 F1 limpio", H1), ("H2 F1 ruido 0.05", H2)]:
        for r in t.itertuples():
            out.append({"Hipótesis": nombre, "Rival": r.rival, "Semillas": int(r.R), "Diferencia": round(r.media_dif, 4),
                        "IC 95 %": f"[{r.ic_lo:+.4f}, {r.ic_hi:+.4f}]", "p máx. Holm": float(f"{r.p_max_holm:.2g}"),
                        "Cliff δ": round(r.cliff, 2), "d_z": round(r.dz, 2), "Veredicto": "Supera" if r.supera else "No supera"})
    return out


def seccion_original() -> html.Div:
    return html.Div([
        html.H3("Modelo original: NAGFuse"),
        html.P("Procedencia: el estimador de ruido y la contracción vienen de la eliminación de ruido por ondículas "
               "(Donoho y Johnstone, 1994; Donoho, 1995); la mezcla, de las mezclas de expertos (Jacobs et al., 1991). "
               "Diferencias: la compuerta es analítica (sin parámetros entrenables), el estimador usa los tres canales, "
               "la contracción solo se aplica a la rama sensible al ruido y la mezcla se hace en el espacio de "
               "probabilidades. El paper guía y los rivales usan un único extractor sin adaptación al ruido."),
        html.Pre("""entrada x (3×224×224, escala [0, 1])
σ̂  ← mediana(|HH1(x)|) / 0.6745            # Haar ortonormal, 3 canales
π  ← π0 + (1 − π0) · clip((σ̂ − s_lo) / ((ρ − 1) s_lo), 0, 1)
p_V ← softmax(ViT-B/16(x))
p_C ← softmax(EfficientNet-B0(contracción_ondícula(x, σ̂)))
p  ← π · p_V + (1 − π) · p_C                  # predicción = argmax p""", style={"background": "#fff", "padding": "10px"}),
        html.H4("Curvas de entrenamiento"),
        dcc.Dropdown(DIAG, ["NAGFuse", "ViT-B16"], multi=True, id="sel-curvas"),
        html.Div([dcc.Graph(id="fig-curvas-loss"), dcc.Graph(id="fig-curvas-f1")], className="dos"),
        html.P("Línea discontinua: entrenamiento; continua: validación; vertical punteada: mediana de la época elegida "
               "por early stopping. NAGFuse tiene 20 semillas; los rivales, las semillas guardadas en el repositorio "
               "del E3 (4 a 12). La curva de aprendizaje frente al tamaño de muestra no se calculó (ver capítulo 29, limitaciones).",
               className="n"),
        html.H4("Pruebas de hipótesis (p ajustados por Holm, máximo de t, Wilcoxon y bootstrap jerárquico)"),
        dash_table.DataTable(data=tabla_hipotesis(), sort_action="native", filter_action="native", page_size=12,
                             style_cell={"fontSize": 12}, style_table={"overflowX": "auto"}),
        html.H4("Ablación"),
        dash_table.DataTable(data=ABL[["cond", "variante", "completo", "variante_f1", "dif", "p_max_holm"]].round(4)
                             .to_dict("records"), style_cell={"fontSize": 12}),
        html.H4("Veredicto"),
        html.P("No superioridad general: NAGFuse es equivalente al mejor rival en datos limpios (TOST con RWCNet) y "
               "supera a las CNN con ruido σ = 0.05, pero no supera a ViT-B16 con ruido ni en los conjuntos adicionales, "
               "donde ViT-B16 es mejor. Robustez sin pérdida en limpio, no superioridad.", className="aviso"),
    ])


def pagina_ficha() -> html.Div:
    return html.Div([dcc.Markdown(FICHA, mathjax=True)])


def pagina_modelos() -> html.Div:
    return html.Div([
        html.H2("Modelos y resultados"),
        html.H3("Tabla de líderes"),
        dash_table.DataTable(data=tabla_lideres(), sort_action="native", filter_action="native",
                             sort_by=[{"column_id": "F1 macro (media)", "direction": "desc"}],
                             style_cell={"fontSize": 12, "whiteSpace": "normal", "textAlign": "left"},
                             style_table={"overflowX": "auto"},
                             style_data_conditional=[{"if": {"filter_query": '{Modelo} = "NAGFuse"'},
                                                      "fontWeight": "bold"}]),
        html.P("F1 macro limpio en prueba: media y desviación estándar entre semillas. Parámetros, GFLOPs y latencia: "
               "mediana de las mediciones guardadas en el entrenamiento (GPU A100 con AMP, lote de 1). La latencia en "
               "CPU del ONNX de NAGFuse se muestra al clasificar una imagen propia. Los baselines desde cero no "
               "registraron FLOPs.", className="n"),
        html.Label("Modelos a comparar"),
        html.Label("Modelos a comparar"),
        dcc.Dropdown(MODELOS, ["NAGFuse", "ViT-B16", "RWCNet", "GFN"], multi=True, id="sel-modelos"),
        dash_table.DataTable(id="tabla-metricas", style_cell={"fontSize": 12, "whiteSpace": "normal"},
                             style_table={"overflowX": "auto"}),
        html.P("Media sobre semillas con IC 95 % t. En los conjuntos adicionales, los rivales tienen solo las "
               "semillas 8 en adelante.", className="n"),
        dcc.Graph(id="fig-ruido"),
        html.Div([dcc.Graph(figure=fig_forest(H1, "H1: F1 limpio")),
                  dcc.Graph(figure=fig_forest(H2, "H2: F1 con ruido σ = 0.05"))], className="dos"),
        html.P("Rojo: supera por más de δ = 0.02 con p de Holm < 0.05 (máximo de t, Wilcoxon y bootstrap jerárquico); "
               "naranja: significativo pero menor que δ; gris: no significativo.", className="n"),
        html.H3("Diagnóstico del modelo seleccionado"),
        dcc.Dropdown(DIAG, "NAGFuse", id="sel-diag", clearable=False),
        html.Div([dcc.Graph(id="fig-conf"), dcc.Graph(id="fig-fiab")], className="dos"),
        html.Div([dcc.Graph(id="fig-roc"), dcc.Graph(id="fig-pr")], className="dos"),
        html.H3("Comparación estadística y costo"),
        html.Img(src="/assets/diferencia_critica.png", alt="Diagrama de diferencia crítica de Nemenyi con los 11 "
                 "modelos ordenados por rango medio en 12 escenarios", style={"maxWidth": "100%"}),
        html.P("Diagrama de diferencia crítica (Friedman + Nemenyi, CD = 4.36, 11 modelos, 12 escenarios). Los modelos "
               "unidos por una barra no difieren significativamente.", className="n"),
        html.Div([dcc.Graph(figure=fig_pareto("gflops", "F1 limpio frente a GFLOPs")),
                  dcc.Graph(figure=fig_pareto("lat_bs1_ms", "F1 limpio frente a latencia (ms)"))], className="dos"),
        html.H3("Interpretabilidad global"),
        dcc.Graph(figure=fig_global()),
        html.P("Con ruido, π sube a 1 y NAGFuse responde como su rama ViT (el mejor benchmark con ruido); en limpio "
               "promedia ambas ramas. Esto compara directamente la explicación del modelo original con la del mejor "
               "benchmark: la decisión con ruido la toma la misma red.", className="n"),
        seccion_original(),
        html.H3("Explorar una imagen de prueba"),
        html.Div([
            html.Div([html.Label("Imagen (índice 0–360)"),
                      dcc.Slider(0, 360, 1, value=0, id="sel-img", marks=None, tooltip={"placement": "bottom"})],
                     style={"flex": 2}),
            html.Div([html.Label("Perturbación"),
                      dcc.Dropdown([{"label": v, "value": k} for k, v in COND.items()], "ruido_0.05", id="sel-cond",
                                   clearable=False)], style={"flex": 1}),
        ], style={"display": "flex", "gap": "20px"}),
        html.Div(id="detalle-img"),
        html.H3("Clasificar una imagen propia"),
        dcc.Upload(id="subir", children=html.Div(["Arrastre una imagen o ", html.A("selecciónela")]),
                   className="upload", accept="image/*"),
        dcc.Checklist([{"label": " Calcular mapa de oclusión (16 pasadas, unos segundos)", "value": "oc"}], [],
                      id="ocl"),
        dcc.Loading(html.Div(id="resultado-subida")),
    ])


def barras_prob(p: np.ndarray, pv: np.ndarray, pc: np.ndarray, titulo: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=CL, y=p, name="NAGFuse p", marker_color="#c0392b")
    fig.add_bar(x=CL, y=pv, name="Rama ViT p_V", marker_color="#2e86c1", opacity=0.6)
    fig.add_bar(x=CL, y=pc, name="Rama CNN p_C", marker_color="#27ae60", opacity=0.6)
    fig.update_layout(barmode="group", yaxis_range=[0, 1])
    return layout_fig(fig, titulo, "Clase", "Probabilidad", 320)


def resumen_incertidumbre(p: np.ndarray, sigma: float, pi: float, fondo: float | None = None) -> html.Div:
    conj = [CL[k] for k in range(6) if 1 - p[k] <= CFG["q_conforme"]]
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(6))
    avisos = []
    if sigma >= CFG["rho"] * CFG["s_lo"]:
        avisos.append("Se detecta ruido fuerte: la compuerta está abierta (π = 1) y responde solo la rama ViT.")
    elif sigma > CFG["s_lo"]:
        avisos.append("Se detecta algo de ruido: la compuerta está parcialmente abierta.")
    if len(conj) != 1:
        avisos.append(f"Predicción incierta: el conjunto conforme al 90 % tiene {len(conj)} clases.")
    if fondo is not None and fondo < 0.2:
        avisos.append("Posible imagen fuera de dominio: tiene poco fondo claro y TrashNet se tomó sobre fondo blanco. "
                      "En otros dominios el modelo pierde entre 0.4 y 0.6 de F1 (ver conjuntos adicionales).")
    return html.Div([
        html.P(f"Predicción: {CL[int(p.argmax())]} ({p.max():.2f}) · σ̂ = {sigma:.4f} · π = {pi:.2f} · "
               f"entropía normalizada = {ent:.2f} · conjunto conforme 90 %: {{{', '.join(conj) or 'vacío'}}}"),
        html.Ul([html.Li(a, className="aviso") for a in avisos]) if avisos else html.P("Sin avisos.", className="n"),
    ])


# ------------------------------------------------------------------ inferencia (solo CPU)
@lru_cache(maxsize=2)          # caché de a lo sumo 2 modelos en memoria
def sesion(ruta: str):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    return ort.InferenceSession(ruta, so, providers=["CPUExecutionProvider"])


def preparar(contenido: str) -> tuple[np.ndarray, float]:
    from PIL import Image
    raw = base64.b64decode(contenido.split(",", 1)[1])
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x01 = np.asarray(im, dtype=np.float32) / 255.0                      # (224, 224, 3)
    fondo = float((x01.min(axis=2) >= 200 / 255).mean())               # fracción de fondo claro (criterio del E2)
    x = (x01 - np.array(CFG["media"], np.float32)) / np.array(CFG["desv"], np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32), fondo        # (1, 3, 224, 224)


def oclusion(s, x: np.ndarray, k: int, g: int = 4) -> np.ndarray:
    """Caída de la probabilidad de la clase k al tapar cada celda de una rejilla g×g con gris medio."""
    gris = ((0.5 - np.array(CFG["media"])) / np.array(CFG["desv"])).astype(np.float32)[:, None, None]
    lote, paso = [], 224 // g
    for i in range(g):
        for j in range(g):
            z = x[0].copy(); z[:, i*paso:(i+1)*paso, j*paso:(j+1)*paso] = gris; lote.append(z)
    p0 = s.run(None, {"x": x})[0][0, k]
    pk = s.run(None, {"x": np.stack(lote)})[0][:, k]
    return (p0 - pk).reshape(g, g)


app = Dash(__name__, suppress_callback_exceptions=True, title="NAGFuse · TrashNet")
server = app.server
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Nav([dcc.Link("Contexto", href="/"), dcc.Link("EDA", href="/eda"), dcc.Link("Modelos", href="/modelos"),
              dcc.Link("Ficha del modelo", href="/ficha")],
             className="nav"),
    html.Div(id="pagina", className="page"),
    html.Footer([html.Div("Anyely Sanjuan Benavides y Juan David Díaz Fernández · Deep Learning, Universidad del Norte · "
                          "versión E3 · actualizado el 23-11-2026"),
                 html.Div("Privacidad: las imágenes subidas se procesan solo en memoria para la predicción; no se "
                          "guardan, no se registran y no se comparten.")]),
])


@app.callback(Output("pagina", "children"), Input("url", "pathname"))
def enrutar(path):
    return {"/eda": pagina_eda, "/modelos": pagina_modelos, "/ficha": pagina_ficha}.get(path, pagina_contexto)()


@app.callback(Output("tabla-metricas", "data"), Output("fig-ruido", "figure"), Input("sel-modelos", "value"))
def actualizar(sel):
    sel = sel or ["NAGFuse"]
    return tabla_metricas(sel), fig_ruido(sel)


@app.callback(Output("fig-curvas-loss", "figure"), Output("fig-curvas-f1", "figure"), Input("sel-curvas", "value"))
def curvas(sel):
    sel = sel or ["NAGFuse"]
    return fig_curvas(sel, "loss"), fig_curvas(sel, "f1")


@app.callback(Output("fig-conf", "figure"), Output("fig-roc", "figure"), Output("fig-pr", "figure"),
              Output("fig-fiab", "figure"), Input("sel-diag", "value"))
def diagnostico(m):
    f = figs_diagnostico(m or "NAGFuse")
    return f[0], f[1], f[2], f[3]


@app.callback(Output("detalle-img", "children"), Input("sel-img", "value"), Input("sel-cond", "value"))
def detalle(i, cond):
    r = IMG[(IMG.cond == cond) & (IMG.i == int(i))].iloc[0]
    p = np.array([r[f"p_{k}"] for k in range(6)]); pv = np.array([r[f"pv_{k}"] for k in range(6)])
    pc = np.array([r[f"pc_{k}"] for k in range(6)])
    tit = f"{r.archivo} · clase real: {CL[int(r.y)]} · {COND[cond]} (semilla 0, salidas precalculadas)"
    return html.Div([dcc.Graph(figure=barras_prob(p, pv, pc, tit)), resumen_incertidumbre(p, r.sigma, r.pi)])


@app.callback(Output("resultado-subida", "children"), Input("subir", "contents"), State("ocl", "value"))
def clasificar(contenido, ocl):
    if not contenido:
        return html.P("Aún no hay imagen.", className="n")
    if len(contenido) > 8_000_000:                           # ~6 MB de imagen codificada en base64
        return html.P("La imagen supera el tamaño máximo (6 MB). Suba una más pequeña.", className="aviso")
    if not MODEL_PATH.exists():
        return html.P("El modelo ONNX no está disponible en este despliegue (supera la memoria del plan gratuito). "
                      "Use el explorador de imágenes de prueba o ejecute el dashboard localmente con "
                      "NAGFUSE_ONNX=<ruta al .onnx>.", className="aviso")
    try:
        x, fondo = preparar(contenido)
        s = sesion(str(MODEL_PATH))
        t0 = time.perf_counter()
        p, sigma, pi, pv, pc = [o[0] for o in s.run(None, {"x": x})]
        lat = (time.perf_counter() - t0) * 1000
    except Exception as e:                                   # imagen ilegible o error de inferencia
        return html.P(f"No se pudo procesar la imagen: {e}", className="aviso")
    hijos = [dcc.Graph(figure=barras_prob(p, pv, pc, "Imagen subida")),
             resumen_incertidumbre(p, float(sigma), float(pi), fondo),
             html.P(f"Latencia de esta inferencia en CPU (ONNX Runtime, 2 hilos): {lat:.0f} ms. Probabilidad "
                    f"calibrada: ECE de NAGFuse en prueba = {fila('NAGFuse', 'ECE'):.3f}; el conjunto conforme tiene "
                    f"cobertura nominal del 90 %.", className="n")]
    if ocl:
        h = oclusion(s, x, int(p.argmax()))
        fig = go.Figure(go.Heatmap(z=h, colorscale="Reds", colorbar=dict(title="Caída de p")))
        fig.update_yaxes(autorange="reversed")
        hijos.append(dcc.Graph(figure=layout_fig(fig, f"Mapa de oclusión para '{CL[int(p.argmax())]}'",
                                                 "Columna de la rejilla", "Fila de la rejilla", 320)))
        hijos.append(html.P("Las celdas rojas son las regiones que, al taparlas, más reducen la probabilidad de la "
                            "clase predicha: es donde el modelo mira.", className="n"))
    return html.Div(hijos)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
