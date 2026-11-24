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

COLORES = {"NAGFuse": "#c0392b", "ViT-B16": "#2e86c1", "RWCNet": "#27ae60", "GFN": "#8e44ad", "Celik-HC": "#d68910"}
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


def pagina_modelos() -> html.Div:
    return html.Div([
        html.H2("Modelos y resultados"),
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
    html.Nav([dcc.Link("Contexto", href="/"), dcc.Link("EDA", href="/eda"), dcc.Link("Modelos", href="/modelos")],
             className="nav"),
    html.Div(id="pagina", className="page"),
    html.Footer("Anyely Sanjuan Benavides y Juan David Díaz Fernández · Deep Learning, Universidad del Norte · 2026"),
])


@app.callback(Output("pagina", "children"), Input("url", "pathname"))
def enrutar(path):
    return {"/eda": pagina_eda, "/modelos": pagina_modelos}.get(path, pagina_contexto)()


@app.callback(Output("tabla-metricas", "data"), Output("fig-ruido", "figure"), Input("sel-modelos", "value"))
def actualizar(sel):
    sel = sel or ["NAGFuse"]
    return tabla_metricas(sel), fig_ruido(sel)


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
    if not MODEL_PATH.exists():
        return html.P("El modelo ONNX no está disponible en este despliegue (supera la memoria del plan gratuito). "
                      "Use el explorador de imágenes de prueba o ejecute el dashboard localmente con "
                      "NAGFUSE_ONNX=<ruta al .onnx>.", className="aviso")
    try:
        x, fondo = preparar(contenido)
        s = sesion(str(MODEL_PATH))
        p, sigma, pi, pv, pc = [o[0] for o in s.run(None, {"x": x})]
    except Exception as e:                                   # imagen ilegible o error de inferencia
        return html.P(f"No se pudo procesar la imagen: {e}", className="aviso")
    hijos = [dcc.Graph(figure=barras_prob(p, pv, pc, "Imagen subida")),
             resumen_incertidumbre(p, float(sigma), float(pi), fondo)]
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
