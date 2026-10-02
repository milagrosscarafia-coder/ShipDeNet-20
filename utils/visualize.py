"""
Visualizacion con Plotly y los colores del paper:
  verde = ground truth detectado, rojo = barco no detectado,
  azul = deteccion correcta, amarillo = falsa alarma.
Formato de cajas: (x1, y1, x2, y2) en pixeles.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.metrics import _np, match_image

ESTILO = {  # tipo: (color, ancho, trazo, nombre en la leyenda)
    "gt": ("limegreen", 1.5, "solid", "GT detectado"),
    "fn": ("red", 2.0, "solid", "no detectado (FN)"),
    "tp": ("dodgerblue", 1.5, "dash", "correcta (TP)"),
    "fp": ("yellow", 1.5, "dash", "falsa alarma (FP)"),
}


def _gray(img):
    """(H, W), (1, H, W) o (3, H, W) -> (H, W) normalizada a [0, 1]."""
    img = _np(img)
    if img.ndim == 3:
        img = img[0] if img.shape[0] in (1, 3) else img[..., 0]
    return (img - img.min()) / (img.max() - img.min() + 1e-9)


def _rects(boxes, scores=None):
    """Todas las cajas en una sola traza: rectangulos cerrados separados por None."""
    xs, ys, txt = [], [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes[:, :4]):
        xs += [x1, x2, x2, x1, x1, None]
        ys += [y1, y1, y2, y2, y1, None]
        t = f"score {scores[i]:.2f}" if scores is not None else ""
        txt += [t] * 5 + [None]
    return xs, ys, txt


def _add_image(fig, img, dets, gts, row, col, iou_thr, en_leyenda):
    """Dibuja una imagen con sus cajas en el subplot (row, col). Devuelve (TP, FP, FN)."""
    gts = _np(gts).reshape(-1, 4)
    dets, is_tp, matched = match_image(dets, gts, iou_thr)

    fig.add_trace(go.Heatmap(z=_gray(img), colorscale="gray", showscale=False,
                             hoverinfo="skip"), row=row, col=col)
    grupos = {"gt": (gts[matched], None), "fn": (gts[~matched], None),
              "tp": (dets[is_tp], dets[is_tp, 4]), "fp": (dets[~is_tp], dets[~is_tp, 4])}
    for k, (cajas, scores) in grupos.items():
        if len(cajas) == 0:
            continue
        color, ancho, trazo, nombre = ESTILO[k]
        xs, ys, txt = _rects(cajas, scores)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", text=txt,
            hoverinfo="text" if scores is not None else "skip",
            line=dict(color=color, width=ancho, dash=trazo),
            name=nombre, legendgroup=k, showlegend=k not in en_leyenda), row=row, col=col)
        en_leyenda.add(k)
    return int(is_tp.sum()), int((~is_tp).sum()), int((~matched).sum())


def show_grid(images, all_dets, all_gts, n_cols=4, iou_thr=0.5, save_path=None):
    """Varias imagenes en una grilla, como la Fig. 18 de [12].
    Pasando el mouse sobre una deteccion se ve su score."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[" "] * n,
                        horizontal_spacing=0.02, vertical_spacing=0.06)

    en_leyenda = set()
    for i, (img, d, g) in enumerate(zip(images, all_dets, all_gts)):
        tp, fp, fn = _add_image(fig, img, d, g, i // n_cols + 1, i % n_cols + 1,
                                iou_thr, en_leyenda)
        fig.layout.annotations[i].text = f"TP={tp}  FP={fp}  FN={fn}"
        fig.layout.annotations[i].font.size = 11

    # imagen: eje y hacia abajo, pixeles cuadrados, sin ejes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    for i in range(1, n_rows * n_cols + 1):
        s = "" if i == 1 else str(i)
        fig.layout[f"yaxis{s}"].scaleanchor = f"x{s}"

    fig.update_layout(width=260 * n_cols, height=270 * n_rows + 60,
                      template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.02 + 0.3 / n_rows, x=0))
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_pr_curves(results, save_path=None, puntos=None):
    """results: dict {nombre: salida de metrics.evaluate}. Una curva por experimento,
    util para la tabla de ablacion (Fig. 3a de ShipDeNet-20).
    puntos: dict {nombre: (recall, precision)} para marcar el punto de operacion."""
    fig = go.Figure()
    for name, r in results.items():
        rec, prec = r["curve"]
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", legendgroup=name,
                                 name=f"{name} (AP50={r['AP50']:.3f})"))
        if puntos and name in puntos:
            R, P = puntos[name]
            fig.add_trace(go.Scatter(x=[R], y=[P], mode="markers", legendgroup=name,
                                     marker=dict(size=10, symbol="x"), showlegend=False,
                                     hovertemplate=f"{name}<br>R=%{{x:.3f}}  P=%{{y:.3f}}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Recall", range=[0, 1]),
                      yaxis=dict(title="Precision", range=[0, 1.02]),
                      width=560, height=440, template="plotly_white",
                      hovermode="closest", legend=dict(x=0.02, y=0.02))
    if save_path:
        fig.write_html(save_path)
    return fig