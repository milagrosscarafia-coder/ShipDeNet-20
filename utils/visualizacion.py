"""
Visualizacion de resultados de un modelo entrenado.

Uso desde otro script:
    from visualizacion import visualizar
    visualizar(model, val_loader, anchors, vals=5, device=device, nombre="ShipDeNet-20")
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import torch

from utils.decode import decode, xywh_to_xyxy
from utils.nms import postprocess
from utils.metrics import evaluate, count_params, model_size_mb
from utils.visualize import show_grid, plot_pr_curves


@torch.no_grad()
def predict_all(model, loader, anchors, vals, device):
    """Corre el modelo sobre todo el loader.
    Devuelve imagenes, detecciones (score >= 0.001) y ground truth; todo en CPU, formato xyxy."""
    model.eval()
    images_all, dets_all, gts_all = [], [], []
    for images, targets in loader:
        preds = model(images.to(device))
        dets = postprocess(decode(preds, anchors, vals), score_thr=0.001)
        images_all += list(images)
        dets_all += [d.cpu() for d in dets]
        gts_all += [xywh_to_xyxy(t.float()) for t in targets]
    return images_all, dets_all, gts_all


def visualizar(model, loader, anchors, vals, device, nombre="ShipDeNet-20",
               n_show=15, out_dir="resultados", score_thr=0.5, show=True):
    """Metricas + grilla de detecciones + curva P-R (Plotly).

    Guarda en out_dir:
      detecciones.html : n_show imagenes con los colores del paper
                         (verde = GT, azul = correcta, rojo = no detectado, amarillo = falsa alarma)
      curva_pr.html    : curva precision-recall, con el punto de operacion (score_thr) marcado
    Devuelve el diccionario de metricas de utils.metrics.evaluate.
    """
    os.makedirs(out_dir, exist_ok=True)
    images, dets, gts = predict_all(model, loader, anchors, vals, device)

    # ---- metricas (AP con todas las detecciones; P y R con score >= score_thr)
    r = evaluate(dets, gts, iou_thr=0.5, score_thr=score_thr)
    print(f"\n{nombre}")
    print(f"  AP50 {r['AP50']:.4f} | Precision {r['precision']:.4f} | Recall {r['recall']:.4f}")
    print(f"  TP {r['TP']}  FP {r['FP']}  FN {r['FN']}  (barcos reales: {r['n_gt']})")
    print(f"  {count_params(model):,} parametros | {model_size_mb(model):.2f} MB")

    # ---- grilla de detecciones (solo score >= score_thr, como en el paper)
    dets_thr = [d[d[:, 4] >= score_thr] for d in dets]
    fig_grid = show_grid(images[:n_show], dets_thr[:n_show], gts[:n_show], n_cols=4,
                         save_path=os.path.join(out_dir, "detecciones.html"))

    # ---- curva P-R, con la cruz en el punto (recall, precision) del umbral usado
    fig_pr = plot_pr_curves({nombre: r}, save_path=os.path.join(out_dir, "curva_pr.html"),
                            puntos={nombre: (r["recall"], r["precision"])})

    if show:
        fig_grid.show()
        fig_pr.show()
    return r




def plotear(tr_losses, val_losses, tr_parts, val_parts, path, desde=0, eval_every=50):
    """Grafica las curvas de loss y AP50, precision y recall en la misma figura (Plotly).
    desde: cantidad de epochs iniciales que no se grafican en la loss.
    eval_every: cada cuántas epochs se calcularon las métricas de val_parts."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Métricas de validación"))

    epochs = list(range(desde + 1, len(tr_losses) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=tr_losses[desde:], name="train", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses[desde:], name="val", mode="lines"), row=1, col=1)

    if val_parts:
        ep_m = [eval_every * (i + 1) for i in range(len(val_parts))]   # epochs reales
        for key, label in [("AP50", "AP50"), ("precision", "precision"), ("recall", "recall")]:
            fig.add_trace(go.Scatter(x=ep_m, y=[p[key] for p in val_parts], name=label,
                                     mode="lines+markers"), row=1, col=2)

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(type="log",title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="valor", range=[0, 1], row=1, col=2)
    fig.update_layout(width=1000, height=420, hovermode="x unified", template="plotly_white")

    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.write_html(path)            # interactivo: zoom, hover, ocultar curvas
    return fig