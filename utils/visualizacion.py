"""
Visualizacion de resultados de un modelo entrenado.

Uso desde otro script:
    from visualizacion import visualizar
    visualizar(model, val_loader, anchors, vals=5, device=device, nombre="ShipDeNet-20")
"""

import os

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
               n_show=12, out_dir="resultados", score_thr=0.5, show=True):
    """Metricas + grilla de detecciones + curva P-R.

    Guarda en out_dir:
      detecciones.png : n_show imagenes con los colores del paper
                        (verde = GT, azul = correcta, rojo = no detectado, amarillo = falsa alarma)
      curva_pr.png    : curva precision-recall
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
    show_grid(images[:n_show], dets_thr[:n_show], gts[:n_show], n_cols=4,
              save_path=os.path.join(out_dir, "detecciones.png"))

    # ---- curva P-R
    plot_pr_curves({nombre: r}, save_path=os.path.join(out_dir, "curva_pr.png"))

    if show:
        plt.show()
    return r