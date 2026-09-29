"""
Metricas de evaluacion (Sec. 4.3 de [12] y Sec. III-C de ShipDeNet-20):
  - Precision = TP / (TP + FP)          (Ec. 17 / 5)
  - Recall    = TP / (TP + FN)          (Ec. 18 / 4)
  - mAP       = area bajo la curva P-R  (Ec. 19 / 6), con IoU >= 0.5  (AP50)
  - FPS, cantidad de parametros y tamano del modelo

Las funciones de deteccion trabajan con numpy (aceptan tensores de torch tambien).
Formato de cajas: (x1, y1, x2, y2) en pixeles.
"""

import io
import time

import numpy as np


def _np(x):
    """Tensor de torch o lista -> array de numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def iou_matrix(a, b):
    """IoU entre cada caja de a (N, 4) y de b (M, 4), formato xyxy. -> (N, M)"""
    a, b = _np(a).reshape(-1, 4), _np(b).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(dets, gts, iou_thr=0.5):
    """Empareja las detecciones de UNA imagen con su ground truth.

    Recorre las detecciones de mayor a menor score; cada una es TP si su mejor
    ground truth tiene IoU >= iou_thr y todavia no fue usado. Si no, es FP
    (incluye detecciones duplicadas del mismo barco).

    Devuelve (dets_ordenadas, is_tp, matched_gt): las detecciones 
    ordenadas por score,
    si cada una es TP, y la mascara de GT detectados.
    """
    dets, gts = _np(dets).reshape(-1, 5), _np(gts).reshape(-1, 4)
    order = np.argsort(-dets[:, 4])
    dets = dets[order]
    is_tp = np.zeros(len(dets), dtype=bool)
    matched = np.zeros(len(gts), dtype=bool)

    if len(dets) and len(gts):
        ious = iou_matrix(dets[:, :4], gts)
        for i in range(len(dets)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thr and not matched[j]:
                is_tp[i] = True
                matched[j] = True
    return dets, is_tp, matched


def average_precision(recall, precision):
    """Area bajo la curva P-R con interpolacion en todos
    los puntos (VOC 2010+)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):          # envolvente monotona decreciente
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(all_dets, all_gts, iou_thr=0.5, score_thr=0.5):
    """all_dets: lista (una por imagen) de (n, 5) [x1, y1, x2, y2, score].
    all_gts:  lista (una por imagen) de (m, 4) [x1, y1, x2, y2].

    Para que el AP sea correcto, all_dets debe venir de postprocess con un
    score_thr bajo (p. ej. 0.001). Precision y recall se reportan con 
    score >= score_thr,
    como en las tablas del paper.
    """
    scores, tps = [], []
    n_gt = 0
    for dets, gts in zip(all_dets, all_gts):
        d, tp, _ = match_image(dets, gts, iou_thr)
        scores.append(d[:, 4])
        tps.append(tp)
        n_gt += len(_np(gts).reshape(-1, 4))

    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
    order = np.argsort(-scores)
    scores, tps = scores[order], tps[order]

    ctp = np.cumsum(tps)
    cfp = np.cumsum(~tps)
    recall_curve = ctp / max(n_gt, 1)
    precision_curve = ctp / np.maximum(ctp + cfp, 1)
    ap = average_precision(recall_curve, precision_curve)

    sel = scores >= score_thr
    tp = int(tps[sel].sum())
    fp = int(sel.sum() - tp)
    fn = n_gt - tp
    return {
        "AP50": ap,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(n_gt, 1),
        "TP": tp, "FP": fp, "FN": fn, "n_gt": n_gt,
        "curve": (recall_curve, precision_curve),
    }


# ------------------------------------------------ velocidad y tamano del modelo

def measure_fps(model, img_size=160, c_in=1, n_iter=200, warmup=20, device="cuda"):
    """Tiempo medio por imagen (ms) y FPS con batch = 1, como en el paper."""
    import torch
    model = model.to(device).eval()
    x = torch.randn(1, c_in, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):                     # calentamiento: no se mide
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()               # esperar a que la GPU termine
    ms = (time.perf_counter() - t0) / n_iter * 1000
    return {"time_ms": ms, "FPS": 1000 / ms}


def count_params(model):
    """Parametros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """Tamano en MB de los pesos guardados (state_dict)."""
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024 ** 2
