"""
Post-proceso: filtrado por score + Non-Maximum Suppression (Step 6 de [12]).

Entrada: la salida de decode(), (B, N, 5) con [cx, cy, w, h, score] normalizado.
Salida: lista de largo B; cada elemento es (n, 5) con [x1, y1, x2, y2, score]
en pixeles de la imagen de 160x160.
"""

import torch

from utils.decode import cxcywh_to_xyxy

try:
    from torchvision.ops import nms as _tv_nms
except ImportError:          # torchvision no instalado: se usa la version propia
    _tv_nms = None


def box_iou_xyxy(a, b):
    """IoU entre cada caja de a (N, 4) y cada caja de b (M, 4), formato xyxy. -> (N, M)"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms_simple(boxes, scores, iou_thr):
    """NMS explicito, para entender el algoritmo (torchvision hace lo mismo, mas rapido).
    1. Tomar la caja de mayor score y conservarla.
    2. Descartar todas las que se superponen con ella mas que iou_thr.
    3. Repetir con las que quedan."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou_xyxy(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long)


def postprocess(decoded, score_thr=0.5, iou_thr=0.5, max_det=100):
    """decoded: (B, N, 5). Devuelve lista de (n, 5) [x1, y1, x2, y2, score].

    score_thr = 0.5 e iou_thr = 0.5 son los umbrales del paper (Sec. 5).
    Para calcular el AP (curva P-R completa) usar un score_thr bajo, p. ej. 0.001.
    """
    results = []
    for det in decoded:                          # una imagen a la vez
        det = det[det[:, 4] >= score_thr]
        if det.numel() == 0:
            results.append(torch.zeros(0, 5, device=decoded.device))
            continue
        boxes = cxcywh_to_xyxy(det[:, :4])
        scores = det[:, 4]
        if _tv_nms is not None:
            keep = _tv_nms(boxes, scores, iou_thr)
        else:
            keep = nms_simple(boxes, scores, iou_thr)
        keep = keep[:max_det]
        results.append(torch.cat([boxes[keep], scores[keep, None]], dim=1))
    return results
