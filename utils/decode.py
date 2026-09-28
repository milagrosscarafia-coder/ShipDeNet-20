"""
Decodificacion de la salida de ShipDeNet-20.

Convierte [p32, p16, p8] (valores crudos de la red) en cajas con score,
usando exactamente la misma parametrizacion que loss.py:
    cx = (sigmoid(t_x) + col) / S        w = anchor_w * exp(t_w)
    cy = (sigmoid(t_y) + fila) / S       h = anchor_h * exp(t_h)
    score = sigmoid(s)
"""

import torch

from loss import ANCHOR_GROUPS, IMG


def decode(preds, anchors, vals=6):
    """preds: [p32, p16, p8], cada uno (B, 3*vals, S, S).
    anchors: (9, 2) en pixeles de 160x160, ordenados de chico a grande.

    Devuelve (B, N, 5) con [cx, cy, w, h, score], coordenadas normalizadas a [0, 1].
    N = 3 * (5*5 + 10*10 + 20*20) = 1575 cajas por imagen.
    """
    B = preds[0].shape[0]
    device = preds[0].device
    anchors = torch.as_tensor(anchors, dtype=torch.float, device=device)
    out = []

    for p, group in zip(preds, ANCHOR_GROUPS):
        S = p.shape[-1]
        p = p.view(B, 3, vals, S, S).permute(0, 1, 3, 4, 2)        # (B, 3, S, S, vals)

        pxy = torch.sigmoid(p[..., 0:2])
        anc = (anchors[group] / IMG).view(1, 3, 1, 1, 2)
        pwh = anc * torch.exp(p[..., 2:4].clamp(max=4.0))
        ps = torch.sigmoid(p[..., 4:5])

        gy, gx = torch.meshgrid(torch.arange(S, device=device),
                                torch.arange(S, device=device), indexing="ij")
        grid = torch.stack((gx, gy), dim=-1).view(1, 1, S, S, 2)

        boxes = torch.cat(((pxy + grid) / S, pwh, ps), dim=-1)     # (B, 3, S, S, 5)
        out.append(boxes.reshape(B, -1, 5))

    return torch.cat(out, dim=1)                                    # (B, 1575, 5)


def cxcywh_to_xyxy(b, img=IMG):
    """(cx, cy, w, h) normalizado -> (x1, y1, x2, y2) en pixeles de img x img."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([(cx - w / 2) * img, (cy - h / 2) * img,
                        (cx + w / 2) * img, (cy + h / 2) * img], dim=-1)


def xywh_to_xyxy(b):
    """Formato del dataset [x, y, w, h] (esquina sup. izq.) -> (x1, y1, x2, y2)."""
    return torch.stack([b[..., 0], b[..., 1],
                        b[..., 0] + b[..., 2], b[..., 1] + b[..., 3]], dim=-1)


def xyxy_to_coco(b, orig_w, orig_h, img=IMG):
    """(x1, y1, x2, y2) en pixeles de 160x160 -> [x, y, w, h] COCO en pixeles de la
    imagen original, para evaluar con pycocotools."""
    sx, sy = orig_w / img, orig_h / img
    return torch.stack([b[..., 0] * sx, b[..., 1] * sy,
                        (b[..., 2] - b[..., 0]) * sx, (b[..., 3] - b[..., 1]) * sy], dim=-1)
