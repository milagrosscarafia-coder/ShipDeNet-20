"""
Loss de ShipDeNet-20 (Ec. 22-25 de [12]: Zhang et al., Remote Sens. 2019).

Adaptada al proyecto:
  - targets en el formato del dataset: [x, y, w, h] en pixeles de 160x160,
    con (x, y) = esquina superior izquierda (COCO).
  - anchors calculados con anchors.py (best_of_runs), ordenados de chico a grande.
  - 15 o 18 canales por cabeza (vals = 5 o 6).

La red devuelve [p32, p16, p8]:
    p32: (B, 3*vals, 5, 5)    p16: (B, 3*vals, 10, 10)    p8: (B, 3*vals, 20, 20)
"""

import torch
import torch.nn as nn

IMG = 160 # dimension de la imagen 
STRIDES = [32, 16, 8] # mismo orden que [p32 (L/32), p16 (L/16), p8 (L/8)]
ANCHOR_GROUPS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]   # grilla gruesa -> anchors grandes


# ---------------------------------------------------------------- utilidades

def to_cxcywh_norm(b, img=IMG):
    """[x, y, w, h] en pixeles (esquina sup. izq.) -> (cx, cy, w, h) normalizado a [0, 1]."""

    # En una imagen es al revés que en matematica: "arriba" es y chica, porque 
    # las filas se numeran de arriba hacia abajo (fila 0 arriba, fila 159 abajo).
    cx = (b[:, 0] + b[:, 2] / 2) / img
    cy =(b[:, 1] + b[:, 3] / 2) / img
    return torch.stack([cx, cy, b[:, 2] / img, b[:, 3] / img], dim=1)


def shape_iou(wh, anchors):
    """IoU entre una caja (w, h) y cada anchor (K, 2), todas centradas en el origen."""

    inter = torch.min(wh[0],anchors[:,0]) * torch.min(wh[1], anchors[ :, 1])

    area_b = wh[0] * wh[1]
    area_c = anchors[:,0] * anchors[:,1]
    union = area_b + area_c - inter
    iou = inter / union
    return iou


def box_iou(a, b, eps=1e-9):
    """IoU elemento a elemento entre cajas (..., 4) en formato (cx, cy, w, h), con posicion."""

    a1, a2 = a[..., :2] - a[..., 2:] / 2, a[..., :2] + a[..., 2:] / 2
    b1, b2 = b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2
    wh = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
    return inter / (union + eps)


# ------------------------------------------------------------- build_targets

def build_targets(targets, anchors, batch_size):
    """Traduce las cajas del ground truth a tensores con la forma de la salida de la red.

    targets: lista de largo B; targets[b] es (n_b, 4) en [x, y, w, h] pixeles de 160x160.
    anchors: (9, 2) en pixeles de 160x160, ordenados de chico a grande.

    Devuelve una lista (una entrada por escala, orden [L/32, L/16, L/8]) de dicts con:
      obj: (B, 3, S, S)     P_cell(ship) por (celda, anchor)          -> Ec. 13
      txy: (B, 3, S, S, 2)  offset del centro dentro de la celda      -> Ec. 22
      box: (B, 3, S, S, 4)  caja GT (cx, cy, w, h) normalizada        -> Ec. 23 y IoU
    """
    anchors = anchors.cpu()
    out = []
    for stride in STRIDES:
        S = IMG // stride #floor division, divedes and rounds the resulto down to de nearest integer
        
        out.append({"obj": torch.zeros(batch_size, 3, S, S),
                    "txy": torch.zeros(batch_size, 3, S, S, 2),
                    "box": torch.zeros(batch_size, 3, S, S, 4)})

    for b, gts in enumerate(targets):
        if len(gts) == 0:
            continue
        for cx, cy, w, h in to_cxcywh_norm(gts.float().cpu()).tolist():
            # 1) anchor ganador entre los 9, por forma
            best = int(shape_iou(torch.tensor([w * IMG, h * IMG]), anchors).argmax())
            k = next(i for i, g in enumerate(ANCHOR_GROUPS) if best in g)   # escala
            a = ANCHOR_GROUPS[k].index(best)                                # slot 0, 1 o 2

            # 2) celda que contiene el centro, en esa escala
            S = IMG // STRIDES[k]
            col, row = min(int(cx * S), S - 1), min(int(cy * S), S - 1)

            # 3) escribir los targets en esa direccion [b, a, row, col]
            t = out[k]
            t["obj"][b, a, row, col] = 1.0
            t["txy"][b, a, row, col] = torch.tensor([cx * S - col, cy * S - row])
            t["box"][b, a, row, col] = torch.tensor([cx, cy, w, h])
    return out


# ---------------------------------------------------------------------- loss

class ShipDeNetLoss(nn.Module):
    """loss = alpha*L_xy + beta*L_wh + gamma*L_score                       (Ec. 25)
    con L_score = (1/gamma)*L_obj + L_noobj                                 (Ec. 24)
    => loss = alpha*L_xy + beta*L_wh + L_obj + gamma*L_noobj
    """

    def __init__(self, anchors, vals=6, alpha=5.0, beta=5.0, gamma=0.5):
        super().__init__()
        # buffer: se mueve con model.to(device) pero no se entrena
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float))
        self.vals = vals                  # 6 como el paper (18 canales) o 5 (15 canales)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, preds, targets):
        B, device = preds[0].shape[0], preds[0].device
        tgts = build_targets(targets, self.anchors, B)
        l_xy = l_wh = l_obj = l_noobj = torch.zeros((), device=device)

        for p, t, group in zip(preds, tgts, ANCHOR_GROUPS):
            S = p.shape[-1]
            obj = t["obj"].to(device)
            txy = t["txy"].to(device)
            tbox = t["box"].to(device)

            # (B, 3*vals, S, S) -> (B, 3, S, S, vals)
            p = p.view(B, 3, self.vals, S, S).permute(0, 1, 3, 4, 2)

            # --- decodificacion ---
            pxy = torch.sigmoid(p[..., 0:2])                              # offset en la celda
            anc = (self.anchors[group] / IMG).view(1, 3, 1, 1, 2)
            pwh = anc * torch.exp(p[..., 2:4].clamp(max=4.0))             # w, h normalizados
            ps = torch.sigmoid(p[..., 4])                                 # score

            gy, gx = torch.meshgrid(torch.arange(S, device=device),
                                    torch.arange(S, device=device), indexing="ij")
            grid = torch.stack((gx, gy), dim=-1).view(1, 1, S, S, 2)
            pbox = torch.cat(((pxy + grid) / S, pwh), dim=-1)             # (cx, cy, w, h)

            # Ec. 22: coordenadas del centro
            l_xy = l_xy + (obj * ((pxy - txy) ** 2).sum(-1)).sum()

            # Ec. 23: ancho y alto con raiz cuadrada
            l_wh = l_wh + (obj * ((tbox[..., 2:].sqrt() - pwh.sqrt()) ** 2).sum(-1)).sum()

            # Ec. 24: el score aprende el IoU donde hay barco y 0 donde no
            iou = box_iou(pbox, tbox).detach()
            l_obj = l_obj + (obj * (ps - iou) ** 2).sum()
            l_noobj = l_noobj + ((1 - obj) * ps ** 2).sum()

        loss = (self.alpha * l_xy + self.beta * l_wh
                + l_obj + self.gamma * l_noobj) / B
        parts = {"xy": l_xy.item() / B, "wh": l_wh.item() / B,
                 "obj": l_obj.item() / B, "noobj": l_noobj.item() / B}
        return loss, parts


# --------------------------------------------------------------------- prueba

if __name__ == "__main__":
    anchors = torch.tensor([[9, 12], [12, 25], [17, 12], [21, 45], [27, 17],
                            [36, 64], [50, 25], [59, 115], [105, 45]], dtype=torch.float)
    B, vals = 2, 6
    preds = [torch.randn(B, 3 * vals, 5, 5, requires_grad=True),
             torch.randn(B, 3 * vals, 10, 10, requires_grad=True),
             torch.randn(B, 3 * vals, 20, 20, requires_grad=True)]
    # formato del dataset: [x, y, w, h] en pixeles de 160x160
    targets = [torch.tensor([[77.0, 35.0, 15.0, 30.0],     # barco chico -> L/8
                             [20.0, 90.0, 60.0, 45.0]]),   # barco grande
               torch.zeros(0, 4)]                          # imagen sin barcos

    tg = build_targets(targets, anchors, B)
    for name, t in zip(["L/32", "L/16", "L/8"], tg):
        print(name, "barcos asignados:", int(t["obj"].sum().item()))

    loss, parts = ShipDeNetLoss(anchors, vals=vals)(preds, targets)
    loss.backward()
    print("loss:", loss.item(), parts)
    print("gradiente ok:", all(p.grad is not None for p in preds))