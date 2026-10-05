"""
Elige el umbral de score que maximiza F1 sobre VALIDACIÓN (sin reentrenar)
y, opcionalmente, evalúa TEST con ese umbral fijo.

Uso (desde la carpeta del repo):
    %run umbral.py          (en el notebook, para ver los gráficos)
    !python umbral.py       (solo imprime y guarda los archivos)

IMPORTANTE: la configuración de abajo tiene que ser la MISMA que la del
entrenamiento (módulos activos, semilla del split, carpeta de resultados).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import plotly.graph_objects as go

from shipBaseline import ShipDeNet
from read_data_set.rdata import SSDD_BBox_coco
from utils.visualizacion import predict_all
from utils.metrics import evaluate, match_image, _np

# ------------------------------------------------------------------ configuración
OUT = r"C:\Users\Usuario\Desktop\5to Semestre\Aprendizaje profundo con vision artificial\ShipDeNet\ShipDeNet-20\SSFP"        # carpeta con best.pt y anchors.pt
USE_FF, USE_FE, USE_SSFP = True, True, True      # igual que en el entrenamiento
VALS, BATCH, SEED = 5, 32, 0
EVAL_TEST = True      # True solo al final del trabajo: el test se mira UNA vez

BASE = os.path.join("Official-SSDD-OPEN", "Official-SSDD-OPEN", "BBox_SSDD", "coco_style")
device = "cuda" if torch.cuda.is_available() else "cpu"


def collate(batch):
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


# ------------------------------------------------------------------ F1 vs umbral
def best_f1_threshold(all_dets, all_gts, iou_thr=0.5):
    """Recorre todos los umbrales posibles y devuelve el de F1 máximo.
    all_dets debe venir con score_thr bajo (predict_all usa 0.001)."""
    scores, tps, n_gt = [], [], 0
    for dets, gts in zip(all_dets, all_gts):
        d, tp, _ = match_image(dets, gts, iou_thr)
        scores.append(d[:, 4])
        tps.append(tp)
        n_gt += len(_np(gts).reshape(-1, 4))
    scores, tps = np.concatenate(scores), np.concatenate(tps)
    order = np.argsort(-scores)
    scores, tps = scores[order], tps[order]

    ctp = np.cumsum(tps)
    P = ctp / np.arange(1, len(tps) + 1)       # precisión si corto en cada score
    R = ctp / max(n_gt, 1)                     # recall si corto en cada score
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)

    i = int(np.argmax(F1))
    return {"thr": float(scores[i]), "F1": float(F1[i]),
            "precision": float(P[i]), "recall": float(R[i]),
            "curves": (scores, P, R, F1)}


def plot_f1(curves, best, path=None):
    """Precisión, recall y F1 en función del umbral de score."""
    scores, P, R, F1 = curves
    fig = go.Figure()
    for y, name in [(P, "precision"), (R, "recall"), (F1, "F1")]:
        fig.add_trace(go.Scatter(x=scores, y=y, name=name, mode="lines"))
    fig.add_vline(x=best["thr"], line_dash="dash",
                  annotation_text=f"F1 máx: umbral {best['thr']:.2f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.5")
    fig.update_layout(xaxis=dict(title="umbral de score", range=[0, 1]),
                      yaxis=dict(title="valor", range=[0, 1.02]),
                      width=650, height=420, template="plotly_white",
                      hovermode="x unified", title="Validación: P, R y F1 vs umbral")
    if path:
        fig.write_html(path)
        fig.write_json(path.replace(".html", ".json"))
    return fig


def fila(nombre, r):
    p, rc = r["precision"], r["recall"]
    f1 = 2 * p * rc / max(p + rc, 1e-9)
    print(f"  {nombre:<22} AP50 {r['AP50']:.4f} | P {p:.4f} | R {rc:.4f} | F1 {f1:.4f} "
          f"| TP {r['TP']} FP {r['FP']} FN {r['FN']}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    # mismo split que en main.py (misma semilla) -> mismas imágenes de validación
    dataset = SSDD_BBox_coco(os.path.join(BASE, "images", "train"),
                             os.path.join(BASE, "annotations", "train.json"), size=160)
    n_val = int(0.2 * len(dataset))
    _, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, collate_fn=collate)

    # modelo y anchors del entrenamiento
    anchors = torch.load(os.path.join(OUT, "anchors.pt"), map_location=device).to(device)
    model = ShipDeNet(c_in=1, use_ff=USE_FF, use_fe=USE_FE, use_ssfp=USE_SSFP,
                      out_ch=3 * VALS).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), map_location=device))
    model.eval()

    # 1) detecciones en validación (con todos los scores)
    with torch.no_grad():
        _, val_dets, val_gts = predict_all(model, val_loader, anchors, VALS, device)

    # 2) umbral de F1 máximo
    best = best_f1_threshold(val_dets, val_gts)
    print(f"\nUmbral óptimo (validación): {best['thr']:.3f}  ->  "
          f"F1 {best['F1']:.4f} | P {best['precision']:.4f} | R {best['recall']:.4f}")

    print("\nValidación con distintos umbrales:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, best["thr"]]:
        fila(f"umbral {thr:.3f}", evaluate(val_dets, val_gts, score_thr=thr))

    fig = plot_f1(best["curves"], best, path=os.path.join(OUT, "umbral_f1.html"))
    with open(os.path.join(OUT, "umbral.json"), "w") as f:
        json.dump({k: best[k] for k in ["thr", "F1", "precision", "recall"]}, f, indent=2)
    print(f"\nGuardado: {OUT}/umbral_f1.html y {OUT}/umbral.json")

    # 3) test: una sola vez, con el umbral elegido en validación (y 0.5 para el paper)
    if EVAL_TEST:
        test_set = SSDD_BBox_coco(os.path.join(BASE, "images", "test"),
                                  os.path.join(BASE, "annotations", "test.json"), size=160)
        test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            _, test_dets, test_gts = predict_all(model, test_loader, anchors, VALS, device)
        print("\nTEST:")
        fila("umbral 0.5 (paper)", evaluate(test_dets, test_gts, score_thr=0.5))
        fila(f"umbral {best['thr']:.3f} (F1 val)", evaluate(test_dets, test_gts, score_thr=best["thr"]))

    try:                       # solo se ve si se corre con %run en el notebook
        fig.show()
    except Exception:
        pass
