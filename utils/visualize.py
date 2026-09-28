"""
Visualizacion con los colores del paper:
  verde = ground truth, azul = deteccion correcta,
  rojo = barco no detectado, amarillo = falsa alarma.
Formato de cajas: (x1, y1, x2, y2) en pixeles.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from utils.metrics import _np, match_image


def _add_box(ax, b, color, lw=1.5, ls="-"):
    x1, y1, x2, y2 = b[:4]
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, linewidth=lw, linestyle=ls))


def show_detections(img, dets, gts, iou_thr=0.5, ax=None, show_scores=True, title=None):
    """img: (H, W), (1, H, W) o (3, H, W). dets: (n, 5) xyxy + score. gts: (m, 4) xyxy."""
    img = _np(img)
    if img.ndim == 3:
        img = img[0] if img.shape[0] in (1, 3) else img[..., 0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    gts = _np(gts).reshape(-1, 4)
    dets, is_tp, matched = match_image(dets, gts, iou_thr)

    ax.imshow(img, cmap="gray")
    for g, ok in zip(gts, matched):
        _add_box(ax, g, "limegreen" if ok else "red", lw=1.5 if ok else 2)
    for d, tp in zip(dets, is_tp):
        color = "dodgerblue" if tp else "yellow"
        _add_box(ax, d, color, ls="--")
        if show_scores:
            ax.text(d[0], d[1] - 2, f"{d[4]:.2f}", color=color, fontsize=7)

    ax.set_axis_off()
    ax.set_title(title or f"TP={is_tp.sum()}  FP={(~is_tp).sum()}  FN={(~matched).sum()}",
                 fontsize=9)
    return ax


def show_grid(images, all_dets, all_gts, n_cols=4, iou_thr=0.5, save_path=None):
    """Varias imagenes en una grilla, como la Fig. 18 de [12]."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, d, g in zip(axes, images, all_dets, all_gts):
        show_detections(img, d, g, iou_thr, ax=ax, show_scores=False)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pr_curves(results, save_path=None):
    """results: dict {nombre: salida de metrics.evaluate}. Una curva por experimento,
    util para la tabla de ablacion (Fig. 3a de ShipDeNet-20)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, r in results.items():
        rec, prec = r["curve"]
        ax.plot(rec, prec, label=f"{name} (AP50={r['AP50']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
