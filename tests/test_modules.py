"""
Verificacion de los modulos de ShipDeNet-20: python test_modules.py

Chequea, para cada pieza:
  - formas de las salidas,
  - cantidad de parametros (donde hay un valor esperado),
  - que el gradiente llegue a TODOS los parametros (backward),
  - y el modelo completo con la loss.

Supone:
  backbone.py -> Backbone(c_in)   que devuelve {"f1", "f3", "s8", "s16", "s32"}
  modules.py  -> FF(), FE(), SSFP(c32, c16, c8, out_ch)
"""

import torch
import torch.nn as nn

from backbone import Backbone
from modules import FF, FE, SSFP

B, IMG, C_IN = 2, 160, 1
VALS = 5                      # 5 -> 15 canales por escala; 6 -> 18 como el paper
OUT = 3 * VALS

n_fail = 0


def check(cond, msg):
    global n_fail
    print(("  OK    " if cond else "  FAIL  ") + msg)
    if not cond:
        n_fail += 1


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------ modelo completo

class ShipDeNet20(nn.Module):
    """Arma el modelo con o sin FF / FE, para los experimentos de ablacion (Tabla I).
    SSFP siempre esta, porque contiene las capas de deteccion 18-20."""

    def __init__(self, use_ff=True, use_fe=True, vals=VALS):
        super().__init__()
        self.use_ff, self.use_fe = use_ff, use_fe
        self.backbone = Backbone(C_IN)
        self.ff = FF() if use_ff else None
        self.fe = FE() if use_fe else None
        c32 = (640 if use_fe else 128) + (8 if use_ff else 0)
        c16 = (256 if use_fe else 64) + (16 if use_ff else 0)
        c8 = 96 if use_fe else 32
        self.ssfp = SSFP(c32, c16, c8, out_ch=3 * vals)

    def forward(self, x):
        f = self.backbone(x)
        f16, f17 = self.ff(f) if self.use_ff else (None, None)
        if self.use_fe:
            x32, x16, x8 = self.fe(f, f16, f17)
        else:
            x32 = f["s32"][-1] if f16 is None else torch.cat([f["s32"][-1], f16], dim=1)
            x16 = f["s16"][-1] if f17 is None else torch.cat([f["s16"][-1], f17], dim=1)
            x8 = f["s8"][-1]
        return self.ssfp(x32, x16, x8)          # [p32, p16, p8]


# ---------------------------------------------------------------------- tests

def test_backbone():
    print("\nBackbone")
    bb = Backbone(C_IN)
    f = bb(torch.randn(B, C_IN, IMG, IMG))
    check(f["f1"].shape == (B, 8, 80, 80), f"f1 {tuple(f['f1'].shape)} == (B, 8, 80, 80)")
    check(f["f3"].shape == (B, 16, 40, 40), f"f3 {tuple(f['f3'].shape)} == (B, 16, 40, 40)")
    for key, n, c, s in [("s8", 3, 32, 20), ("s16", 4, 64, 10), ("s32", 5, 128, 5)]:
        ok = len(f[key]) == n and all(t.shape == (B, c, s, s) for t in f[key])
        check(ok, f"{key}: {n} salidas de (B, {c}, {s}, {s})")
    check(n_params(bb) == 103035, f"parametros {n_params(bb)} == 103035")
    return f


def test_ff(f):
    print("\nFF-Module")
    ff = FF()
    f16, f17 = ff(f)
    check(f16.shape == (B, 8, 5, 5), f"f16 {tuple(f16.shape)} == (B, 8, 5, 5)")
    check(f17.shape == (B, 16, 10, 10), f"f17 {tuple(f17.shape)} == (B, 16, 10, 10)")
    check(n_params(ff) == 3128, f"parametros {n_params(ff)} == 3128")
    return f16, f17


def test_fe(f, f16, f17):
    print("\nFE-Module")
    fe = FE()
    check(n_params(fe) == 0, "no tiene parametros")
    for name, args, chans in [("con FF", (f16, f17), (648, 272, 96)),
                              ("sin FF", (None, None), (640, 256, 96))]:
        x32, x16, x8 = fe(f, *args)
        ok = (x32.shape == (B, chans[0], 5, 5) and x16.shape == (B, chans[1], 10, 10)
              and x8.shape == (B, chans[2], 20, 20))
        check(ok, f"{name}: canales {x32.shape[1]}, {x16.shape[1]}, {x8.shape[1]} == {chans}")


def test_ssfp():
    print("\nSSFP-Module")
    ssfp = SSFP(648, 272, 96, out_ch=OUT)
    out = ssfp(torch.randn(B, 648, 5, 5), torch.randn(B, 272, 10, 10), torch.randn(B, 96, 20, 20))
    check(isinstance(out, (list, tuple)) and len(out) == 3, "devuelve 3 salidas")
    for t, s, name in zip(out, [5, 10, 20], ["p32", "p16", "p8"]):
        check(t.shape == (B, OUT, s, s), f"{name} {tuple(t.shape)} == (B, {OUT}, {s}, {s})")
    # la salida debe poder ser negativa (capa de salida lineal, sin activacion)
    check(min(t.min().item() for t in out) < 0, "la salida toma valores negativos (salida lineal)")


def test_full_models():
    print("\nModelo completo (variantes de la Tabla I)")
    for use_ff, use_fe in [(False, False), (True, False), (False, True), (True, True)]:
        name = f"FF={'si' if use_ff else 'no'} FE={'si' if use_fe else 'no'}"
        model = ShipDeNet20(use_ff, use_fe)
        out = model(torch.randn(B, C_IN, IMG, IMG))
        shapes_ok = [tuple(t.shape) for t in out] == [(B, OUT, 5, 5), (B, OUT, 10, 10),
                                                       (B, OUT, 20, 20)]
        sum(t.sum() for t in out).backward()
        grads = [p.grad for p in model.parameters()]
        grad_ok = all(g is not None and torch.isfinite(g).all() for g in grads)
        check(shapes_ok and grad_ok,
              f"{name}: formas ok, gradiente en todos los parametros | {n_params(model):,} params")


def test_with_loss():
    print("\nModelo completo + loss")
    try:
        from loss import ShipDeNetLoss
    except ImportError as e:
        print(f"  (salteado: no se pudo importar loss.py: {e})")
        return
    anchors = torch.tensor([[9, 12], [12, 25], [17, 12], [21, 45], [27, 17],
                            [36, 64], [50, 25], [59, 115], [105, 45]], dtype=torch.float)
    model = ShipDeNet20()
    criterion = ShipDeNetLoss(anchors, vals=VALS)
    targets = [torch.tensor([[77.0, 35.0, 15.0, 30.0]]), torch.zeros(0, 4)]
    loss, parts = criterion(model(torch.randn(B, C_IN, IMG, IMG)), targets)
    loss.backward()
    check(torch.isfinite(loss).item(), f"loss finita: {loss.item():.3f}  {parts}")


if __name__ == "__main__":
    torch.manual_seed(0)
    f = test_backbone()
    f16, f17 = test_ff(f)
    test_fe(f, f16, f17)
    test_ssfp()
    test_full_models()
    test_with_loss()
    print(f"\n{'Todo OK' if n_fail == 0 else f'{n_fail} chequeo(s) fallaron'}")