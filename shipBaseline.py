import torch
import torch.nn as nn
from models.modules import  FE, FF, SSFP
from models.backbone import Backbone

NUM_ANCHORS = 3
OUT_CH = NUM_ANCHORS * 5   # (tx, ty, tw, th, conf) por anchor = 15



class ShipDeNet(nn.Module):
    def __init__(self, c_in=1, use_ff=True, use_fe=True, use_ssfp=True, out_ch=OUT_CH):
        super().__init__()
        self.use_ff, self.use_fe = use_ff, use_fe
        self.backbone = Backbone(c_in)
        self.ff = FF() if use_ff else None
        self.fe = FE() if use_fe else None
        # canales que llegan a las capas de detección, según los módulos activos
        c32 = (640 if use_fe else 128) + (8 if use_ff else 0)
        c16 = (256 if use_fe else 64) + (16 if use_ff else 0)
        c8 = 96 if use_fe else 32
        self.ssfp = SSFP(c32, c16, c8, out_ch=out_ch, share=use_ssfp)

    def forward(self, x):
        f = self.backbone(x)
        f16, f17 = self.ff(f) if self.use_ff else (None, None)
        if self.use_fe:
            x32, x16, x8 = self.fe(f, f16, f17)
        else:
            x32 = f["s32"][-1] if f16 is None else torch.cat([f["s32"][-1], f16], dim=1)
            x16 = f["s16"][-1] if f17 is None else torch.cat([f["s16"][-1], f17], dim=1)
            x8 = f["s8"][-1]
        return self.ssfp(x32, x16, x8)        # siempre [p32, p16, p8]

        

            

