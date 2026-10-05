# (FF, FE, SSFP)

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from models.layers import DSConv
from models.backbone import Backbone


class FF(nn.Module): 
    def __init__(self, slope =1/5.5):
        super().__init__()

        # L1: 80X80X8 ---> 5X5X128 
        self.layer16 = DSConv(8,8, stride = 16, slope = slope, kernel_size = 17)
        self.layer17 = DSConv(16,16, stride = 4, slope = slope, kernel_size = 5)

    def forward (self,f): 
        f16 = self.layer16(f["f1"])    # (B, 8, 5, 5)
        f17 = self.layer17(f["f3"])    # (B, 16, 10, 10)
        return f16, f17

# bb, ff = Backbone(), FF()
# f = bb(torch.randn(2, 1, 160, 160))
# f16, f17 = ff(f)
# print(f16.shape, f17.shape)                      # (2, 8, 5, 5) (2, 16, 10, 10)
# print(sum(p.numel() for p in ff.parameters()))   # 2408 + 720 = 3128


class FE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, f16=None, f17=None):
        s32 = f["s32"] + ([f16] if f16 is not None else [])
        s16 = f["s16"] + ([f17] if f17 is not None else [])
        x32 = torch.cat(s32, dim=1)          # capas 11-15 (+16): 640 (648) canales, 5x5
        x16 = torch.cat(s16, dim=1)          # capas 7-10 (+17):  256 (272) canales, 10x10
        x8 = torch.cat(f["s8"], dim=1)       # capas 4-6:          96 canales, 20x20
        return x32, x16, x8

class SSFP(nn.Module):
    """Scale share feature pyramid: en cada escala se concatenan las features de las
    tres escalas (llevadas a la misma resolución) y recién ahí va la capa de detección."""
    def __init__(self, c32, c16, c8, out_ch=15, slope=1/5.5, share=True):
        super().__init__()
        self.share = share
        c_in = [c32 + c16 + c8] * 3 if share else [c32, c16, c8]
        self.layer18 = DSConv(c_in[0], out_ch, slope=slope, head=True)   # D_L/32
        self.layer19 = DSConv(c_in[1], out_ch, slope=slope, head=True)   # D_L/16
        self.layer20 = DSConv(c_in[2], out_ch, slope=slope, head=True)   # D_L/8

    def forward(self, x32, x16, x8):
        if self.share:
            up = lambda x, s: F.interpolate(x, scale_factor=s, mode="nearest")
            down = lambda x, s: F.max_pool2d(x, s)
            x32, x16, x8 = (
                torch.cat([x32, down(x16, 2), down(x8, 4)], dim=1),       # todo a 5x5
                torch.cat([up(x32, 2), x16, down(x8, 2)], dim=1),         # todo a 10x10
                torch.cat([up(x32, 4), up(x16, 2), x8], dim=1),           # todo a 20x20
            )
        return [self.layer18(x32), self.layer19(x16), self.layer20(x8)]

# class SSFP (nn.Module):
#     def __init__(self,c32, c16, c8, out_ch=15, slope = 1/5.5, share=True):

#         # los canales de salida no son los del backbone sino los que se tiene luego de aplicarle los distintos modulos a la red,
#         #  por lo que conviene que sean un parametro de la red, debido a que va a depender de si se ponen o no todos los modulos
#         #  de la red.
#         super().__init__()

#     #  se necesitan 3 de downsampling y 3 de upsampling

#         self.layer18 = DSConv(c32, out_ch, slope = slope, head = True)
#         self.layer19 = DSConv(c16, out_ch, slope = slope, head = True )
#         self.layer20 = DSConv(c8, out_ch, slope = slope, head = True )

#         self.share = share

#     def forward(self, x32, x16, x8):

#         d32 = self.layer18(x32)
#         d16 = self.layer19(x16)
#         d8 = self.layer20(x8)

#         if not self.share:
#             return [d32, d16, d8]

#         d32_prima = d32 + F.avg_pool2d(d16, 2) + F.avg_pool2d(d8, 4)          # todo a 5x5
#         d16_prima = d16 + F.interpolate(d32, scale_factor=2) + F.avg_pool2d(d8, 2)   # todo a 10x10
#         d8_prima = d8 + F.interpolate(d16, scale_factor=2) + F.interpolate(d32, scale_factor=4)  # todo a 20x20

#         return [d32_prima, d16_prima, d8_prima]

# class SSFP(nn.Module):
#     def __init__(self, c32, c16, c8, out_ch=15, mid_ch=32, slope=1/5.5, share=True):
#         super().__init__()
#         self.share = share
#         self.layer18 = DSConv(c32, mid_ch, slope=slope)   # características, con BN y activación
#         self.layer19 = DSConv(c16, mid_ch, slope=slope)
#         self.layer20 = DSConv(c8, mid_ch, slope=slope)
#         self.pred32 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)   # predicción lineal, con bias
#         self.pred16 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
#         self.pred8 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

#     def forward(self, x32, x16, x8):
#         d32 = self.layer18(x32)
#         d16 = self.layer19(x16)
#         d8 = self.layer20(x8)
#         if self.share:
#             d32, d16, d8 = (
#                 d32 + F.avg_pool2d(d16, 2) + F.avg_pool2d(d8, 4),
#                 d16 + F.interpolate(d32, scale_factor=2) + F.avg_pool2d(d8, 2),
#                 d8 + F.interpolate(d16, scale_factor=2) + F.interpolate(d32, scale_factor=4),
#             )
#         return [self.pred32(d32), self.pred16(d16), self.pred8(d8)]
        





