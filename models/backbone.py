import torch 
import torch.nn as nn 

from models.layers import DSConv

class Backbone(nn.Module):
    def __init__(self, c_in=1, slope=1/5.5):
        super().__init__()
        # L/2: layer 1
        self.layer1 = DSConv(c_in, 8, stride=2, slope=slope)
        # L/4: layers 2-3
        self.layer2 = DSConv(8, 16, stride=2, slope=slope)
        self.layer3 = DSConv(16, 16, stride=1, slope=slope)
        # L/8: layers 4-6
        self.layer4 = DSConv(16, 32, stride=2, slope=slope)
        self.layer5 = DSConv(32, 32, stride=1, slope=slope)
        self.layer6 = DSConv(32, 32, stride=1, slope=slope)


        # L/16: layers 7-10
        self.layer7 = DSConv(32, 64, stride=2, slope=slope)
        self.layer8 = DSConv(64, 64, stride=1, slope=slope)
        self.layer9 = DSConv(64, 64, stride=1, slope=slope)
        self.layer10 = DSConv(64, 64, stride=1, slope=slope)



        # L/32: layers 11-15
        self.layer11 = DSConv(64, 128, stride=2, slope=slope)
        self.layer12 = DSConv(128, 128, stride=1, slope=slope)
        self.layer13 = DSConv(128, 128, stride=1, slope=slope)
        self.layer14 = DSConv(128, 128, stride=1, slope=slope)
        self.layer15 = DSConv(128, 128, stride=1, slope=slope)

        
    def forward(self, x):
        f1 = self.layer1(x)      # (B, 8, 80, 80)    L/2
        f2 = self.layer2(f1)     # (B, 16, 40, 40)   L/4
        f3 = self.layer3(f2)     # (B, 16, 40, 40)
        f4 = self.layer4(f3)     # (B, 32, 20, 20)   L/8
        f5 = self.layer5(f4)
        f6 = self.layer6(f5)
        f7 = self.layer7(f6)     # (B, 64, 10, 10)   L/16
        f8 = self.layer8(f7)
        f9 = self.layer9(f8)
        f10 = self.layer10(f9)
        f11 = self.layer11(f10)  # (B, 128, 5, 5)    L/32
        f12 = self.layer12(f11)
        f13 = self.layer13(f12)
        f14 = self.layer14(f13)
        f15 = self.layer15(f14)

        return {
            "f1": f1,                       # para FF-Module
            "f3": f3,                       # para FF-Module
            "s8": [f4, f5, f6],             # L/8:  para FE-Module
            "s16": [f7, f8, f9, f10],       # L/16: para FE-Module
            "s32": [f11, f12, f13, f14, f15],  # L/32: para FE-Module
        }