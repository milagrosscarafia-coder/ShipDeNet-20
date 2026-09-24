import torch 
import torch.nn as nn 

from layers import DSConv

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


        
    def forward (self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)

        return x