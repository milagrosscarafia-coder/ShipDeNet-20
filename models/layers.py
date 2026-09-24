

import torch
import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, c_in, c_out, stride=1, slope=1/5.5):
        super().__init__()
        # Depthwise: one 3x3 filter per channel, does the downsampling
        self.depth_conv = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride,
                                    padding=1, groups=c_in, bias=False)
        
        self.bn1 = nn.BatchNorm2d(c_in) #normalizacion sugerida por ia
        # Pointwise: 1x1 conv that mixes all channels -> c_out
        self.point_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        #El paper usa Leaky-ReLU con α = 5.5, definida como y = x/α para x < 0, lo que 
        #en pytorch es negative_slope = 1/5.5 ≈ 0.18
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depth_conv(x)))
        x = self.act(self.bn2(self.point_conv(x)))
        return x


layer = DSConv(16, 8, stride=2)
x = torch.randn(1, 16, 160, 160)
print(layer(x).shape)                                   # torch.Size([1, 8, 80, 80])
print(sum(p.numel() for p in layer.parameters()))       # 9 + 2 + 8 + 16 = 35