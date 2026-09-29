import torch
import torch.nn as nn

class ConvAC(nn.Module):
    def __init__(self,c_in,c_out,stride=1,slope=1/5.5,kernel_size=3, head=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,kernel_size=kernel_size,stride=stride,padding=1,bias=head)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(negative_slope=slope,inplace=True)
        self.head = head

    def forward(self,x):
        x = self.conv(x)
        if not self.head:
            x = self.bn(x)
            x = self.act(x)
        return x
class DSConv(nn.Module):
    def __init__(self, c_in, c_out, stride=1, slope=1/5.5, kernel_size=3, head = False ):
        super().__init__()
        kernel_size=kernel_size
        # Depthwise: one 3x3 filter per channel, does the downsampling
        self.depth_conv = nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride,
                                    padding=1, groups=c_in, bias=False)
        
        self.bn1 = nn.BatchNorm2d(c_in) #normalizacion sugerida por ia
        # Pointwise: 1x1 conv that mixes all channels -> c_outg
        self.point_conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=head)
        #El paper usa Leaky-ReLU con α = 5.5, definida como y = x/α para x < 0, lo que 
        #en pytorch es negative_slope = 1/5.5 ≈ 0.18
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.head = head 
        if head is not True : 
            self.bn2 = nn.BatchNorm2d(c_out)
   

    def forward(self, x):
        x = self.act(self.bn1(self.depth_conv(x)))
        x = self.point_conv(x)
        if not self.head:                         # capas normales: BN + activación
            x = self.act(self.bn2(x))
        return x    


# layer = DSConv(16, 8, stride=2)
# x = torch.randn(1, 16, 160, 160)
# print(layer(x).shape)                                  
# print(sum(p.numel() for p in layer.parameters()))       