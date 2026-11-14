import torch
import torch.nn as nn
import math
class Linear(nn.Module):
    def __init__(self,in_features,out_features,device = None,dtype= None):
        super().__init__()
        self.in_features:int = in_features
        self.out_features:int = out_features
        self.device = device
        self.dtype = dtype

        std = math.sqrt(2/(self.in_features + out_features))

        self.W = nn.Parameter(torch.empty(in_features,out_features))
        nn.init.trunc_normal_(
            self.W,
            mean = 0,
            std = std,
            a = -3*std,
            b = 3*std
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.W@x

l1 = Linear(10,20)
input = torch.empty(20,10)
print(l1(input).shape)
