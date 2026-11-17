import torch
from torch import nn
from einops import einsum
import math

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        assert int(d_model*8//3) == d_model*8//3
        self.W1 = nn.Parameter(torch.randn(d_model*8//3,d_model)) #shape d_ff d
        self.W3 = nn.Parameter(torch.randn(d_model*8//3,d_model))
        std = math.sqrt(2/(d_model+d_model*8/3))
        nn.init.trunc_normal_(
                    self.W1,
                    mean = 0,
                    std = std,
                    a = -3*std,
                    b = 3*std
                        )            
        nn.init.trunc_normal_(
                    self.W3,
                    mean = 0,
                    std = std,
                    a = -3*std,
                    b = 3*std
                )
        self.silu = SiLU()
    def forward(self,x:torch.Tensor):
        h1 = einsum(
            x,self.W1,
            "... d_model,d_ff d_model -> ... d_ff"
        )#shape b h d_ff
        h2 = einsum(
            x,self.W3,
            "... d_model,d_ff d_model -> ... d_ff"
        )#shape b h d_ff
        return self.silu(h1) * (h2)

class SwiGLU(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        
        assert int(d_model*8/3) == d_model*8/3
        self.glu = GLU(d_model)
        self.W2 = nn.Parameter(torch.randn(d_model,d_model*8//3))
        std = math.sqrt(2/(d_model+d_model*8/3))
        nn.init.trunc_normal_(
                    self.W2,
                    mean = 0,
                    std = std,
                    a = -3*std,
                    b = 3*std
                        )   
    def forward(self,x):
        return einsum(
            self.W2,self.glu(x),
            "d d_ff,... d_ff -> ... d"
        )


# silu = SiLU()
# # x1 = torch.Tensor([1,2,3,4])
# # print(silu(x1))
# glu = GLU(768)
# swiglu = SwiGLU(768)
# # x2 = torch.randn(10,768)
# # print(swiglu(x2))

# x = torch.randn(10,20,768)
# print((silu(x)*x).shape)