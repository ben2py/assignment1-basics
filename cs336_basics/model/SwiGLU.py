import torch
from torch import nn
from einops import einsum

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        assert int(d_model*8/3) == d_model*8/3
        self.W1 = nn.Parameter(torch.randn(d_model*8//3,d_model))
        self.W3 = nn.Parameter(torch.randn(d_model*8//3,d_model))
        self.silu = SiLU()
    def forward(self,x:torch.Tensor):
        h1 = einsum(
            x,self.W1,
            "... d_model,d_ff d_model -> ... d_ff"
        )
        h2 = einsum(
            x,self.W3,
            "... d_model,d_ff d_model -> ... d_ff"
        )
        return self.silu(h1) * (h2)

class SwiGLU(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        
        assert int(d_model*8/3) == d_model*8/3
        self.glu = GLU(d_model)
        self.W2 = nn.Parameter(torch.randn(d_model,d_model*8//3))

    def forward(self,x):
        return einsum(
            self.W2,self.glu(x),
            "... d_ff,d_model d_ff -> ... d_model"
        )


silu = SiLU()
x1 = torch.Tensor([1,2,3,4])
print(silu(x1))

swiglu = SwiGLU(768)
x2 = torch.randn(10,768)
print(swiglu(x2))