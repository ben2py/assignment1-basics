import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model,device=device,dtype = dtype))

    def forward(self,x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)#shape b l d

        res = (x/torch.sqrt(torch.mean(torch.pow(x,2),dim=-1,keepdim=True) + self.eps) )*self.g

        res = res.to(in_dtype)

        return res

# rmsnorm = RMSNorm(768,1e-5)

# x = torch.randn(10,768,dtype = torch.float16)
# print(x.dtype)
# print(rmsnorm(x))