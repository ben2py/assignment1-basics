from torch import nn
import torch
from einops import einsum,rearrange,reduce
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta = theta #theta value for RoPE
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        self.device = device

        i = torch.arange(0,d_k,2)
        angles = theta**(-i/d_k)
        positions = torch.arange(max_seq_len)
        angles = einsum(
            positions,angles,
            "i,l->i l"
        )
        self.register_buffer(
            "cos_cache",
            angles.cos(),
            persistent = False
        )# shape max_len d_k//2
        self.register_buffer(
            "sin_cache",
            angles.sin(),
            persistent = False
        )

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        assert x.shape[-1] %2 == 0
        x = rearrange(
            x,
            "... (d two) -> ... d two",two =2
        )#shape b l d//2 2

        x0,x1 = x.unbind(dim=-1) # x0 shape b l d//2
        y0 = x0*cos - x1*sin 
        y1 = x0*sin + x1*cos

        y = torch.stack([y0,y1],dim=-1)
        return rearrange(
            y,
            "... d two -> ... (d two)",
        )

def softmax_last_dim(x:torch.Tensor,i:int):
    #x.shape : B head_dim l d
    x_max = reduce(
        x,"... d -> ... 1","max"
    )
    x = x- x_max
    exp_x = torch.exp(x)
    denominator = (
        e**x,"... d -> ... 1","sum"
    )
    
    return x/sum

def softmax(x:torch.Tensor,i:int):
    x_max = torch.amax(x,dim=i,keepdim=True)
    x = x- x_max
    exp_x = torch.exp(x)
    denominator = torch.sum(
        exp_x,dim=i,keepdim=True
    )
    return exp_x/denominator

def scaled_dot_product_attention(keys:torch.Tensor,queries:torch.Tensor,values,mask):
    
    d_k = keys.shape[-1]

    scores = einsum(
        queries,keys,
        "b ... l d,b ... s d -> b ... l s" 
    )# replicated alpha is not allowed
    scores = scores / math.sqrt(d_k)

    socres_masked = scores.masked_fill(mask,float("-inf"))
    scores_softmax = softmax(socres_masked,-1)
    
    attention = einsum(
        scores_softmax,values,
        "b ... l l,b ... l d -> b ... l d"
    )
    return attention

class Multihead_Self_Attention(nn.Module):
    def __init__(self,d_models,num_heads):
        super().__init__()
        self.d_models = d_models
        self.num_heads = num_heads
        assert d_models % num_heads == 0

        self.d_k = d_models/num_heads
        self.d_v = d_models/num_heads

        self.W_q = nn.Parameter(torch.empty(d_models,d_models))
        self.W_k = nn.Parameter(torch.empty(d_models,d_models))
        self.W_v = nn.Parameter(torch.empty(d_models,d_models))
        self.W_o = nn.Parameter(torch.empty(d_models,d_models))

        

        self.rope = RotaryPositionEmbedding(10000,d_models,max_seq_len=10000)

    def forward(self,x:torch.Tensor):
        #x.shape b l d
        b,l,d = x.shape
        token_positions = torch.arange(l)
        self.rope(x,token_positions)
        queries = einsum(
            x,self.W_q,
            "b l d,d d -> b l d"
        )
        queries = rearrange(
            queries,
            "b l (d h) -> b h l d",
            h = self.num_heads
        )
        keys = einsum(
            x,self.W_k,
            "b l d,d d -> b l d"
        )
        keys = rearrange(
            keys,
            "b l (d h) -> b h l d",
            h = self.num_heads
        )
        values = einsum(
            x,self.W_v,
            "b l d,d d -> b l d"
        )
        values = rearrange(
            values,
            "b l (d h) -> b h l d",
            h = self.num_heads
        )
        mask = torch.triu(torch.ones(l,l),diagonal=1).bool()
        values_updated = scaled_dot_product_attention(queries,keys,values,mask) #shape b h l d_v
        
        updated = rearrange(
            values_updated,
            "b h l d -> b l (d h)"
        )
        return einsum(
            updated,self.W_o,
            "b l d,d d -> b l d"
        )


# rope = RotaryPositionEmbedding(10000,768,10)

# x = torch.randn(10,768)
# x_rotate = rope(x,torch.arange(10))
# print(x_rotate.shape)

# print(torch.linalg.vector_norm(x, dim=-1))
# print(torch.linalg.vector_norm(x_rotate, dim=-1))

# assert torch.allclose(
#     torch.linalg.vector_norm(x, dim=-1),
#     torch.linalg.vector_norm(x_rotate, dim=-1),
#     atol=1e-5
# )


# keys = torch.randn(10,8,10,8)
# queries = torch.randn(10,8,10,8)#b h l d_k

# seq_len = keys.shape[-2]
# values = torch.randn(10,8,10,10) #b h l d_v
# mask = torch.triu(torch.ones(seq_len,seq_len),diagonal =1).bool()

# print(keys.shape)
# print(queries.shape)

# res = scaled_dot_product_attention(keys,queries,values,mask)
# print(res.shape)
# print(res[0][0])


x = torch.randn(2,20,768)# shape b l d
MHA = Multihead_Self_Attention(768,12)

print(MHA(x).shape)