from .RoPE import Multihead_Self_Attention
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm
import torch
from torch import nn
from .Embedding import Embedding
class Transformer_Block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.MHA = Multihead_Self_Attention(d_model,num_heads)
        self.swiglu = SwiGLU(d_model)
        self.rmsnorm1 = RMSNorm(d_model,1e-5)
        self.rmsnorm2 = RMSNorm(d_model,1e-5)
    def forward(self,x:torch.Tensor):
        x = x + self.MHA(self.rmsnorm1(x))

        y = x + self.swiglu(self.rmsnorm2(x))
        return y

class Transformer_LM(nn.Module):
    def __init__(self,vocab_size,context_length,num_layers,d_model,num_heads,d_ff):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = nn.Sequential(
            *[Transformer_Block(d_model,num_heads,d_ff) for i in range(num_layers)] 
        )
        self.embedding = Embedding(vocab_size,d_model)
        self.final_norm = RMSNorm(d_model,1e-5)
        self.lm_head = nn.Linear(d_model,vocab_size)
        
    def forward(self,x:torch.Tensor): #shape b l
        
        embedding = self.embedding(x) #shape b l d
        
        x = self.layers(embedding)

        logits = self.final_norm(x)
        y = self.lm_head(logits)

        return y

def cross_entropy(logits,targets):
    #logits shape b vocab_size targets shape b
    #loss = -logsoftmax(oi)[xi+1]
    m = torch.amax(logits,dim=-1,keepdim=True)
    b = targets.shape[0]
    # print(logits[torch.arange(b),targets].unsqueeze(-1).shape)
    # print(logits[targets].shape)
    # print(targets.shape)
    # torch.Tensor 的高级缩影，可以在维度上用一个tensor数组去索引
    return torch.mean(-logits[torch.arange(b),targets].unsqueeze(-1) + m +torch.log(torch.sum(torch.exp(logits - m),dim=-1,keepdim=True)))


# transformer_blk = Transformer_Block(768,12,2048)
# x = torch.randn(10,20,768)
# print(transformer_blk(x).shape)
# print(transformer_blk(x))

# x = torch.arange(100).reshape(10,10)

# lm = Transformer_LM(32000,256,4,768,12,2048)
# res = lm(x)
# print(res.shape)
# logits = torch.randn(10,32000)
# targets = torch.arange(10)

# print(cross_entropy(logits,targets))