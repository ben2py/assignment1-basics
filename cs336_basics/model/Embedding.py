import torch
import torch.nn as nn
import math
class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings:int = num_embeddings
        self.num_embedding_dim:int = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Parameter(torch.empty(self.num_embeddings,self.num_embedding_dim))
        std = math.sqrt(2/(self.num_embeddings + self.num_embedding_dim))
        nn.init.trunc_normal_(
            self.embedding,
            mean = 0,
            std = std,
            a = -3*std,
            b = 3*std
        )

    def forward(self,token_ids):
        return self.embedding[token_ids]



# embedding = Embedding(32000,768)
# print(embedding(10).shape)

