import torch
from torch import nn
import numpy as np
from pathlib import Path
import random
def data_load(x:np.ndarray,batch_size,context_length,device):
    # print(f"size of dataset is {x.size}")
    # start = random.randint(0,x.size - context_length)
    start = 0
    
    left = x[start:start + batch_size*context_length]
    right = x[start+1:start+batch_size*context_length+1]
    left = left.reshape(batch_size,context_length)
    right = right.reshape(batch_size,context_length)
    left_tensor = torch.tensor(
        left,dtype = torch.long,device=device
    )
    
    right_tensor = torch.tensor(
        right,dtype = torch.long,device=device
    )
    
    return (left_tensor,right_tensor)
# root = Path("Output/Tokenized")
# arr = np.memmap(
#     root / "train-00000-of-00080.bin",
#     dtype = np.uint16,
#     mode = "r",
# )
# print(data_load(arr,10,1000,"mps"))
