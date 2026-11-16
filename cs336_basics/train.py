from dataclasses import dataclass
import argparse
from .model.Full_Transfomer import Transformer_LM,cross_entropy
from .model.DataLoader import data_load
from .model.checkpoint import save_checkpoint,load_checkpoint
from .model.Optimizer import AdamW,gradient_clipping
import numpy as np
from pathlib import Path
from einops import rearrange
import torch
import time
import random


def get_available_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else: return "cpu"

@dataclass
class ModelConfig:
    vocab_size:int = 32000
    context_length:int = 256
    d_model:int = 768
    num_layers:int = 6
    num_heads:int = 8

@dataclass
class OptimizerConfig:
    beta1:float = 0.9
    beat2:float = 0.95
    eps:float = 1e-6
    lr:float = 3e-4
    weight_decay:float = 0.01

@dataclass
class TrainConfig:
    batch_size:int =  8
    max_step:int = 10000
    max_grad_norm:int = 1
    device:str = get_available_device()
    out:Path = Path("Output/checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type = float,
        default=3e-4
    )
    
    args = parser.parse_args()
    model_config = ModelConfig()
    optimizer_config = OptimizerConfig(
        lr = args.learning_rate
    )
    train_config = TrainConfig()

    vocab_size = model_config.vocab_size
    context_length = model_config.context_length
    d_model = model_config.d_model
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads

    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beat2
    lr = optimizer_config.lr
    weight_decay = optimizer_config.weight_decay
    eps = optimizer_config.eps

    batch_size = train_config.batch_size
    max_steps = train_config.max_step
    max_grad_norm = train_config.max_grad_norm
    device = train_config.device
    out = train_config.out
    out.mkdir(exist_ok=True)

    root = Path("Output/Tokenized")
    root.mkdir(exist_ok=True)

    
    net = Transformer_LM(vocab_size,context_length,num_layers,d_model,num_heads,d_model*8/3)
    net.to(device)
    print(f"running on {device}")

    AdamW = AdamW(net.parameters(),lr,(beta1,beta2),eps,weight_decay)
    
    for i in range(train_config.max_step):
        start = time.perf_counter()
        file = random.choice(list(root.glob("*.bin")))
        arr = np.memmap(
                file,dtype=np.uint16,mode="r"
            )
        
        x,targets = data_load(arr,batch_size,context_length,device)
        logits = net(x) # shape b l d
        
    
        logits = rearrange(
            logits,"b l d -> (b l) d"
        )
        targets = rearrange(
            targets,"b l -> (b l)"
        )

        loss = cross_entropy(logits,targets)
        
        loss.backward()
        gradient_clipping(net.parameters(),max_grad_norm,eps)
        AdamW.step()
        AdamW.zero_grad()
        print(f"step {i}, loss {loss};time consumed is {time.perf_counter() - start}")
        if i%25 == 0 and i!=0:
            dst = out / f"{i}"
            print(f"step {i}: save checkpoint to {dst} loss: {loss}")
            save_checkpoint(net,AdamW,i,dst)
            


    
