import torch
from torch import nn
import numpy as np
from pathlib import Path
def data_load(x,batch_size,context_length,device):
    pass

root = Path("Output/Tokenized")
np.memmap(
    root / "train-00000-of-00080.bin",
    dtype = np.uint16
)