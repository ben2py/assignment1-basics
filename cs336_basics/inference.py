from .model.Full_Transfomer import Transformer_LM
from dataclasses import dataclass
import torch
from pathlib import Path

from .model.Sampler import softmax_temparature,top_p
from .Tokenizer import Tokenizer
from einops import rearrange
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
class InferenceConfig:
    batch_size:int =  8
    device:str = get_available_device()
    src:Path = Path("Output/checkpoints")


if __name__ == "__main__":
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    vocab_size = model_config.vocab_size
    context_length = model_config.context_length
    d_model = model_config.d_model
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads
    batch_size = inference_config.batch_size
    device = inference_config.device
    src = inference_config.src

    net = Transformer_LM(vocab_size,context_length,num_layers,d_model,num_heads,d_model*8/3)
    state_dict = torch.load(src/"random_whole_OWT/9975")
    net.load_state_dict(state_dict["model"])
    input_text = "When I moved into my first apartment, I had no idea how much money I would spend on"

    root = Path("Output/openwebtext")
    t = Tokenizer.from_files(root/"vocab.pkl",root/"merges.pkl",special_tokens=["<|endoftext|>"])

    encoded = t.encode(input_text)
    last_token = torch.tensor([[encoded[-1]]])
    print(last_token.shape)
    encoded = torch.tensor(encoded) #shape l
    new_input = rearrange(encoded,"... -> 1 ...")
    cnt = 0
    while last_token.item()!=256 and len(new_input[0]) < vocab_size and cnt <256 :

        logits = net(new_input) #shape B L vocab_size
        next_logit = logits[:,-1,:] #shape (1,1,vocab_size)

        next_logit_softmax = softmax_temparature(next_logit,-1,1)
        token = top_p(next_logit_softmax,0.9)
        print(new_input.shape)
    
        new_input = torch.cat((new_input,token),dim=-1)

        last_token = token
        
        cnt +=1
    #new_input shape 1 l
    print(t.decode(new_input[0].tolist()))