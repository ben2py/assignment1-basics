import torch
from torch import nn
from typing import Optional
from collections.abc import Callable,Iterable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,betas,eps,weight_decay):
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)


    def step(self,closure:Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:continue
                grad = p.grad.data
                state = self.state[p]

                t = state.get("t",1)#track the number of parameter update
                exp_avg = state.get("exp_avg",torch.zeros_like(p))
                exp_avg_sq = state.get("exp_avg_sq",torch.zeros_like(p))
                #update the exp_avg & exp_avg_sq
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul(beta2).addcmul_(
                    grad,grad,value = 1-beta2
                )
                #update the parameters
                data = p.data
                data = data - lr*weight_decay*data
                #bias correction
                alpha = lr*math.sqrt(1-beta1**t)/(1-beta2**t) 
                #decoupled weight decay
                p.data = data - alpha*exp_avg/torch.sqrt(exp_avg_sq+eps)

                state[t] = t+1
                
def learning_rate_schedule(t,lr_max,lr_min,t_w,t_c):
    if t < t_w:return lr_max/t_w*t
    elif t > t_c:return lr_min
    else:
        return lr_min + 0.5*(lr_max - lr_min)*(1+cos((t-t_w)/(t_c-t_w)*math.pi)) 
        
def gradient_clipping(params:Iterable[torch.Tensor],max_norm,eps):
    grads = [p.grad for p in params if p.grad is not None]
    
    if grads is None: return 

    total_norm = torch.sqrt(
        sum(torch.sum(g**2) for g in grads)
    )
    grad_clip = min(1,max_norm/(total_norm+eps))

    for g in grads:
        g.mul_(grad_clip)

def perplexity(loss:torch.tensor):
    #loss shpae b l 

    return torch.exp(torch.mean(loss))