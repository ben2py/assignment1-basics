import torch


def softmax_temparature(x:torch.Tensor,i:int,temparature:int,):
    # x shape vocabsize
    x_max = torch.amax(x,dim=i,keepdim=True)
    exp_x = torch.exp(x)
    denominator = torch.sum(
        exp_x,dim = i,keepdim=True
    )
    return exp_x / denominator
    
def top_p(x:torch.Tensor,p):
    # x shape b vocabsize
    sorted_prob,sorted_index = torch.sort(
        x,dim = -1,decending=True
    )
    new_probs = list()
    init = 0
    for i in range(p.shape[-1]):
        if init >= p:
            break
        init += sorted_prob[i]
        new_probs.append(sorted_index[i])
    new_probs = torch.tensor(new_prob)
    new_probs = softmax_temparature(new_probs,-1,1)
    sampled_position = torch.multinomial(
        new_probs,
        num_samples = 1
    )
    token_id = torch.gather(
        sorted_index,
        dim = -1,
        index = sampled_position
    )
    return token_id
        
