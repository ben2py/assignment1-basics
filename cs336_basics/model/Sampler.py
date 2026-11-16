import torch


def softmax_temparature(x:torch.Tensor,i:int,temparature:float):
    # x shape vocabsize
    assert temparature > 0
    x_max = torch.amax(x,dim=i,keepdim=True)
    exp_x = torch.exp((x-x_max)/temparature)
    denominator = torch.sum(
        exp_x,dim = i,keepdim=True
    )
    return exp_x / denominator
    
def top_p(x:torch.Tensor,p):
    # x shape b vocabsize
    sorted_prob,sorted_index = torch.sort(
        x,dim = -1,descending=True
    )
    new_probs = list()
    init = 0
    
    for i in range(x.shape[-1]):
        if init >= p:
            break
        init += sorted_prob[0][i]
        
        new_probs.append(sorted_prob[0][i])
    new_probs = torch.stack(new_probs)
    sampled_position = torch.multinomial(
        new_probs,
        num_samples = 1
    ).unsqueeze(0)
    token_id = torch.gather(
        sorted_index,
        dim = -1,
        index = sampled_position
    )
    return token_id
        
