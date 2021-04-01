''' This file contain common tools shared across different OPE algorithms '''

import torch
from tqdm import tqdm
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.evaluator import Policy

def bc(dataset : OPEDataset, 
       policy : Policy, 
       optim : torch.optim.Optimizer,
       steps : int = 100000,
       verbose : bool = False):

    device = next(policy.parameters()).device
    
    if verbose: timer = tqdm(total=steps)

    for _ in range(steps):
        data = dataset.sample(256)
        data = to_torch(data, device=device)
        action_dist = policy(data['obs'])
        loss = - action_dist.log_prob(data['action']).mean()
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        if verbose: timer.update(1)

    return policy