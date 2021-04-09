''' This file contain common tools shared across different OPE algorithms '''

import torch
from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Union
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.evaluator import Policy
from d3pe.utils.net import MLP, DistributionalCritic

def bc(dataset : OPEDataset, 
       policy : Policy, 
       optim : torch.optim.Optimizer,
       steps : int = 100000,
       verbose : bool = False):

    ''' clone the policy in the dataset '''
    
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

def FQI(dataset : OPEDataset,
        policy : Policy,
        num_steps : int = 500000,
        batch_size : int = 256,
        lr : float = 1e-4,
        weight_decay : float = 1e-5,
        init_critic : Optional[Union[MLP, DistributionalCritic]] = None,
        critic_hidden_features : int = 1024,
        critic_hidden_layers : int = 4,
        critic_type : str = 'distributional',
        atoms : int = 51,
        gamma : float = 0.99,
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
        log : str = None,
        verbose : bool = False,
        *args, **kwargs):

        ''' solve the value function of the policy given the dataset '''

        writer = torch.utils.tensorboard.SummaryWriter(log) if log is not None else None

        min_reward = dataset[:]['reward'].min()
        max_reward = dataset[:]['reward'].max()
        max_value = (1.2 * max_reward - 0.2 * min_reward) / (1 - gamma)
        min_value = (1.2 * min_reward - 0.2 * max_reward) / (1 - gamma)

        policy = deepcopy(policy)
        policy = policy.to(device)

        data = dataset.sample(batch_size)
        if init_critic is not None:
            critic = deepcopy(init_critic)
        else:
            if critic_type == 'mlp':
                critic = MLP(data['obs'].shape[-1] + data['action'].shape[-1], 1, critic_hidden_features, critic_hidden_layers).to(device)
            elif critic_type == 'distributional':
                critic = DistributionalCritic(data['obs'].shape[-1], data['action'].shape[-1], 
                                            critic_hidden_features, critic_hidden_layers,
                                            min_value, max_value, atoms).to(device)

        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)
        target_critic = deepcopy(critic).to(device)
        target_critic.requires_grad_(False)

        if verbose:
            counter = tqdm(total=num_steps)

        for t in range(num_steps):
            batch = dataset.sample(batch_size)
            data = to_torch(batch, torch.float32, device=device)
            r = data['reward']
            terminals = data['done']
            o = data['obs']
            a = data['action']

            o_ = data['next_obs']
            a_ = torch.as_tensor(policy.get_action(o_), dtype=torch.float32, device=device)

            if isinstance(critic, MLP):
                q_target = target_critic(torch.cat((o_, a_), -1)).detach()
                current_discount = gamma * (1 - terminals)
                backup = r + current_discount * q_target
                backup = torch.clamp(backup, min_value, max_value) # prevent explosion
                
                q = critic(torch.cat((o, a), -1))
                critic_loss = ((q - backup) ** 2).mean()
            elif isinstance(critic, DistributionalCritic):
                q, p = critic(o, a, with_p=True)
                target_p = target_critic.get_target(o_, a_, r, gamma * (1 - terminals))
                critic_loss = - (target_p * torch.log(p + 1e-8)).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if writer is not None:
                writer.add_scalar('q', scalar_value=q.mean().item(), global_step=t)
        
            if t % 100 == 0:
                with torch.no_grad():
                    target_critic.load_state_dict(critic.state_dict())

            if verbose:
                counter.update(1)

        return critic