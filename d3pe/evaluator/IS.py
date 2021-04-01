import torch
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.utils.tools import bc
from d3pe.utils.net import GaussianActor

class ISEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   gamma : float = 0.99,
                   device : str = "cuda" if torch.cuda.is_available() else "cpu",
                   log : str = None,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run IS'
        self.dataset = val_dataset or train_dataset
        assert self.dataset.has_trajectory, 'Important Sampling Evaluator only work with trajectory dataset!'
        self.gamma = gamma
        self.device = device
        self.writer = SummaryWriter(log) if log is not None else None

        ''' clone the behaviorial policy '''
        data = self.dataset[0]
        behavior_policy = GaussianActor(data['obs'].shape[-1], data['action'].shape[-1], 1024, 2).to(self.device)
        behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=1e-3)
        self.behavior_policy = bc(self.dataset, behavior_policy, behavior_policy_optim, 10000)

        self.is_initialized = True

    def __call__(self, policy : Policy) -> float:
        assert self.is_initialized, "`initialize` should be called before call."

        policy = deepcopy(policy)
        policy = policy.to(self.device)

        ''' recover the evaluated policy '''
        recover_dataset = deepcopy(self.dataset)
        obs = recover_dataset.data['obs']
        recovered_action = []
        with torch.no_grad():
            for i in range(obs.shape[0] // 256 + (obs.shape[0] % 256 > 0)):
                recovered_action.append(policy.get_action(obs[i*256:(i+1)*256]))
            recover_dataset.data['action'] = np.concatenate(recovered_action, axis=0)
        data = recover_dataset[0]
        recover_policy = GaussianActor(data['obs'].shape[-1], data['action'].shape[-1], 1024, 2).to(self.device)
        recover_policy_optim = torch.optim.Adam(recover_policy.parameters(), lr=1e-3)
        recover_policy = bc(self.dataset, recover_policy, recover_policy_optim, 10000)

        with torch.no_grad():
            ratios = []
            discounted_rewards = []
            for traj in self.dataset.get_trajectory():
                traj = to_torch(traj, device=self.device)
                behavior_action_dist = self.behavior_policy(traj['obs'])
                behavior_policy_log_prob = behavior_action_dist.log_prob(traj['action']).sum(dim=-1, keepdim=True)
                evaluated_action_dist = recover_policy(traj['obs'])
                evaluated_policy_log_prob = evaluated_action_dist.log_prob(traj['action']).sum(dim=-1, keepdim=True)
                ratio = evaluated_policy_log_prob - behavior_policy_log_prob
                ratio = torch.sum(ratio, dim=0)
                ratios.append(ratio)
                discounted_reward = traj['reward'] * (self.gamma ** torch.arange(traj['reward'].shape[0], device=self.device).unsqueeze(dim=-1))
                discounted_reward = torch.sum(discounted_reward, dim=0)
                discounted_rewards.append(discounted_reward)
            ratios = torch.cat(ratios)
            # ratios = (ratios - ratios.mean()) / ratios.std() # this can prevent dominatation of a single trajectory
            ratios = torch.softmax(ratios, dim=0)
            discounted_rewards = torch.cat(discounted_rewards)
            
        return torch.sum(discounted_rewards * ratios).item()