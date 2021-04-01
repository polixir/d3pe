import torch
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from d3pe.evaluator import Evaluator, Policy
from d3pe.utils.data import OPEDataset, to_torch
from d3pe.utils.tools import bc
from d3pe.utils.net import MLP, GaussianActor, DistributionalCritic

class FQEEvaluator(Evaluator):
    def initialize(self, 
                   train_dataset : OPEDataset = None, 
                   val_dataset : OPEDataset = None, 
                   pretrain : bool = False, 
                   critic_hidden_features : int = 1024,
                   critic_hidden_layers : int = 4,
                   critic_type : str = 'distributional',
                   atoms : int = 51,
                   gamma : float = 0.99,
                   device : str = "cuda" if torch.cuda.is_available() else "cpu",
                   log : str = None,
                   *args, **kwargs):
        assert train_dataset is not None or val_dataset is not None, 'you need to provide at least one dataset to run FQE'
        self.dataset = val_dataset or train_dataset
        self.pretrain = pretrain
        self.critic_hidden_features = critic_hidden_features
        self.critic_hidden_layers = critic_hidden_layers
        self.critic_type = critic_type
        self.atoms = atoms
        self.gamma = gamma
        self.device = device
        self.writer = SummaryWriter(log) if log is not None else None

        self.min_reward = self.dataset[:]['reward'].min()
        self.max_reward = self.dataset[:]['reward'].max()
        self.max_value = (1.2 * self.max_reward - 0.2 * self.min_reward) / (1 - self.gamma)
        self.min_value = (1.2 * self.min_reward - 0.2 * self.max_reward) / (1 - self.gamma)

        if self.pretrain:
            '''implement a base value function here'''

            # clone the behaviorial policy
            data = self.dataset[0]
            policy = GaussianActor(data['obs'].shape[-1], data['action'].shape[-1], 1024, 2).to(self.device)
            optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
            policy = bc(self.dataset, policy, optim, steps=10000)

            policy.get_action = lambda x: policy(x).mean
            self.init_critic = self.train_estimator(policy, num_steps=100000)
        else:
            self.init_critic = None

        self.is_initialized = True

    def __call__(self, policy : Policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before call."

        policy = deepcopy(policy)
        policy = policy.to(self.device)

        if self.pretrain:
            critic = self.train_estimator(policy, self.init_critic, num_steps=250000)
        else:
            critic = self.train_estimator(policy, num_steps=500000)

        if self.dataset.has_trajectory:
            data = self.dataset.get_initial_states()
            obs = data['obs']
            obs = torch.tensor(obs).float()
            batches = torch.split(obs, 256, dim=0)
        else:
            batches = [torch.tensor(self.dataset.sample(256)['obs']).float() for _ in range(100)]

        estimate_q0 = []
        with torch.no_grad():
            for o in batches:
                o = o.to(self.device)
                a = torch.as_tensor(policy.get_action(o)).to(o)
                if self.critic_type == 'mlp':
                    init_sa = torch.cat((o, a), -1).to(self.device)
                    estimate_q0.append(critic(init_sa).cpu())
                elif self.critic_type == 'distributional':
                    p, q = critic(o, a, with_q=True)
                    estimate_q0.append(q.cpu())
        estimate_q0 = torch.cat(estimate_q0, dim=0)

        return estimate_q0.mean().item()

    def train_estimator(self,
                        policy,
                        init_critic=None,
                        num_steps=500000,
                        batch_size=256,
                        verbose=False):

        data = self.dataset.sample(batch_size)
        if self.critic_type == 'mlp':
            critic = MLP(data['obs'].shape[-1] + data['action'].shape[-1], 1, self.critic_hidden_features, self.critic_hidden_layers).to(self.device)
        elif self.critic_type == 'distributional':
            critic = DistributionalCritic(data['obs'].shape[-1], data['action'].shape[-1], 
                                          self.critic_hidden_features, self.critic_hidden_layers,
                                          self.min_value, self.max_value, self.atoms).to(self.device)
        if init_critic is not None: critic.load_state_dict(init_critic.state_dict())
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, weight_decay=1e-5)
        target_critic = deepcopy(critic).to(self.device)
        target_critic.requires_grad_(False)

        if verbose:
            counter = tqdm(total=num_steps)

        for t in range(num_steps):
            batch = self.dataset.sample(batch_size)
            data = to_torch(batch, torch.float32, device=self.device)
            r = data['reward']
            terminals = data['done']
            o = data['obs']
            a = data['action']

            o_ = data['next_obs']
            a_ = torch.as_tensor(policy.get_action(o_), dtype=torch.float32, device=self.device)

            if self.critic_type == 'mlp':
                q_target = target_critic(torch.cat((o_, a_), -1)).detach()
                current_discount = self.gamma * (1 - terminals)
                backup = r + current_discount * q_target
                backup = torch.clamp(backup, self.min_value, self.max_value) # prevent explosion
                
                q = critic(torch.cat((o, a), -1))
                critic_loss = ((q - backup) ** 2).mean()
            elif self.critic_type == 'distributional':
                p, q = critic(o, a, with_q=True)
                target_p = target_critic.get_target(o_, a_, r, self.gamma * (1 - terminals))
                critic_loss = - (target_p * torch.log(p + 1e-8)).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar('q', scalar_value=q.mean().item(), global_step=t)
        
            if t % 100 == 0:
                with torch.no_grad():
                    target_critic.load_state_dict(critic.state_dict())

            if verbose:
                counter.update(1)

        return critic