import torch
from torch import nn

from d3pe.utils.func import soft_clamp

ACTIVATION_CREATORS = {
    'relu' : lambda dim: nn.ReLU(inplace=True),
    'elu' : lambda dim: nn.ELU(),
    'leakyrelu' : lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'tanh' : lambda dim: nn.Tanh(),
    'sigmoid' : lambda dim: nn.Sigmoid(),
    'identity' : lambda dim: nn.Identity(),
    'prelu' : lambda dim: nn.PReLU(dim),
    'gelu' : lambda dim: nn.GELU(),
    'swish' : lambda dim: Swish(),
}

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    r"""
        Multi-layer Perceptron
        Inputs:
            in_features : int, features numbers of the input
            out_features : int, features numbers of the output
            hidden_features : int, features numbers of the hidden layers
            hidden_layers : int, numbers of the hidden layers 
            norm : str, normalization method between hidden layers, default : None 
            hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu' 
            output_activation : str, activation function used in output layer, default : 'identity' 
    """
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, hidden_layers : int, 
                 norm : str = None, 
                 hidden_activation : str = 'leakyrelu', 
                 output_activation : str = 'identity'):
        super(MLP, self).__init__()

        hidden_activation_creator = ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creator(hidden_features))
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out

class GaussianActor(torch.nn.Module):
    def __init__(self,
                 obs_dim : int,
                 action_dim : int,
                 features : int,
                 layers : int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.features = features
        self.layers = layers

        self.backbone = MLP(obs_dim, 2 * action_dim, features, layers)

        self.max_logstd = nn.Parameter(torch.ones(action_dim) * 0, requires_grad=True)
        self.min_logstd = nn.Parameter(torch.ones(action_dim) * -10, requires_grad=True)

    def forward(self, obs):
        output = self.backbone(obs)
        mu, log_std = torch.chunk(output, 2, dim=-1)
        log_std = soft_clamp(log_std, self.min_logstd, self.max_logstd)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

class DistributionalCritic(torch.nn.Module):
    def __init__(self, 
                 obs_dim : int, 
                 action_dim : int, 
                 features : int, 
                 layers : int, 
                 min_value : int, 
                 max_value : int,
                 atoms : int = 51) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.atoms = atoms

        self.net = MLP(obs_dim + action_dim, atoms, features, layers)

        self.register_buffer('z', torch.linspace(min_value, max_value, atoms))
        self.delta_z = (max_value - min_value) / (atoms - 1)

    def forward(self, obs, action, with_q=False):
        obs_action = torch.cat([obs, action], dim=-1)
        logits = self.net(obs_action)
        p = torch.softmax(logits, dim=-1)
        if with_q:
            q = torch.sum(p * self.z, dim=-1, keepdim=True)
            return p, q
        else:
            return p

    @torch.no_grad()
    def get_target(self, obs, action, reward, discount):
        p = self(obs, action) # [*B, N]

        # shift the atoms by reward
        target_z = reward + discount * self.z # [*B, N]
        target_z = torch.clamp(target_z, self.min_value, self.max_value) # [*B, N]

        # reproject the value to the nearby atoms
        target_z = target_z.unsqueeze(dim=-1) # [*B, N, 1]
        distance = torch.abs(target_z - self.z) # [*B, N, N]
        ratio = torch.clamp(1 - distance / self.delta_z, 0, 1) # [*B, N, N]
        target_p = torch.sum(p.unsqueeze(dim=-1) * ratio, dim=-2) # [*B, N]

        return target_p