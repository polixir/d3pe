import torch
from abc import ABC

from d3pe.utils.data import OPEDataset

class Policy(ABC):
    def get_action(self, obs : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs : torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

class Evaluator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_initialized = False
    
    def initialize(self, train_dataset : OPEDataset, val_dataset : OPEDataset, *args, **kwargs):
        self.is_initialized = True

    def __call__(self, policy : Policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before calls."
        raise NotImplementedError

