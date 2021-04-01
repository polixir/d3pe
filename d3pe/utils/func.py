import torch
from torch.functional import F

from typing import Optional, Union

def soft_clamp(x : torch.Tensor, 
               _min : Optional[Union[torch.Tensor, float]] = None, 
               _max : Optional[Union[torch.Tensor, float]] = None,) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def get_evaluator_by_name(ope_algo : str):
    if ope_algo == 'online':
        from d3pe.evaluator.online import OnlineEvaluator
        return OnlineEvaluator
    elif ope_algo == 'fqe':
        from d3pe.evaluator.fqe import FQEEvaluator
        return FQEEvaluator
    elif ope_algo == 'mbope':
        from d3pe.evaluator.mbope import MBOPEEvaluator
        return MBOPEEvaluator
    elif ope_algo == 'is':
        from d3pe.evaluator.IS import ISEvaluator
        return ISEvaluator
    else:
        raise KeyError(f'Algorithm {ope_algo} is not supported!')