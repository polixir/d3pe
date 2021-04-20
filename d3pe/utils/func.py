import torch
import numpy as np
from torch.functional import F

from typing import Any, List, Optional, Union

from d3pe.evaluator import Evaluator

def soft_clamp(x : torch.Tensor, 
               _min : Optional[Union[torch.Tensor, float]] = None, 
               _max : Optional[Union[torch.Tensor, float]] = None,) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def vector_stack(vectors : List[np.ndarray], padding_value : float):
    ''' stack vectors of scalar with different length '''
    max_length = max([len(vector) for vector in vectors])
    vectors = [np.pad(vector, pad_width=(0, max_length - len(vector)), constant_values=(padding_value)) for vector in vectors]
    return np.stack(vectors)

def get_evaluator_by_name(ope_algo : str) -> Evaluator:
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