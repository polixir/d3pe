from typing import Optional, Union
import torch
from torch.functional import F

from typing import *

def soft_clamp(x : torch.Tensor, 
               _min : Optional[Union[torch.Tensor, float]] = None, 
               _max : Optional[Union[torch.Tensor, float]] = None,) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x