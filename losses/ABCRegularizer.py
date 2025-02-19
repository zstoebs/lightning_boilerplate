import numpy as np
import torch
from torch import nn
from typing import Any, Literal, List, Callable, Tuple, Mapping, Optional
from abc import ABC, abstractmethod


class ABCRegularizer(ABC, nn.Module):
    def __init__(self, weight: float = 0.1, 
                 constraints: List[nn.Module]=[],
                 update_weight: bool = False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.weight = nn.Parameter(torch.tensor(weight)) if update_weight else weight
        self.constraints = constraints # to apply to input before computing regularization loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.clone()
        for constraint in self.constraints:
            xi = constraint(xi)
        return self.weight * self.reg_fnc(xi)
    
    @abstractmethod
    def reg_fnc(self, x: torch.Tensor) -> torch.Tensor:
        pass