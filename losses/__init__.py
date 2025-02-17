import numpy as np
import torch
from torch import nn
from typing import Any, Literal, List, Callable, Tuple, Mapping, Optional
from abc import ABC, abstractmethod


class ABCLoss(ABC, nn.Module):
    def __init__(self, 
                 loss_type: Any, 
                 regularizers: List[nn.Module]=[], 
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.loss_type = loss_type  # type of loss
        self.regularizers = regularizers # list of regularizers

    def forward(self, **kwargs) -> torch.Tensor:
        loss = self.loss_fnc(**kwargs)
        
        reg_val = 0.
        for reg in self.regularizers:
            reg_val += reg(**kwargs)

        return loss + reg_val
    
    @abstractmethod
    def set_params(self) -> None:
        pass
    
    @abstractmethod
    def reconstruct_params(self) -> Mapping[str, np.ndarray]:
        pass

    @abstractmethod
    def prepare_input(self) -> Mapping[str, torch.Tensor]:
        pass

    @abstractmethod
    def prepare_output(self) -> Mapping[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def loss_fnc(self, **kwargs) -> torch.Tensor:
        pass


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