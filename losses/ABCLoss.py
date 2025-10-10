import numpy as np
import torch
from torch import nn
from typing import Any, Literal, List, Callable, Tuple, Mapping, Optional
from abc import ABC, abstractmethod


class ABCLoss(ABC, nn.Module):
    def __init__(self, 
                 loss_type: Optional[str]=None, 
                 regularizers: List[nn.Module]=[], 
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.loss_type = loss_type if loss_type is not None else self.__class__.__name__  # type of loss
        self.regularizers = regularizers # list of regularizers

    def forward(self, **kwargs) -> torch.Tensor:
        loss = self.loss_fnc(**kwargs)
        
        reg_val = 0.
        for reg in self.regularizers:
            reg_val += reg(**kwargs)

        return loss + reg_val
    
    def __str__(self):
        return self.loss_type 

    def __repr__(self):
        return str(self)
    
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
