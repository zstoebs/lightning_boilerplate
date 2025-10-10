from abc import ABC, abstractmethod
from typing import Union
import torch
import torch.nn as nn


class ABCTransform(ABC, nn.Module):
    def __init__(self, **plotting_kwargs):
        self.plotting_kwargs = plotting_kwargs
    
    @abstractmethod
    def transform_fnc(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_fnc(x)