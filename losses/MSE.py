import numpy as np
import torch
from torch import nn
from typing import Tuple, Callable, Optional, Union


def mse(x: torch.Tensor, reduction: Callable=torch.mean, **kwargs) -> torch.Tensor:
	return reduction(abs(x)**2)

class MSELoss(nn.Module): 
	def __init__(self, reduction: Callable=torch.mean, **kwargs):
		super().__init__(**kwargs)
		self.reduction = reduction
	
	def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
		return mse(x, reduction=self.reduction)
	
def nrmse(pred: torch.Tensor, target: torch.Tensor, dim: Tuple[int, ...] = (0,1), **kwargs) -> torch.Tensor:
	return torch.sqrt(torch.mean(abs(pred-target)**2, dim=dim)) / torch.sqrt(torch.mean(abs(target)**2, dim=dim))

class NRMSELoss(nn.Module):
	def __init__(self, dim: Tuple[int,...]=(0,1), **kwargs):
		super().__init__()
		self.dim = dim
	
	def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
		return nrmse(pred, target, dim=self.dim)
