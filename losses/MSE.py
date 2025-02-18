import numpy as np
import torch
from torch import nn
from typing import Tuple


def mse(x: torch.Tensor, dim: Tuple[int,...]=(0,1), **kwargs) -> torch.Tensor:
	N = np.prod([x.shape[d] for d in dim])
	return torch.sum(torch.abs(x)**2) / N


class MSELoss(nn.Module): 
	def __init__(self, dim: Tuple[int,...]=(0,1), **kwargs):
		super().__init__()
		self.dim = dim
	
	def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
		return mse(x, dim=self.dim)
	

def NRMSE(pred: np.ndarray, target: np.ndarray, dim: Tuple[int, ...] = (0,1), **kwargs) -> np.ndarray:
	return np.mean(np.sqrt(np.mean(np.square(np.abs(pred-target)), axis=dim)) / np.sqrt(np.mean(np.square(np.abs(target)), axis=dim)))


class NRMSELoss(nn.Module):
	def __init__(self, dim: Tuple[int,...]=(0,1), **kwargs):
		super().__init__()
		self.dim = dim
		self.mse = MSELoss(dim=dim)
	
	def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
		return torch.mean(torch.sqrt(self.mse(pred-target)) / torch.sqrt(mse(target, dim=self.dim)))
