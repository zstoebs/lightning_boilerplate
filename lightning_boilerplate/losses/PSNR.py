import torch as pt
from torch import Tensor
from typing import Callable

from .MSE import mse

def psnr(pred: Tensor, target: Tensor, max_val: float=1.0, reduction: Callable=pt.mean, **kwargs) -> Tensor:
    mse_loss = mse(pred - target, reduction=reduction)
    return 10.0 * pt.log10((max_val ** 2) / mse_loss)
