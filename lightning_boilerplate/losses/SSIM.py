import torch as pt
from torch import Tensor
from typing import Callable
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def ssim(pred: Tensor, target: Tensor, data_range: float=1., reduction: Callable | None=pt.mean, **kwargs) -> Tensor:
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range, **kwargs).to(pred.device)
    ssim_value = ssim_metric(pred, target)
    return reduction(ssim_value) if reduction is not None else ssim_value