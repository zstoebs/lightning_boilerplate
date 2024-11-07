import os
from os import path
from typing import Literal, List, Callable
import numpy as np
from abc import ABC, abstractmethod

from lightning.pytorch.callbacks import Callback

from ..utils import save_imgs, apply_transforms
from ..data import ABCDataset
from ..transforms import ABCTransform


class ABCImageLogger(ABC, Callback):
    def __init__(
        self,
        view_transforms: List[ABCTransform] = [],
        save_freq: int = 1,
        best_only=True,
        disabled=False,
        **kwargs,
    ):
        super().__init__()
        self.view_transforms = view_transforms
        self.disabled = disabled
        self.save_freq = save_freq
        self.val_idx = 0
        self.image_shape = None

        self.best_only = best_only
        self.min_loss = np.inf

    def _save_imgs(
        self,
        imgs: List[np.ndarray],
        save_path: os.PathLike,
        labels: List[str] = [],
        metrics: dict = {},
        method: Literal["pil", "plt", "cv2"] = "pil",
        **plotting_kwargs,
    ):
        save_imgs(
            imgs,
            save_path,
            labels=labels,
            metrics=metrics,
            method=method,
            **plotting_kwargs,
        )

    def _apply_transforms(self, imgs: List[np.ndarray]):
        return apply_transforms(imgs, self.view_transforms)

    def setup(self, trainer, pl_module, stage):
        """Called when fit, validate, test, or predict begins"""

        dataset = trainer.datamodule.datasets[stage]
        assert isinstance(
            dataset, ABCDataset
        ), f"Dataset must be derived from ABCDataset but is {type(dataset)}"

        self.orig_img = dataset.image  # MUST OCCUR FOR VALIDATION
        self.image_shape = dataset.input_shape  # MUST OCCUR FOR VALIDATION

        transf_orig = self._apply_transforms([self.orig_img])
        for transf_name in transf_orig:
            save_path = path.join(
                trainer.logger.log_dir, f"original_image_{transf_name}.png"
            )

            transf_imgs, plotting_kwargs = transf_orig[transf_name]
            save_imgs = [img for img in transf_imgs]

            self._save_imgs(save_imgs, save_path, method="pil", **plotting_kwargs)

    @abstractmethod
    def on_validation_end(self, trainer, pl_module, **kwargs):
        pass
