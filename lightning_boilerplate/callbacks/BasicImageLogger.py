from os import path
import numpy as np
from . import ABCImageLogger
from ..models.BasicMLP import BasicMLP


class BasicImageLogger(ABCImageLogger):
    def on_validation_end(self, trainer, pl_module: BasicMLP, **kwargs):
        assert self.image_shape is not None, "Image shape must be set"

        state_fn = trainer.state.fn

        if self.disabled or self.val_idx % self.save_freq != 0:
            self.val_idx += 1
            return

        # get images
        out = pl_module.reconstruct(self.image_shape)
        imgs = list(out.values())
        labels = list(out.keys())

        # compute metrics
        scores = pl_module.scores
        loss = scores["val_loss"]

        # save image data
        if not self.best_only or (self.best_only and loss < self.min_loss):
            self.min_loss = loss
            for img, label in zip(imgs, labels):
                fname = label
                if not self.best_only:
                    fname += f"_e={trainer.current_epoch}_v={self.val_idx}"
                img_path = path.join(trainer.logger.log_dir, fname)
                np.save(img_path, np.squeeze(img))
            
        # transform for view and comparison
        transf_out = self._apply_transforms(imgs)
        del transf_out['raw']
        
        for transf_name in transf_out:
            transf_imgs, plotting_kwargs = transf_out[transf_name]

            # save images
            save_imgs = [img for img in transf_imgs]
            

            fname = (
                "_".join(
                    [
                        state_fn,
                        transf_name,
                        f"e={trainer.current_epoch}_v={self.val_idx}",
                    ]
                )
                + ".png"
            )
            img_path = path.join(trainer.logger.log_dir, fname)
            self._save_imgs(
                save_imgs,
                img_path,
                labels=labels,
                metrics=scores,
                method="plt",
                **plotting_kwargs,
            )

        self.val_idx += 1