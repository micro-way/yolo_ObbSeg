# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import ObbSegModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results

import torch


class ObbSegTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        # overrides["task"] = "segment"
        overrides["task"] = "obb_seg"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return ObbSegModel initialized with specified config and weights."""
        model = ObbSegModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.obb_seg.ObbSegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        if "bboxes_angle" in batch:
            batch_bboxes5 = torch.cat([batch["bboxes"], (batch["bboxes_angle"]).unsqueeze(1)], dim=1)
        else:
            batch_bboxes5 = batch["bboxes"]
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            # batch["bboxes"],
            batch_bboxes5,
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
