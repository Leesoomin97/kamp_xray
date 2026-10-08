from __future__ import annotations

from typing import Any


WEAK_GAUSSIAN_BLUR_CONFIG: dict[str, Any] = {
    "probability": 0.15,
    "kernel_size": 3,
    "sigma_min": 0.10,
    "sigma_max": 0.50,
    "application_scope": "training_images_only_after_resize",
    "bbox_changed": False,
}


def make_weak_gaussian_blur_trainer():
    """Return an Ultralytics detection trainer with train-only weak Gaussian blur.

    The transform is applied to normalized training tensors after the standard
    detection augmentations and resize. Validation uses Ultralytics' separate
    validator preprocessing path and is therefore unchanged.
    """
    import torch
    from torch.nn import functional as F
    from ultralytics.models.yolo.detect import DetectionTrainer

    config = WEAK_GAUSSIAN_BLUR_CONFIG

    class WeakGaussianBlurDetectionTrainer(DetectionTrainer):
        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            images = batch["img"]
            selected = torch.rand(images.shape[0], device=images.device) < config["probability"]
            if not bool(selected.any()):
                return batch

            blurred = images.clone()
            selected_indexes = selected.nonzero(as_tuple=False).flatten().tolist()
            sigmas = torch.empty(len(selected_indexes), device=images.device).uniform_(
                config["sigma_min"], config["sigma_max"]
            )
            coordinates = torch.arange(-1, 2, device=images.device, dtype=images.dtype)
            channels = images.shape[1]
            for index, sigma in zip(selected_indexes, sigmas):
                kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma.square()))
                kernel_1d = kernel_1d / kernel_1d.sum()
                kernel_2d = torch.outer(kernel_1d, kernel_1d)
                kernel = kernel_2d.expand(channels, 1, 3, 3)
                sample = F.pad(images[index : index + 1], (1, 1, 1, 1), mode="reflect")
                blurred[index : index + 1] = F.conv2d(sample, kernel, groups=channels)
            batch["img"] = blurred
            return batch

    return WeakGaussianBlurDetectionTrainer
