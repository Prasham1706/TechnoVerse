"""Inference helpers — web-app version (preprocess + postprocess only)."""

from typing import Any, Mapping

import numpy as np
import torch


def preprocess_array(array: np.ndarray, stats: Mapping[str, Any]) -> torch.Tensor:
    """Normalise one 128×128 float32 array into a [1,1,128,128] model tensor."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape == (128, 128, 1):
        arr = arr[..., 0]
    if arr.shape != (128, 128):
        raise ValueError(f"Expected one 128x128 grayscale LR array, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or Inf.")
    mean = float(stats["normalization"]["mean"])
    std = float(stats["normalization"]["std"])
    return ((torch.from_numpy(np.ascontiguousarray(arr)).float() - mean) / std)[None, None]


def postprocess_tensor(
    tensor: torch.Tensor,
    stats: Mapping[str, Any],
    clip_for_storage: bool = True,
) -> np.ndarray:
    """Denormalise a [1,1,256,256] output tensor back to float32 pixel space."""
    if tuple(tensor.shape) != (1, 1, 256, 256):
        raise ValueError(f"Expected [1,1,256,256], got {tuple(tensor.shape)}")
    mean = float(stats["normalization"]["mean"])
    std = float(stats["normalization"]["std"])
    array = (tensor.detach().float().cpu()[0, 0] * std + mean).numpy()
    if clip_for_storage:
        # Clipping is used only for storage/display, never in reported PSNR/SSIM.
        array = np.clip(array, float(stats["target_min"]), float(stats["target_max"]))
    return array.astype(np.float32, copy=False)
