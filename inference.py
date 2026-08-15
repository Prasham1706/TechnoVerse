"""Common inference helpers copied into every trained member artifact folder."""

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from model import build_model, load_model, predict


def preprocess_array(array: np.ndarray, stats: Mapping[str, Any]) -> torch.Tensor:
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


def postprocess_tensor(tensor: torch.Tensor, stats: Mapping[str, Any], clip_for_storage: bool = True) -> np.ndarray:
    if tuple(tensor.shape) != (1, 1, 256, 256):
        raise ValueError(f"Expected [1,1,256,256], got {tuple(tensor.shape)}")
    mean = float(stats["normalization"]["mean"])
    std = float(stats["normalization"]["std"])
    array = (tensor.detach().float().cpu()[0, 0] * std + mean).numpy()
    if clip_for_storage:
        # Clipping is used only for storage/display, never in reported PSNR/SSIM.
        array = np.clip(array, float(stats["target_min"]), float(stats["target_max"]))
    return array.astype(np.float32, copy=False)


def restore_npy(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[str, Dict[str, Any]]:
    checkpoint = Path(checkpoint_path)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    stats = payload["normalization_stats"]
    model = load_model(str(checkpoint), device)
    lr = preprocess_array(np.load(input_path, allow_pickle=False), stats)
    sr = predict(model, lr, device)
    result = postprocess_tensor(sr, stats, clip_for_storage=True)
    output = Path(output_path)
    if output.suffix.lower() != ".npy":
        output = output.with_suffix(".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    return str(output), stats


def restore_directory(
    checkpoint_path: str,
    input_directory: str,
    output_directory: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Load one model once and restore every six-digit competition .npy file."""
    checkpoint = Path(checkpoint_path)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    stats = payload["normalization_stats"]
    model = load_model(str(checkpoint), device)
    inputs = sorted(
        path for path in Path(input_directory).glob("*.npy")
        if len(path.stem) == 6 and path.stem.isdigit()
    )
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    wall_started = __import__("time").perf_counter()
    for path in inputs:
        lr = preprocess_array(np.load(path, allow_pickle=False), stats)
        sr = predict(model, lr, device)
        if not torch.isfinite(sr).all():
            raise FloatingPointError(f"Non-finite output for {path.name}")
        np.save(output_root / path.name, postprocess_tensor(sr, stats, clip_for_storage=True))
    elapsed = __import__("time").perf_counter() - wall_started
    return {
        "number_of_images": len(inputs),
        "total_inference_time_seconds": elapsed,
        "mean_inference_time_ms": 1000.0 * elapsed / len(inputs) if inputs else None,
        "model_used": payload.get("model_name"),
        "checkpoint_path": str(checkpoint),
        "output_directory": str(output_root),
        "normalization": stats["normalization"],
    }
