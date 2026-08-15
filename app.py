"""Standalone Gradio deployment for the trained DA-SwinSR checkpoint.

This application performs inference only. It never accepts model checkpoints
from users and it never trains or fine-tunes the model.
"""

from __future__ import annotations

import atexit
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from inference import postprocess_tensor, preprocess_array
from model import load_model, predict


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "weights" / "checkpoint_best.pth"
METRICS_PATH = ROOT / "metadata" / "metrics.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "9eae0b4a5fe9d978dc14603e64cf2ac5b099d98e8ecd02af68b8624ebd06549b"
)
EXPECTED_PARAMETER_COUNT = 568_681

OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="da_swinsr_outputs_"))
atexit.register(lambda: shutil.rmtree(OUTPUT_DIR, ignore_errors=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Trusted checkpoint is missing: {CHECKPOINT_PATH}")
if _sha256(CHECKPOINT_PATH) != EXPECTED_CHECKPOINT_SHA256:
    raise RuntimeError("checkpoint_best.pth failed its SHA-256 integrity check.")

try:
    CHECKPOINT_PAYLOAD = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
except TypeError:
    CHECKPOINT_PAYLOAD = torch.load(CHECKPOINT_PATH, map_location="cpu")

if CHECKPOINT_PAYLOAD.get("model_name") != "Degradation-aware Swin":
    raise ValueError("The bundled checkpoint is not the expected DA-SwinSR model.")
if "normalization_stats" not in CHECKPOINT_PAYLOAD:
    raise KeyError("The checkpoint does not contain normalization_stats.")

STATS = CHECKPOINT_PAYLOAD["normalization_stats"]
NORMALIZATION = STATS.get("normalization", {})
NORMALIZATION_MEAN = float(NORMALIZATION.get("mean", float("nan")))
NORMALIZATION_STD = float(NORMALIZATION.get("std", float("nan")))
TARGET_MIN = float(STATS.get("target_min", 0.0))
TARGET_MAX = float(STATS.get("target_max", 1.0))
if not np.isfinite(
    [NORMALIZATION_MEAN, NORMALIZATION_STD, TARGET_MIN, TARGET_MAX]
).all():
    raise ValueError("The checkpoint contains invalid normalization statistics.")
if NORMALIZATION_STD <= 0 or TARGET_MAX <= TARGET_MIN:
    raise ValueError("The checkpoint normalization range is invalid.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(str(CHECKPOINT_PATH), DEVICE)
PARAMETER_COUNT = sum(parameter.numel() for parameter in MODEL.parameters())
if PARAMETER_COUNT != EXPECTED_PARAMETER_COUNT:
    raise ValueError(
        f"Unexpected parameter count {PARAMETER_COUNT:,}; "
        f"expected {EXPECTED_PARAMETER_COUNT:,}."
    )

with METRICS_PATH.open("r", encoding="utf-8") as stream:
    METRICS = json.load(stream)
TEST_METRICS = METRICS["internal_test"]

# Warm up once so first-use initialization does not dominate displayed latency.
_ = predict(MODEL, torch.zeros((1, 1, 128, 128), dtype=torch.float32), DEVICE)
if DEVICE.type == "cuda":
    torch.cuda.synchronize()


def _display_uint8(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), TARGET_MIN, TARGET_MAX)
    scaled = (clipped - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
    return np.rint(scaled * 255.0).astype(np.uint8)


def _uploaded_path(uploaded_file: str | Path | None) -> Path:
    if uploaded_file is None:
        raise ValueError("Upload one 128x128 grayscale .npy file first.")
    path = Path(uploaded_file)
    if path.suffix.lower() != ".npy":
        raise ValueError("Only .npy input files are accepted.")
    if not path.is_file():
        raise FileNotFoundError("The uploaded file is no longer available.")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Input exceeds the 1 MB upload limit.")
    return path


def _prune_outputs(max_age_seconds: int = 600) -> None:
    cutoff = time.time() - max_age_seconds
    for path in OUTPUT_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def restore_for_demo(uploaded_file: str | Path | None):
    """Validate one LR array, restore it, and return previews/downloads."""
    try:
        _prune_outputs()
        input_path = _uploaded_path(uploaded_file)
        try:
            loaded = np.load(input_path, allow_pickle=False)
        except Exception as error:
            raise ValueError(
                "The input is not a readable, non-pickled NumPy array."
            ) from error

        if loaded.dtype.kind not in "iuf":
            raise TypeError(f"Expected a real numeric array; received {loaded.dtype}.")
        lr_array = np.asarray(loaded, dtype=np.float32)
        if lr_array.shape == (128, 128, 1):
            lr_array = lr_array[..., 0]
        if lr_array.shape != (128, 128):
            raise ValueError(
                f"Expected shape (128, 128); received {lr_array.shape}. "
                "Choose a file from the NoisyLR folder, not GT."
            )
        if not np.isfinite(lr_array).all():
            raise ValueError("Input contains NaN or Inf values.")

        bicubic = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(lr_array))[None, None],
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        )[0, 0].numpy()

        lr_tensor = preprocess_array(lr_array, STATS)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        sr_tensor = predict(MODEL, lr_tensor, DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        prediction_ms = 1000.0 * (time.perf_counter() - started)

        restored = postprocess_tensor(sr_tensor, STATS, clip_for_storage=True)
        if restored.shape != (256, 256) or not np.isfinite(restored).all():
            raise RuntimeError("The model returned an invalid output array.")

        run_id = uuid.uuid4().hex[:10]
        npy_path = OUTPUT_DIR / f"DA_SwinSR_restored_{run_id}.npy"
        png_path = OUTPUT_DIR / f"DA_SwinSR_preview_{run_id}.png"
        np.save(npy_path, restored.astype(np.float32, copy=False))
        Image.fromarray(_display_uint8(restored), mode="L").save(png_path)

        status = f"""
### Restoration complete
- **Input -> output:** 128x128 -> 256x256 grayscale
- **Device:** {DEVICE.type.upper()}
- **Prediction-call latency:** {prediction_ms:.2f} ms
- **Parameters:** {PARAMETER_COUNT:,}
- **Stored output range:** {float(restored.min()):.5f} to {float(restored.max()):.5f}

PSNR, SSIM and LPIPS are not calculated for this upload because no matching
ground-truth image was supplied.
"""
        return (
            _display_uint8(lr_array),
            _display_uint8(bicubic),
            _display_uint8(restored),
            status,
            str(npy_path),
            str(png_path),
        )
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise gr.Error(str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise gr.Error("Restoration failed. Please contact the demo operator.") from error


CSS = """
.gradio-container { max-width: 1420px !important; margin: 0 auto !important; }
#hero { background: linear-gradient(120deg, #08133d, #12337a); color: white;
        padding: 22px 26px; border-radius: 18px; margin-bottom: 16px; }
#evidence { border-left: 5px solid #69f542; padding-left: 14px; }
"""

with gr.Blocks(
    title="DA-SwinSR Semiconductor Restoration Demo",
    analytics_enabled=False,
    delete_cache=(300, 600),
) as demo:
    gr.Markdown(
        """
# DA-SwinSR - Semiconductor Image Restoration and 2x Super-Resolution
**Team research prototype:** noisy 128x128 grayscale input -> restored 256x256 output

`CNN local features -> learned 48-D latent conditioning -> 6 FiLM-Swin blocks -> PixelShuffle 2x -> HR refinement + bicubic skip`
""",
        elem_id="hero",
    )
    gr.Markdown(
        f"""
**Previously measured fixed internal test ({int(TEST_METRICS['num_images'])} held-out paired images;
validation-selected checkpoint; metrics on unclipped inverse-normalized predictions):**
{TEST_METRICS['psnr_mean']:.4f} +/- {TEST_METRICS['psnr_std']:.4f} dB PSNR -
{TEST_METRICS['ssim_mean']:.4f} +/- {TEST_METRICS['ssim_std']:.4f} SSIM -
{TEST_METRICS['lpips_mean']:.4f} +/- {TEST_METRICS['lpips_std']:.4f} LPIPS *(lower is better)*.

These are internal experiment results, not a score for the uploaded image, a public leaderboard,
or a factory-line validation.
""",
        elem_id="evidence",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Upload one noisy 128x128 .npy image",
                file_types=[".npy"],
                file_count="single",
                type="filepath",
            )
            restore_button = gr.Button("Restore and 2x Super-Resolve", variant="primary")
            clear_button = gr.ClearButton()
            status_output = gr.Markdown(
                "Upload a valid `.npy` input, then click **Restore**."
            )
        with gr.Column(scale=1):
            npy_download = gr.File(label="Download restored float32 .npy")
            png_download = gr.File(label="Download display preview .png")
            gr.Markdown(
                "**Responsible-use note:** retain the raw capture. The restored image "
                "is an assisted-viewing research output, not a replacement measurement."
            )

    with gr.Row(equal_height=True):
        lr_preview = gr.Image(
            label="Noisy LR input (display clipped to [0,1])",
            image_mode="L",
            height=330,
        )
        bicubic_preview = gr.Image(
            label="Bicubic 2x reference (display clipped to [0,1])",
            image_mode="L",
            height=330,
        )
        restored_preview = gr.Image(
            label="DA-SwinSR output (storage/display clipped to [0,1])",
            image_mode="L",
            height=330,
        )

    restore_button.click(
        fn=restore_for_demo,
        inputs=[input_file],
        outputs=[
            lr_preview,
            bicubic_preview,
            restored_preview,
            status_output,
            npy_download,
            png_download,
        ],
        concurrency_limit=1,
        api_visibility="private",
    )
    clear_button.add(
        [
            input_file,
            lr_preview,
            bicubic_preview,
            restored_preview,
            status_output,
            npy_download,
            png_download,
        ]
    )


if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        show_error=False,
        max_file_size="1mb",
        enable_monitoring=False,
        blocked_paths=[
            str(CHECKPOINT_PATH.resolve()),
            str((ROOT / "model.py").resolve()),
            str((ROOT / "inference.py").resolve()),
        ],
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css=CSS,
    )
