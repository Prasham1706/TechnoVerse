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


DEFAULT_STATUS = """
### Ready for an input
Upload one valid **128 × 128 grayscale `.npy`** file, then run the pretrained model.
"""

DEFAULT_TECHNICAL_DETAILS = """
Run a restoration to view the device, parameter count, prediction-call latency,
and stored output range.
"""


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


def _load_lr_array(uploaded_file: str | Path | None) -> np.ndarray:
    """Load and validate one low-resolution input without changing its scale."""
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
    return lr_array


def inspect_upload(uploaded_file: str | Path | None):
    """Preview and describe a selected input before model inference."""
    if uploaded_file is None:
        return None, DEFAULT_STATUS
    try:
        lr_array = _load_lr_array(uploaded_file)
        status = f"""
### Input validated
**Shape:** 128 × 128 · **dtype used:** float32 · **range:**
{float(lr_array.min()):.5f} to {float(lr_array.max()):.5f}

The file is ready. Select **Run restoration** to generate the 256 × 256 result.
"""
        return _display_uint8(lr_array), status
    except (ValueError, TypeError, FileNotFoundError) as error:
        return None, f"### Input needs attention\n{error}"


def restore_for_demo(uploaded_file: str | Path | None):
    """Validate one LR array, restore it, and return previews/downloads."""
    try:
        _prune_outputs()
        lr_array = _load_lr_array(uploaded_file)

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
### ✓ Restoration complete
**128 × 128 → 256 × 256** · **{prediction_ms:.1f} ms model-call time**

Per-image PSNR, SSIM and LPIPS are unavailable because this upload has no
matching ground-truth image.
"""
        technical_details = f"""
- **Runtime device:** {DEVICE.type.upper()}
- **Model parameters:** {PARAMETER_COUNT:,}
- **Prediction-call latency:** {prediction_ms:.2f} ms
- **Stored float32 output range:** {float(restored.min()):.5f} to {float(restored.max()):.5f}
- **Tensor transformation:** `[1, 1, 128, 128] → [1, 1, 256, 256]`
"""
        return (
            _display_uint8(lr_array),
            (_display_uint8(bicubic), _display_uint8(restored)),
            status,
            str(npy_path),
            str(png_path),
            technical_details,
        )
    except (ValueError, TypeError, FileNotFoundError) as error:
        return (
            None,
            None,
            f"### Input needs attention\n{error}",
            None,
            None,
            DEFAULT_TECHNICAL_DETAILS,
        )
    except Exception as error:
        traceback.print_exc()
        raise gr.Error("Restoration failed. Please contact the demo operator.") from error


def reset_demo():
    """Restore every interactive component to a clear initial state."""
    return (
        None,
        None,
        None,
        DEFAULT_STATUS,
        None,
        None,
        DEFAULT_TECHNICAL_DETAILS,
    )


CSS = """
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: clamp(14px, 2.2vw, 30px) !important;
}
#hero { margin-bottom: 18px; }
.hero-card {
  color: #ffffff;
  padding: clamp(24px, 4vw, 44px);
  border-radius: 24px;
  background:
    radial-gradient(circle at 90% 10%, rgba(56, 189, 248, .25), transparent 32%),
    linear-gradient(135deg, #07152f 0%, #0d3474 58%, #1559a8 100%);
  box-shadow: 0 18px 48px rgba(8, 34, 82, .22);
}
.hero-card h1 { color: #ffffff !important; margin: 8px 0 10px; font-size: clamp(2rem, 4vw, 3.25rem); }
.hero-card p { color: #dbeafe !important; max-width: 760px; font-size: 1.06rem; }
.eyebrow { color: #93c5fd; font-weight: 800; letter-spacing: .12em; font-size: .76rem; }
.hero-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; }
.ready-badge {
  display: inline-flex; align-items: center; gap: 9px; flex: 0 0 auto;
  padding: 10px 14px; border: 1px solid rgba(134, 239, 172, .45);
  border-radius: 999px; background: rgba(20, 83, 45, .32); color: #dcfce7;
  font-size: .86rem; font-weight: 750;
}
.ready-dot { width: 9px; height: 9px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 0 5px rgba(74, 222, 128, .14); }
.chip-row { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 20px; }
.hero-chip { padding: 7px 11px; border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16); color: #eff6ff; font-size: .84rem; }
.steps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 0 0 20px; }
.step-card, .surface-card, .metric-card {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  border-radius: 18px;
  box-shadow: 0 8px 26px rgba(15, 23, 42, .06);
}
.step-card { padding: 16px 18px; display: flex; gap: 13px; align-items: flex-start; }
.step-number { display: grid; place-items: center; width: 30px; height: 30px; flex: 0 0 30px; border-radius: 9px; background: #2563eb; color: white; font-weight: 800; }
.step-copy strong { display: block; color: var(--body-text-color); }
.step-copy span { display: block; margin-top: 3px; color: var(--body-text-color-subdued); font-size: .86rem; line-height: 1.4; }
.surface-card { padding: clamp(16px, 2vw, 22px) !important; }
#run-button { min-height: 50px; font-weight: 800; }
#status-panel { border-left: 4px solid #2563eb; padding: 6px 4px 6px 15px; margin-top: 8px; }
#comparison-heading { margin-top: 22px; }
.preview-card { min-width: 280px; }
#slider-card { border: 1px solid #60a5fa; box-shadow: 0 10px 30px rgba(37, 99, 235, .10); }
.panel-subtitle { color: var(--body-text-color-subdued); margin-top: -6px; font-size: .9rem; }
.download-note { font-size: .9rem; color: var(--body-text-color-subdued); }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0; }
.metric-card { padding: 18px; }
.metric-value { color: #2563eb; font-size: 1.75rem; line-height: 1; font-weight: 850; }
.metric-label { margin-top: 8px; color: var(--body-text-color); font-weight: 750; }
.metric-hint { color: var(--body-text-color-subdued); font-size: .82rem; margin-top: 3px; }
.disclaimer { color: var(--body-text-color-subdued); font-size: .88rem; line-height: 1.5; }
@media (max-width: 760px) {
  .hero-top { flex-direction: column; }
  .steps-grid, .metric-grid { grid-template-columns: 1fr; }
  #workspace, #comparison-row, #downloads-row { flex-direction: column !important; }
  .preview-card { min-width: 100% !important; }
}
"""

with gr.Blocks(
    title="DA-SwinSR Semiconductor Restoration Demo",
    analytics_enabled=False,
    delete_cache=(300, 600),
) as demo:
    gr.HTML(
        f"""
<section class="hero-card">
  <div class="hero-top">
    <div>
      <div class="eyebrow">TEAM TECHNOVERSE · PRETRAINED RESTORATION MODEL</div>
      <h1>DA-SwinSR Semiconductor Image Restoration</h1>
      <p>Restore one noisy grayscale image and generate a clear 2× super-resolved
      output using the team's validation-selected model.</p>
    </div>
    <div class="ready-badge"><span class="ready-dot"></span>Model ready · {DEVICE.type.upper()}</div>
  </div>
  <div class="chip-row">
    <span class="hero-chip">128 × 128 input</span>
    <span class="hero-chip">256 × 256 output</span>
    <span class="hero-chip">Inference only — no retraining</span>
    <span class="hero-chip">Grayscale NumPy input</span>
  </div>
</section>
""",
        elem_id="hero",
    )
    gr.HTML(
        """
<section class="steps-grid">
  <div class="step-card"><span class="step-number">1</span><div class="step-copy"><strong>Upload input</strong><span>.npy · grayscale · 128 × 128 · maximum 1 MB</span></div></div>
  <div class="step-card"><span class="step-number">2</span><div class="step-copy"><strong>Run pretrained model</strong><span>No training or checkpoint upload is required</span></div></div>
  <div class="step-card"><span class="step-number">3</span><div class="step-copy"><strong>Compare and download</strong><span>Original · bicubic baseline · DA-SwinSR result</span></div></div>
</section>
""",
        elem_id="workflow",
    )

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=3, elem_classes=["surface-card"]):
            gr.Markdown(
                """
### Step 1 — Upload a noisy input
Select one file from the **`NoisyLR`** folder. Do **not** upload a 256 × 256
ground-truth image from `GT`.

**Accepted:** `.npy` · shape `(128, 128)` or `(128, 128, 1)` · real numeric
values · no NaN/Inf · maximum 1 MB
"""
            )
            input_file = gr.File(
                label="Choose a 128 × 128 NoisyLR .npy file",
                file_types=[".npy"],
                file_count="single",
                type="filepath",
            )
        with gr.Column(scale=2, elem_classes=["surface-card"]):
            gr.Markdown("### Step 2 — Run the trained model")
            restore_button = gr.Button(
                "Run restoration",
                variant="primary",
                elem_id="run-button",
            )
            reset_button = gr.Button("Reset demo", variant="secondary")
            status_output = gr.Markdown(
                DEFAULT_STATUS,
                elem_id="status-panel",
            )

    gr.Markdown(
        """
## Step 3 — Compare the results
The left panel previews the noisy input. Drag the divider in the comparison
panel to inspect **bicubic interpolation versus DA-SwinSR**. All previews use
the same clipped `[0, 1]` display range.
""",
        elem_id="comparison-heading",
    )
    with gr.Row(equal_height=True, elem_id="comparison-row"):
        with gr.Column(scale=1, elem_classes=["surface-card", "preview-card"]):
            gr.Markdown("### Original input\n<div class='panel-subtitle'>128 × 128 noisy capture</div>")
            lr_preview = gr.Image(
                show_label=False,
                image_mode="L",
                height=390,
            )
        with gr.Column(scale=2, elem_classes=["surface-card", "preview-card"], elem_id="slider-card"):
            gr.Markdown("### Baseline ↔ model result\n<div class='panel-subtitle'>Drag to compare two 256 × 256 outputs</div>")
            comparison_slider = gr.ImageSlider(
                label="Bicubic baseline ↔ DA-SwinSR restoration",
                height=390,
            )

    with gr.Row(elem_id="downloads-row"):
        with gr.Column(scale=1, elem_classes=["surface-card"]):
            gr.Markdown("### Download results\nFiles appear here after a successful restoration.")
            with gr.Row():
                npy_download = gr.File(label="Scientific output · float32 .npy")
                png_download = gr.File(label="Visual preview · 8-bit .png")
        with gr.Column(scale=1, elem_classes=["surface-card"]):
            gr.Markdown(
                """
### Responsible use
Preserve the raw capture. This research output supports visual analysis and is
not a replacement measurement or an automated production-inspection decision.
"""
            )

    with gr.Accordion("Technical run details", open=False):
        technical_output = gr.Markdown(DEFAULT_TECHNICAL_DETAILS)

    gr.HTML(
        f"""
<section>
  <h2>Internal held-out evaluation</h2>
  <p>Mean results on the fixed {int(TEST_METRICS['num_images'])}-image paired
  internal test set using the validation-selected checkpoint.</p>
  <div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{TEST_METRICS['psnr_mean']:.2f} dB</div><div class="metric-label">PSNR ↑</div><div class="metric-hint">Higher is better</div></div>
    <div class="metric-card"><div class="metric-value">{TEST_METRICS['ssim_mean']:.4f}</div><div class="metric-label">SSIM ↑</div><div class="metric-hint">Higher is better</div></div>
    <div class="metric-card"><div class="metric-value">{TEST_METRICS['lpips_mean']:.4f}</div><div class="metric-label">LPIPS ↓</div><div class="metric-hint">Lower is better</div></div>
  </div>
  <p class="disclaimer">These values do not score the uploaded image and are
  not public-leaderboard or manufacturing-line validation results.</p>
</section>
"""
    )

    with gr.Accordion("Evaluation details and model architecture", open=False):
        gr.Markdown(
            f"""
**Internal-test mean ± population standard deviation**

- PSNR: **{TEST_METRICS['psnr_mean']:.4f} ± {TEST_METRICS['psnr_std']:.4f} dB**
- SSIM: **{TEST_METRICS['ssim_mean']:.4f} ± {TEST_METRICS['ssim_std']:.4f}**
- LPIPS: **{TEST_METRICS['lpips_mean']:.4f} ± {TEST_METRICS['lpips_std']:.4f}** *(lower is better)*

**Architecture:** CNN feature extraction → learned 48-D degradation-aware
latent conditioning → six FiLM-conditioned Swin blocks → PixelShuffle 2× →
high-resolution refinement with a bicubic residual path.
"""
        )

    input_file.change(
        fn=inspect_upload,
        inputs=[input_file],
        outputs=[lr_preview, status_output],
        api_visibility="private",
    )

    restore_button.click(
        fn=restore_for_demo,
        inputs=[input_file],
        outputs=[
            lr_preview,
            comparison_slider,
            status_output,
            npy_download,
            png_download,
            technical_output,
        ],
        concurrency_limit=1,
        api_visibility="private",
        show_progress="full",
        show_progress_on=[comparison_slider],
        scroll_to_output=True,
        trigger_mode="once",
    )
    reset_button.click(
        fn=reset_demo,
        inputs=[],
        outputs=[
            input_file,
            lr_preview,
            comparison_slider,
            status_output,
            npy_download,
            png_download,
            technical_output,
        ],
        api_visibility="private",
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
