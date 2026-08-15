---
title: DA-SwinSR Semiconductor Restoration
emoji: "🔬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# DA-SwinSR semiconductor image restoration demo

This repository contains the team's validation-selected, already-trained
degradation-aware Swin Transformer, a Gradio inference app for Spaces/local
use, and a stateless FastAPI website for Vercel.
It accepts one finite grayscale NumPy array with shape 128 x 128 and produces
a restored float32 array with shape 256 x 256.

The application performs inference only. It does not train or fine-tune a
model, and it never accepts checkpoint uploads from public users.

## Included files

- `app.py`: hosted Gradio application.
- `index.py`: root entrypoint for the Vercel website.
- `vercel_app.py`: stateless Vercel API and responsive browser interface.
- `model.py`: complete DA-SwinSR architecture.
- `inference.py`: preprocessing and postprocessing contract.
- `degradation_encoder.py`: compatibility module.
- `weights/checkpoint_best.pth`: trusted validation-selected checkpoint.
- `metadata/config.yaml`: experiment configuration.
- `metadata/metrics.json`: measured aggregate metrics.

The checkpoint SHA-256 is
`9eae0b4a5fe9d978dc14603e64cf2ac5b099d98e8ecd02af68b8624ebd06549b`.
The app verifies this digest before loading the checkpoint.

## Input and output contract

- Input: one `.npy` file, real numeric, finite, shape `(128, 128)` or
  `(128, 128, 1)`.
- Do not divide by 255 or perform per-image min-max normalization.
- Output: one float32 `.npy` array with shape `(256, 256)`.
- Display and downloadable storage output are clipped to the documented target
  range `[0, 1]`.

## Measured evidence

On the fixed 320-image internal test set, after validation-based checkpoint
selection, the model measured 29.0531 dB PSNR, 0.7663 SSIM and 0.2819 LPIPS.
These are internal experiment metrics, not leaderboard or factory-line results.

## Responsible use

Keep the original raw capture. Restored images are research outputs for assisted
viewing and must not replace original measurements. Cross-tool, cross-lot,
downstream defect and metrology performance have not been established.

See `DEPLOYMENT.md` for GitHub, Hugging Face Spaces and Vercel instructions.
