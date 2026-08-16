"""Stateless Vercel UI for the trained Final Order-aware Swin model (Member 4).

The browser sends the complete .npy payload in one request and receives inline
previews/downloads in one response. This avoids Gradio queue and /tmp state
crossing Vercel serverless instances.
"""

from __future__ import annotations

import base64
import io
import threading
import time
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from starlette.concurrency import run_in_threadpool
import torch
import torch.nn.functional as F

import app as core


MAX_INPUT_BYTES = 1024 * 1024
INFERENCE_SLOT = threading.BoundedSemaphore(1)


def _encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _load_array(payload: bytes) -> np.ndarray:
    if not payload:
        raise HTTPException(status_code=400, detail="Choose one .npy input file.")
    if len(payload) > MAX_INPUT_BYTES:
        raise HTTPException(status_code=413, detail="Input exceeds the 1 MB limit.")
    if not payload.startswith(b"\x93NUMPY"):
        raise HTTPException(
            status_code=400,
            detail="The upload is not a valid NumPy .npy file.",
        )

    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=4096,
            )
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=4096,
            )
        else:
            raise ValueError("Unsupported NumPy format version.")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The upload has an invalid NumPy header.",
        ) from error

    if shape not in ((128, 128), (128, 128, 1)):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected shape (128, 128); received {shape}. "
                "Choose a file from NoisyLR, not GT."
            ),
        )
    if dtype.hasobject or dtype.kind not in "iuf" or dtype.itemsize > 8:
        raise HTTPException(
            status_code=400,
            detail="Expected a real numeric NumPy array.",
        )
    expected_data_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload) - stream.tell() != expected_data_bytes:
        raise HTTPException(
            status_code=400,
            detail="The NumPy payload length is invalid.",
        )

    stream.seek(0)
    try:
        loaded = np.load(stream, allow_pickle=False, max_header_size=4096)
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(loaded, dtype=np.float32)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The upload is not a readable, non-pickled NumPy array.",
        ) from error
    if array.shape == (128, 128, 1):
        array = array[..., 0]
    if not np.isfinite(array).all():
        raise HTTPException(status_code=400, detail="Input contains NaN or Inf values.")
    return np.ascontiguousarray(array)


def _restore(payload: bytes) -> dict[str, object]:
    lr_array = _load_array(payload)
    bicubic = F.interpolate(
        torch.from_numpy(lr_array)[None, None],
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )[0, 0].numpy()

    lr_tensor = core.preprocess_array(lr_array, core.STATS)
    with INFERENCE_SLOT:
        if core.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        sr_tensor = core.predict(core.MODEL, lr_tensor, core.DEVICE)
        if core.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        prediction_ms = 1000.0 * (time.perf_counter() - started)

    restored = core.postprocess_tensor(sr_tensor, core.STATS, clip_for_storage=True)
    if restored.shape != (256, 256) or not np.isfinite(restored).all():
        raise RuntimeError("The model returned an invalid output array.")

    npy_buffer = io.BytesIO()
    np.save(
        npy_buffer,
        restored.astype(np.float32, copy=False),
        allow_pickle=False,
    )
    lr_png = core._png_bytes(lr_array)
    bicubic_png = core._png_bytes(bicubic)
    restored_png = core._png_bytes(restored)
    return {
        "input_png": _encode(lr_png),
        "bicubic_png": _encode(bicubic_png),
        "restored_png": _encode(restored_png),
        "restored_npy": _encode(npy_buffer.getvalue()),
        "prediction_ms": round(prediction_ms, 1),
        "input_min": round(float(lr_array.min()), 5),
        "input_max": round(float(lr_array.max()), 5),
        "output_min": round(float(restored.min()), 5),
        "output_max": round(float(restored.max()), 5),
        "device": core.DEVICE.type.upper(),
        "parameters": core.PARAMETER_COUNT,
    }


app = FastAPI(
    title="Final Order-aware Swin · TechnoVerse",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(
        INDEX_HTML,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


@app.post("/api/restore")
async def restore(request: Request) -> JSONResponse:
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type not in ("application/octet-stream", "application/x-npy"):
        raise HTTPException(status_code=415, detail="Expected a NumPy file upload.")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_INPUT_BYTES:
                raise HTTPException(status_code=413, detail="Input exceeds the 1 MB limit.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid content length.") from error

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_INPUT_BYTES:
            raise HTTPException(status_code=413, detail="Input exceeds the 1 MB limit.")
        body.extend(chunk)
    try:
        result = await run_in_threadpool(_restore, bytes(body))
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Restoration failed. Please try again or contact the demo operator.",
        ) from error
    return JSONResponse(
        result,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Team TechnoVerse Final Order-aware Swin semiconductor image restoration demo">
  <title>Order-aware Swin · Team TechnoVerse</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070d19; --surface: #0d1524; --surface2: #111c2e;
      --line: #25334b; --text: #f4f7fb; --muted: #a9b7ca;
      --blue: #4f7cff; --cyan: #4ac8ff; --green: #4ade80;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
    button, input { font: inherit; }
    .page { width: min(1240px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 56px; }
    .hero { padding: clamp(25px, 5vw, 52px); border-radius: 26px; overflow: hidden;
      background: radial-gradient(circle at 88% 12%, rgba(74,200,255,.24), transparent 31%),
        linear-gradient(135deg,#081934,#123c7c 62%,#1762b5); box-shadow: 0 24px 70px #02071299; }
    .hero-top { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; }
    .eyebrow { color: #9ddcff; letter-spacing: .13em; font-size: .76rem; font-weight: 800; }
    h1 { margin: 10px 0 12px; max-width: 800px; font-size: clamp(2rem,5vw,3.5rem); line-height: 1.04; }
    .hero p { margin: 0; max-width: 760px; color: #d8e9ff; font-size: 1.08rem; line-height: 1.65; }
    .ready { display: inline-flex; align-items: center; gap: 9px; flex: 0 0 auto;
      border: 1px solid #86efac80; background: #14532d55; color: #dcfce7;
      border-radius: 999px; padding: 10px 14px; font-size: .86rem; font-weight: 750; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 5px #4ade8024; }
    .chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
    .chip { border: 1px solid #ffffff2b; background: #ffffff12; border-radius: 999px; padding: 8px 12px; font-size: .86rem; }
    .steps { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin: 24px 0; }
    .step, .card { border: 1px solid var(--line); background: var(--surface); border-radius: 18px; }
    .step { padding: 17px 18px; display: flex; gap: 13px; }
    .num { display: grid; place-items: center; width: 31px; height: 31px; flex: 0 0 31px;
      border-radius: 9px; background: var(--blue); font-weight: 850; }
    .step strong { display: block; margin-bottom: 4px; }
    .step span { color: var(--muted); font-size: .87rem; line-height: 1.4; }
    .workspace { display: grid; grid-template-columns: 1.45fr 1fr; gap: 16px; }
    .card { padding: clamp(18px,2.5vw,25px); }
    h2, h3 { margin-top: 0; } h2 { font-size: 1.55rem; } h3 { font-size: 1.12rem; }
    .muted { color: var(--muted); line-height: 1.55; }
    code { background: #1d293b; border: 1px solid #344258; padding: 2px 5px; border-radius: 5px; color: #e6edf7; }
    .upload { display: block; margin-top: 16px; padding: 22px; border: 1.5px dashed #496180;
      border-radius: 14px; background: #111b2b; cursor: pointer; transition: .2s ease; }
    .upload:hover, .upload:focus-within { border-color: var(--cyan); background: #12223a; }
    .upload input { position: absolute; opacity: 0; pointer-events: none; }
    .upload-title { display: block; font-weight: 800; } .upload-note { color: var(--muted); font-size: .86rem; margin-top: 5px; }
    .actions { display: grid; gap: 10px; margin-top: 16px; }
    .button { min-height: 50px; border-radius: 11px; border: 1px solid #5d83ff; cursor: pointer; font-weight: 800; }
    .primary { color: white; background: linear-gradient(135deg,#315fdf,#5984ff); }
    .secondary { color: var(--text); background: #27354a; border-color: #3e4c60; }
    .button:disabled { opacity: .45; cursor: not-allowed; }
    .status { margin-top: 18px; padding: 2px 0 2px 16px; border-left: 4px solid var(--blue); min-height: 76px; }
    .status strong { display: block; margin-bottom: 7px; font-size: 1.02rem; }
    .status.error { border-color: #f87171; } .status.success { border-color: var(--green); }
    .results-title { margin: 34px 0 10px; }
    .results { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
    .result-card { min-width: 0; } .result-card.selected { border-color: #4f8fff; box-shadow: 0 12px 34px #245fd126; }
    .image-frame { aspect-ratio: 1; margin-top: 14px; display: grid; place-items: center;
      border-radius: 12px; background: #111827; overflow: hidden; border: 1px solid #1f2d42; }
    .image-frame img { width: 100%; height: 100%; object-fit: contain; display: none; }
    .placeholder { color: #7f91a9; font-size: .9rem; text-align: center; padding: 15px; }
    .downloads { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; margin-top: 16px; }
    .download-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .download { display: none; justify-content: space-between; align-items: center; gap: 12px;
      border: 1px solid #4f7cff; background: #244eaa24; border-radius: 12px; padding: 14px;
      color: var(--text); text-decoration: none; }
    .download:hover { background: #315fc23b; } .download small { display: block; color: var(--muted); margin-top: 3px; }
    .responsible { border-left: 4px solid #f5b942; }
    .metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 16px 0; }
    .metric { border: 1px solid var(--line); border-radius: 14px; background: var(--surface2); padding: 17px; }
    .metric b { display: block; color: #76a2ff; font-size: 1.65rem; } .metric span { color: var(--muted); font-size: .84rem; }
    .evidence { margin-top: 26px; } details { border-top: 1px solid var(--line); padding: 16px 0; }
    summary { cursor: pointer; font-weight: 750; } details p { color: var(--muted); line-height: 1.6; }
    @media (max-width: 850px) {
      .hero-top { flex-direction: column; } .steps, .results, .metrics { grid-template-columns: 1fr; }
      .workspace, .downloads { grid-template-columns: 1fr; } .download-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-top"><div><div class="eyebrow">TEAM TECHNOVERSE · PRETRAINED RESTORATION MODEL</div>
        <h1>Final Order-aware Swin — Semiconductor Image Restoration</h1>
        <p>Restore one noisy grayscale image and generate a clear 2× super-resolved output using the team's validation-selected Member 4 checkpoint.</p>
      </div><div class="ready"><span class="dot"></span>Model ready · __DEVICE__</div></div>
      <div class="chips"><span class="chip">128 × 128 input</span><span class="chip">256 × 256 output</span>
        <span class="chip">Inference only — no retraining</span><span class="chip">Grayscale NumPy input</span></div>
    </section>

    <section class="steps" aria-label="Demo workflow">
      <div class="step"><b class="num">1</b><div><strong>Upload input</strong><span>.npy · grayscale · 128 × 128 · maximum 1 MB</span></div></div>
      <div class="step"><b class="num">2</b><div><strong>Run pretrained model</strong><span>One secure request; no checkpoint upload</span></div></div>
      <div class="step"><b class="num">3</b><div><strong>Compare and download</strong><span>Original · bicubic baseline · Order-aware Swin result</span></div></div>
    </section>

    <section class="workspace">
      <article class="card"><h3>Step 1 — Upload a noisy input</h3>
        <p class="muted">Choose one file from <strong><code>NoisyLR</code></strong>. Do <strong>not</strong> upload a 256 × 256 ground-truth file from <code>GT</code>.</p>
        <p class="muted"><strong>Accepted:</strong> <code>.npy</code> · shape <code>(128, 128)</code> or <code>(128, 128, 1)</code> · real finite values · ≤ 1 MB</p>
        <label class="upload" for="file"><input id="file" type="file" accept=".npy,application/octet-stream">
          <span class="upload-title" id="file-title">Choose a 128 × 128 NoisyLR .npy file</span>
          <span class="upload-note" id="file-note">The file stays in this page until you select Run restoration.</span>
        </label>
      </article>
      <article class="card"><h3>Step 2 — Run the trained model</h3>
        <div class="actions"><button class="button primary" id="run" disabled>Run restoration</button>
          <button class="button secondary" id="reset">Reset demo</button></div>
        <div class="status" id="status"><strong>Ready for an input</strong><span class="muted">Upload one valid 128 × 128 grayscale .npy file.</span></div>
      </article>
    </section>

    <h2 class="results-title">Step 3 — Compare the results</h2>
    <p class="muted">Compare the noisy input, standard bicubic interpolation, and Order-aware Swin restoration side by side. All previews use the same clipped <code>[0, 1]</code> display range.</p>
    <section class="results">
      <article class="card result-card"><h3>Original input</h3><span class="muted">128 × 128 noisy capture</span><div class="image-frame"><span class="placeholder">Result appears here</span><img id="input-image" alt="Uploaded noisy input"></div></article>
      <article class="card result-card"><h3>Bicubic baseline</h3><span class="muted">256 × 256 standard interpolation</span><div class="image-frame"><span class="placeholder">Result appears here</span><img id="bicubic-image" alt="Bicubic baseline"></div></article>
      <article class="card result-card selected"><h3>Order-aware Swin restoration</h3><span class="muted">256 × 256 trained-model output</span><div class="image-frame"><span class="placeholder">Result appears here</span><img id="restored-image" alt="Order-aware Swin restored output"></div></article>
    </section>

    <section class="downloads">
      <article class="card"><h3>Download results</h3><div class="download-grid">
        <a class="download" id="npy-download" download="OrderAwareSwin_restored.npy"><span><b>Scientific array</b><small>256 × 256 · float32 .npy</small></span><b>Download</b></a>
        <a class="download" id="png-download" download="OrderAwareSwin_preview.png"><span><b>Visual preview</b><small>256 × 256 · clipped PNG</small></span><b>Download</b></a>
      </div><p class="muted" id="download-note">Files appear after a successful restoration.</p></article>
      <article class="card responsible"><h3>Responsible use</h3><p class="muted">Preserve the raw capture. This research output supports visual analysis and is not a replacement measurement or an automated production-inspection decision.</p></article>
    </section>

    <section class="card evidence"><h2>Internal held-out evaluation</h2>
      <p class="muted">Mean results on the fixed 320-image paired internal test set using the validation-selected checkpoint.</p>
      <div class="metrics"><div class="metric"><b>28.88 dB</b><strong>PSNR ↑</strong><br><span>Higher is better</span></div>
        <div class="metric"><b>0.7662</b><strong>SSIM ↑</strong><br><span>Higher is better</span></div>
        <div class="metric"><b>0.2773</b><strong>LPIPS ↓</strong><br><span>Lower is better</span></div></div>
      <p class="muted">These values do not score the uploaded image and are not public-leaderboard or manufacturing-line validation results.</p>
      <details id="technical"><summary>Technical run details</summary><p id="technical-copy">Run a restoration to view device, parameters, latency and output range.</p></details>
      <details><summary>Model architecture</summary><p>CNN feature extraction → learned 48-D <strong>order-aware</strong> degradation-conditioned latent → six FiLM-conditioned Swin blocks → PixelShuffle 2× → high-resolution refinement with a bicubic residual path → auxiliary 6-class degradation-order head (training only).</p></details>
    </section>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const runButton = document.getElementById('run');
    const resetButton = document.getElementById('reset');
    const statusBox = document.getElementById('status');
    let selectedFile = null;
    let requestVersion = 0;
    let activeController = null;

    function setStatus(title, message, kind='') {
      statusBox.className = 'status ' + kind;
      const heading = document.createElement('strong'); heading.textContent = title;
      const copy = document.createElement('span'); copy.className = 'muted'; copy.textContent = message;
      statusBox.replaceChildren(heading, copy);
    }
    function invalidatePendingRequest() {
      requestVersion += 1;
      if (activeController) activeController.abort();
      activeController = null;
    }
    function clearResults() {
      for (const id of ['input-image','bicubic-image','restored-image']) {
        const image = document.getElementById(id); image.removeAttribute('src'); image.style.display = 'none';
        image.previousElementSibling.style.display = 'block';
      }
      for (const id of ['npy-download','png-download']) { const link = document.getElementById(id); link.removeAttribute('href'); link.style.display = 'none'; }
      document.getElementById('download-note').style.display = 'block';
      document.getElementById('technical-copy').textContent = 'Run a restoration to view device, parameters, latency and output range.';
    }
    function showImage(id, encoded) {
      const image = document.getElementById(id); image.src = 'data:image/png;base64,' + encoded;
      image.style.display = 'block'; image.previousElementSibling.style.display = 'none';
    }
    fileInput.addEventListener('change', () => {
      invalidatePendingRequest();
      clearResults(); selectedFile = fileInput.files[0] || null;
      if (!selectedFile) { runButton.disabled = true; return; }
      const validSuffix = selectedFile.name.toLowerCase().endsWith('.npy');
      const validSize = selectedFile.size <= 1024 * 1024;
      if (!validSuffix || !validSize) {
        selectedFile = null; runButton.disabled = true;
        setStatus('Input needs attention', validSuffix ? 'The file exceeds 1 MB.' : 'Only .npy files are accepted.', 'error'); return;
      }
      document.getElementById('file-title').textContent = selectedFile.name;
      document.getElementById('file-note').textContent = `${(selectedFile.size / 1024).toFixed(1)} KB · ready for server validation`;
      runButton.disabled = false;
      setStatus('File selected', 'Select Run restoration to validate the array and generate the 256 × 256 result.');
    });
    resetButton.addEventListener('click', () => {
      invalidatePendingRequest();
      selectedFile = null; fileInput.value = ''; runButton.disabled = true; clearResults();
      document.getElementById('file-title').textContent = 'Choose a 128 × 128 NoisyLR .npy file';
      document.getElementById('file-note').textContent = 'The file stays in this page until you select Run restoration.';
      setStatus('Ready for an input', 'Upload one valid 128 × 128 grayscale .npy file.');
    });
    runButton.addEventListener('click', async () => {
      if (!selectedFile) return;
      const thisRequest = ++requestVersion;
      const fileToRun = selectedFile;
      activeController = new AbortController();
      runButton.disabled = true; setStatus('Restoration in progress', 'The pretrained model is processing this input.');
      let failureTitle = 'Server unavailable';
      try {
        const response = await fetch('/api/restore', {
          method: 'POST', headers: {'Content-Type':'application/octet-stream'},
          body: fileToRun, signal: activeController.signal
        });
        const raw = await response.text();
        let data = null;
        try { data = raw ? JSON.parse(raw) : null; } catch { data = null; }
        if (!response.ok) {
          failureTitle = response.status < 500 ? 'Input needs attention' : 'Server unavailable';
          throw new Error(data?.detail || (response.status >= 500
            ? 'The restoration server is temporarily unavailable.'
            : `Request failed (${response.status}).`));
        }
        if (!data) throw new Error('The server returned an invalid response.');
        if (thisRequest !== requestVersion) return;
        showImage('input-image', data.input_png); showImage('bicubic-image', data.bicubic_png); showImage('restored-image', data.restored_png);
        const npy = document.getElementById('npy-download'); npy.href = 'data:application/octet-stream;base64,' + data.restored_npy; npy.style.display = 'flex';
        const png = document.getElementById('png-download'); png.href = 'data:image/png;base64,' + data.restored_png; png.style.display = 'flex';
        document.getElementById('download-note').style.display = 'none';
        document.getElementById('technical-copy').textContent = `${data.device} · ${data.parameters.toLocaleString()} parameters · ${data.prediction_ms} ms model-call time · output range ${data.output_min} to ${data.output_max}.`;
        setStatus('✓ Restoration complete', `128 × 128 → 256 × 256 · ${data.prediction_ms} ms model-call time. Per-image metrics need matching ground truth.`, 'success');
        document.querySelector('.results-title').scrollIntoView({behavior:'smooth', block:'start'});
      } catch (error) {
        if (error.name !== 'AbortError' && thisRequest === requestVersion) {
          clearResults(); setStatus(failureTitle, error.message, 'error');
        }
      } finally {
        if (thisRequest === requestVersion) {
          activeController = null;
          runButton.disabled = !selectedFile;
        }
      }
    });
  </script>
</body>
</html>
""".replace("__DEVICE__", core.DEVICE.type.upper())
