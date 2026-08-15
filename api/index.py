"""Vercel ASGI entrypoint for the standalone DA-SwinSR Gradio app."""

from fastapi import FastAPI
import gradio as gr

from app import CHECKPOINT_PATH, CSS, ROOT, demo


parent = FastAPI(
    title="DA-SwinSR",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app = gr.mount_gradio_app(
    parent,
    demo.queue(max_size=8, default_concurrency_limit=1),
    path="/",
    show_error=False,
    max_file_size="1mb",
    enable_monitoring=False,
    run_history=False,
    footer_links=[],
    blocked_paths=[
        str(CHECKPOINT_PATH.resolve()),
        str((ROOT / "model.py").resolve()),
        str((ROOT / "inference.py").resolve()),
    ],
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css=CSS,
)
