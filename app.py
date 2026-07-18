"""Gradio WebUI for CUDA-accelerated Sidon speech enhancement."""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from typing import Generator
import warnings

import gradio as gr

from sidon_engine import (
    ENGINE,
    InferenceOptions,
    ProgressUpdate,
    SidonError,
    get_cuda_info,
)


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("sidon.webui")

warnings.filterwarnings(
    "ignore",
    message=r"The '(theme|css|js)' parameter in the Blocks constructor.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'show_api' parameter in event listeners.*",
    category=DeprecationWarning,
)


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.skip-link {
  position: absolute; left: 0.75rem; top: -5rem; z-index: 9999;
  padding: 0.7rem 1rem; border-radius: 8px;
  background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color) !important;
}
.skip-link:focus { top: 0.75rem; }
.hero { text-align: center; margin: 0.75rem 0 1.25rem; }
.hero h1 { font-size: 2.15rem; margin-bottom: 0.25rem; }
.hero p { color: var(--body-text-color-subdued); margin: 0 auto; max-width: 760px; }
.workspace-heading { margin: 0.5rem 0 0.2rem; }
.workspace-help { color: var(--body-text-color-subdued); margin-bottom: 0.8rem; }
.gpu-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  background: var(--background-fill-secondary);
  margin-bottom: 1rem;
}
.progress-shell {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  background: var(--background-fill-secondary);
}
.progress-head {
  display: flex; justify-content: space-between; gap: 1rem;
  font-weight: 650; margin-bottom: 0.5rem;
}
.progress-percent { font-variant-numeric: tabular-nums; }
.progress-track {
  width: 100%; height: 12px; overflow: hidden; border-radius: 999px;
  background: color-mix(in srgb, var(--body-text-color) 12%, transparent);
}
.progress-fill {
  height: 100%; border-radius: inherit;
  background: linear-gradient(90deg, #2563eb, #22c55e);
  transition: width .25s ease;
}
.progress-message {
  margin-top: .55rem; color: var(--body-text-color-subdued); min-height: 1.4em;
}
.progress-error .progress-fill { background: #dc2626; }
.progress-error .progress-message { color: #dc2626; }
#enhance-button { min-height: 46px; font-weight: 700; }
.footer-note { text-align: center; color: var(--body-text-color-subdued); }
"""


def _system_status() -> str:
    try:
        cuda = get_cuda_info()
        readiness = (
            "Sidon loaded on CUDA"
            if ENGINE.loaded
            else "CUDA ready (model not preloaded in this process)"
        )
        return (
            '<div class="gpu-card" role="status" aria-live="polite" '
            'aria-atomic="true">'
            '<span aria-hidden="true">✓</span> '
            f"<strong>{readiness}:</strong> {html.escape(cuda.name)} · "
            f"{cuda.total_gb:.1f} GB VRAM · CUDA {html.escape(cuda.cuda_version)}"
            "</div>"
        )
    except SidonError as exc:
        return (
            '<div class="gpu-card" role="alert" aria-live="assertive" '
            'aria-atomic="true"><span aria-hidden="true">✕</span> '
            "<strong>CUDA unavailable:</strong> "
            f"{html.escape(str(exc))}</div>"
        )


def _progress_html(update: ProgressUpdate, *, error: bool = False) -> str:
    percent = max(0, min(100, int(update.percent)))
    css_class = "progress-shell progress-error" if error else "progress-shell"
    if error:
        label = "Error"
    elif percent == 100:
        label = "Complete"
    elif update.message.startswith("Ready"):
        label = "Ready"
    elif "cancelled" in update.message.lower():
        label = "Cancelled"
    else:
        label = "Processing"
    container_role = "alert" if error else "status"
    live_mode = "assertive" if error else "polite"
    return (
        f'<div class="{css_class}" role="{container_role}" '
        f'aria-live="{live_mode}" aria-atomic="true">'
        '<div class="progress-head">'
        f'<span>{label}</span><span class="progress-percent">{percent}%</span>'
        "</div>"
        '<div class="progress-track" role="progressbar" '
        'aria-label="Speech enhancement progress" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{percent}">'
        f'<div class="progress-fill" style="width:{percent}%"></div>'
        "</div>"
        f'<div class="progress-message">{html.escape(update.message)}</div>'
        "</div>"
    )


def _enhance_audio(
    input_audio: str | None,
    auto_chunk: bool,
    manual_chunk_seconds: float,
    highpass_hz: float,
    normalize_input: bool,
    input_peak: float,
    oom_recovery: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Generator[tuple[str | None, str, str], None, None]:
    if not input_audio:
        update = ProgressUpdate(0, "Please upload an audio file first.")
        yield None, _progress_html(update, error=True), ""
        return

    options = InferenceOptions(
        auto_chunk=auto_chunk,
        chunk_seconds=manual_chunk_seconds,
        highpass_hz=highpass_hz,
        normalize_input=normalize_input,
        input_peak=input_peak,
        oom_recovery=oom_recovery,
    )

    try:
        for update in ENGINE.enhance(input_audio, options, OUTPUT_DIR):
            progress(
                update.percent / 100,
                desc=f"{update.percent}% — {update.message}",
            )
            yield (
                update.output_path,
                _progress_html(update),
                update.details,
            )
    except SidonError as exc:
        LOGGER.warning("Expected inference failure: %s", exc)
        update = ProgressUpdate(0, str(exc))
        yield (
            None,
            _progress_html(update, error=True),
            f"**Could not enhance audio:** {exc}",
        )
    except Exception as exc:  # keep the UI alive and include an actionable log
        LOGGER.exception("Unexpected Sidon inference failure")
        update = ProgressUpdate(
            0,
            "Unexpected inference failure. See the terminal for details.",
        )
        yield (
            None,
            _progress_html(update, error=True),
            f"**Unexpected error:** `{type(exc).__name__}: {exc}`",
        )


def _toggle_manual_chunk(auto_chunk: bool) -> gr.Slider:
    return gr.Slider(interactive=not auto_chunk)


def _clear_ui() -> tuple[None, None, str, str]:
    ready = ProgressUpdate(0, "Ready. Upload audio to begin.")
    return None, None, _progress_html(ready), ""


def _cancel_ui() -> tuple[str, str]:
    cancelled = ProgressUpdate(0, "Enhancement cancelled.")
    return _progress_html(cancelled), "**Enhancement cancelled.**"


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
    with gr.Blocks(
        title="Sidon Speech Enhancement",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
        js="() => { document.documentElement.lang = 'en'; }",
    ) as demo:
        gr.HTML(
            """
            <a class="skip-link" href="#audio-workspace">Skip to audio controls</a>
            <div class="hero" role="banner">
              <h1>Sidon Speech Enhancement</h1>
              <p>Upload noisy or degraded speech, then let Sidon restore a clean
              48 kHz waveform on your NVIDIA GPU.</p>
            </div>
            """
        )
        gr.HTML(_system_status())
        gr.HTML(
            """
            <section aria-labelledby="audio-workspace">
              <h2 id="audio-workspace" class="workspace-heading" tabindex="-1">
                Audio workspace
              </h2>
              <p class="workspace-help">
                Step 1: upload audio. Step 2: choose Enhance Speech.
                Step 3: listen to or download the enhanced result.
              </p>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_audio = gr.Audio(
                    label="Step 1 — Original or noisy speech",
                    sources=["upload"],
                    type="filepath",
                    format="wav",
                )
            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Step 3 — Enhanced speech result",
                    type="filepath",
                    format="wav",
                    interactive=False,
                    editable=False,
                )

        with gr.Row():
            enhance_button = gr.Button(
                "Enhance Speech",
                variant="primary",
                elem_id="enhance-button",
                scale=5,
                min_width=220,
            )
            cancel_button = gr.Button(
                "Cancel enhancement",
                variant="stop",
                scale=1,
                min_width=160,
            )
            clear_button = gr.Button(
                "Clear input and result",
                variant="secondary",
                scale=1,
                min_width=180,
            )

        gr.HTML('<h2 class="workspace-heading">Processing status</h2>')
        progress_display = gr.HTML(
            _progress_html(ProgressUpdate(0, "Ready. Upload audio to begin."))
        )
        result_details = gr.Markdown()

        with gr.Accordion("Advanced settings (optional)", open=False):
            gr.Markdown(
                "Automatic chunk sizing is recommended. It uses detected VRAM "
                "and retries with smaller chunks if CUDA memory is tight. "
                "The default settings are suitable for most files."
            )
            auto_chunk = gr.Checkbox(
                value=True,
                label="Automatic internal chunk length from VRAM",
                info="Recommended. This does not limit the total audio length.",
            )
            manual_chunk_seconds = gr.Slider(
                minimum=4,
                maximum=96,
                value=60,
                step=1,
                label="Internal processing chunk length (seconds)",
                info=(
                    "Used only when automatic mode is disabled. Longer files "
                    "are split into multiple chunks automatically."
                ),
                interactive=False,
            )
            with gr.Row():
                highpass_hz = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=50,
                    step=5,
                    label="High-pass filter (Hz)",
                    info="50 Hz removes low-frequency rumble; 0 disables it.",
                )
                input_peak = gr.Slider(
                    minimum=0.10,
                    maximum=0.99,
                    value=0.90,
                    step=0.01,
                    label="Input normalization peak",
                    info="Used only when input normalization is enabled.",
                )
            with gr.Row():
                normalize_input = gr.Checkbox(
                    value=True,
                    label="Normalize input before enhancement",
                )
                oom_recovery = gr.Checkbox(
                    value=True,
                    label="Automatically recover from CUDA out-of-memory",
                )

        auto_chunk.change(
            fn=_toggle_manual_chunk,
            inputs=auto_chunk,
            outputs=manual_chunk_seconds,
            queue=False,
        )

        job = enhance_button.click(
            fn=_enhance_audio,
            inputs=[
                input_audio,
                auto_chunk,
                manual_chunk_seconds,
                highpass_hz,
                normalize_input,
                input_peak,
                oom_recovery,
            ],
            outputs=[output_audio, progress_display, result_details],
            show_progress="minimal",
            concurrency_limit=1,
            concurrency_id="sidon_gpu",
            api_name="enhance_speech",
        )
        cancel_button.click(
            fn=_cancel_ui,
            cancels=[job],
            outputs=[progress_display, result_details],
            queue=False,
        )
        clear_button.click(
            fn=_clear_ui,
            outputs=[
                input_audio,
                output_audio,
                progress_display,
                result_details,
            ],
            queue=False,
        )

        gr.Markdown(
            "Model: [sarulab-speech/sidon-v0.1]"
            "(https://huggingface.co/sarulab-speech/sidon-v0.1) · "
            "Models stay loaded in CUDA VRAM after the first run.",
            elem_classes=["footer-note"],
        )

    return demo


def _preload_before_launch() -> None:
    print()
    print("=" * 62)
    print("  Preparing Sidon on CUDA before the WebUI starts")
    print("=" * 62)
    print("[  0%] Checking CUDA and model cache...", flush=True)
    try:
        for update in ENGINE.preload():
            startup_percent = min(99, round(update.percent / 15 * 100))
            print(f"[{startup_percent:3d}%] {update.message}", flush=True)
    except SidonError as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise SystemExit(1) from exc
    print("[100%] Loaded. Ready. Opening the WebUI...", flush=True)
    print()


if __name__ == "__main__":
    _preload_before_launch()

demo = build_demo()


if __name__ == "__main__":
    server_name = os.environ.get("SIDON_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("SIDON_SERVER_PORT", "7860"))
    inbrowser = os.environ.get("SIDON_INBROWSER", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=inbrowser,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
    )
