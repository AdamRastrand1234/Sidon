# Sidon Speech Enhancement WebUI

An inference-only Windows WebUI for restoring noisy or degraded speech with
[SaruLab's Sidon v0.1 model](https://huggingface.co/sarulab-speech/sidon-v0.1).
Training, dataset preparation, cluster scripts, and experiment dependencies
have been removed from this fork.

## Requirements

- Windows 10 or 11
- NVIDIA GPU with at least 4 GB VRAM (8 GB or more recommended)
- Current NVIDIA driver
- 64-bit Python 3.10 or 3.11
- Internet access for installation and the first model download

The target configuration is an RTX 4060 with 8 GB VRAM and 16 GB system RAM.

## Install and run

1. Double-click `install.bat` once.
2. Double-click `run.bat`.
3. Your browser opens at <http://127.0.0.1:7860>.
4. Upload an audio file and click **Enhance Speech**.

The first start downloads the two official CUDA model files (about 1 GB total)
to `model_cache/`. `run.bat` waits until both models are loaded on CUDA before
opening the browser, so the first **Enhance Speech** click can begin immediately.
Later starts reuse the local cache.

Enhanced 48 kHz WAV files are saved in `outputs/` and can also be downloaded
or played directly with the standard Gradio audio player in the WebUI.

## Accessibility

The main workflow is keyboard-friendly and keeps technical chunk controls
inside the closed optional settings panel. CUDA readiness and processing
updates use screen-reader live regions, the visible percentage is exposed as a
semantic progress bar, errors use alert announcements, and Clear/Cancel restore
an unambiguous status.

## VRAM-aware chunks

Automatic chunk sizing is enabled by default. Lower-VRAM GPUs use shorter
chunks; larger GPUs use longer chunks. An 8 GB GPU starts at 60 seconds.
Available free VRAM is considered as well, so other GPU applications can lower
the recommendation.

**This is not an upload-duration limit.** A recording can be much longer than
60 seconds. Sidon automatically processes it as multiple internal chunks and
returns one complete enhanced WAV file.

Recordings around two hours are supported. Enhanced chunks are streamed to
disk so the complete 48 kHz result does not accumulate in RAM. A two-hour
recording needs about 2.2 GiB of free working disk space in addition to the
uploaded source file; the WebUI checks this before GPU processing begins.

If a chunk still causes a CUDA out-of-memory error, Sidon clears the failed
allocation, halves the chunk length, and retries. This recovery can be disabled
under **Advanced settings**. Manual chunk lengths from 4 to 96 seconds are also
available there.

## Advanced settings

- **Automatic internal chunk length from VRAM**: recommended for stable
  operation; it does not limit total file length.
- **Internal processing chunk length**: used only when automatic mode is
  disabled. Longer recordings are split automatically.
- **High-pass filter**: 50 Hz removes low-frequency rumble; 0 disables it.
- **Input normalization peak**: controls pre-model peak normalization.
- **CUDA out-of-memory recovery**: retries safely with smaller chunks.

The model always runs on `cuda:0`; CPU fallback is intentionally disabled.

## Command-line start options

`run.bat` is the normal start method. To change the bind address or port:

```bat
set SIDON_SERVER_NAME=0.0.0.0
set SIDON_SERVER_PORT=7861
.venv\Scripts\python.exe app.py
```

Do not expose the WebUI directly to the internet without authentication.

## Model and license

The WebUI downloads model artifacts from
[`sarulab-speech/sidon-v0.1`](https://huggingface.co/sarulab-speech/sidon-v0.1).
See the upstream repository and model page for their respective license terms
and attribution.
