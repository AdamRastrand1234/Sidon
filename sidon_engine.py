"""Inference-only engine for Sidon speech restoration.

The original repository contains the complete training stack.  This module
only keeps the small amount of code required to run the published TorchScript
models from ``sarulab-speech/sidon-v0.1`` on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Generator

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
import numpy as np
import soundfile as sf
import torch
import torchaudio


MODEL_REPO_ID = "sarulab-speech/sidon-v0.1"
FEATURE_MODEL_FILE = "feature_extractor_cuda.pt"
DECODER_MODEL_FILE = "decoder_cuda.pt"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 48_000
DEFAULT_END_PADDING_SECONDS = 1.5
DECODER_FRAME_SAMPLES = 960
MIN_CHUNK_SECONDS = 4.0
MAX_CHUNK_SECONDS = 96.0
OUTPUT_FINALIZE_BLOCK_SAMPLES = 1_048_576
OUTPUT_DISK_SAFETY_BYTES = 256 * 1024**2


class SidonError(RuntimeError):
    """Expected, user-readable inference failure."""


@dataclass(frozen=True)
class CudaInfo:
    name: str
    total_gb: float
    free_gb: float
    torch_version: str
    cuda_version: str


@dataclass(frozen=True)
class InferenceOptions:
    auto_chunk: bool = True
    chunk_seconds: float = 60.0
    highpass_hz: float = 50.0
    normalize_input: bool = True
    input_peak: float = 0.9
    oom_recovery: bool = True

    def validated(self) -> "InferenceOptions":
        chunk_seconds = float(self.chunk_seconds)
        highpass_hz = float(self.highpass_hz)
        input_peak = float(self.input_peak)
        if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
            raise SidonError(
                f"Chunk length must be between {MIN_CHUNK_SECONDS:g} and "
                f"{MAX_CHUNK_SECONDS:g} seconds."
            )
        if not 0.0 <= highpass_hz <= 300.0:
            raise SidonError("High-pass cutoff must be between 0 and 300 Hz.")
        if not 0.1 <= input_peak <= 0.99:
            raise SidonError("Input peak must be between 0.10 and 0.99.")
        return InferenceOptions(
            auto_chunk=bool(self.auto_chunk),
            chunk_seconds=chunk_seconds,
            highpass_hz=highpass_hz,
            normalize_input=bool(self.normalize_input),
            input_peak=input_peak,
            oom_recovery=bool(self.oom_recovery),
        )


@dataclass(frozen=True)
class ProgressUpdate:
    percent: int
    message: str
    output_path: str | None = None
    details: str = ""


def get_cuda_info() -> CudaInfo:
    """Return current CUDA device information or raise a readable error."""
    if not torch.cuda.is_available():
        raise SidonError(
            "CUDA is not available. Sidon requires an NVIDIA GPU and a "
            "CUDA-enabled PyTorch installation. Run install.bat first."
        )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        props = torch.cuda.get_device_properties(0)
    except Exception as exc:  # pragma: no cover - depends on driver state
        raise SidonError(f"CUDA device 0 could not be queried: {exc}") from exc
    gib = 1024**3
    return CudaInfo(
        name=props.name,
        total_gb=total_bytes / gib,
        free_gb=free_bytes / gib,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda or "unknown"),
    )


def recommend_chunk_seconds(total_gb: float, free_gb: float | None = None) -> float:
    """Choose a conservative chunk length for the available VRAM.

    The SSL encoder's attention memory grows faster than audio duration, so the
    mapping is deliberately conservative.  Free-memory limits also account for
    browsers, games, or other CUDA programs using the same GPU.
    """
    total_gb = max(0.0, float(total_gb))
    if total_gb < 5.0:
        by_total = 30.0
    elif total_gb < 7.0:
        by_total = 45.0
    elif total_gb < 10.0:
        by_total = 60.0
    elif total_gb < 14.0:
        by_total = 72.0
    elif total_gb < 20.0:
        by_total = 84.0
    elif total_gb < 32.0:
        by_total = 90.0
    else:
        by_total = MAX_CHUNK_SECONDS

    if free_gb is None:
        return by_total

    free_gb = max(0.0, float(free_gb))
    if free_gb < 2.0:
        by_free = 10.0
    elif free_gb < 3.0:
        by_free = 20.0
    elif free_gb < 4.0:
        by_free = 30.0
    elif free_gb < 5.0:
        by_free = 45.0
    elif free_gb < 6.0:
        by_free = 60.0
    elif free_gb < 8.0:
        by_free = 72.0
    elif free_gb < 12.0:
        by_free = 84.0
    elif free_gb < 20.0:
        by_free = 90.0
    else:
        by_free = MAX_CHUNK_SECONDS
    return max(MIN_CHUNK_SECONDS, min(by_total, by_free))


def effective_free_vram_gb(driver_free_gb: float) -> float:
    """Include CUDA allocator cache that PyTorch can reuse without a new allocation."""
    if not torch.cuda.is_available():
        return max(0.0, float(driver_free_gb))
    gib = 1024**3
    cached_but_reusable = max(
        0,
        torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0),
    )
    return max(0.0, float(driver_free_gb)) + cached_but_reusable / gib


def _extract_features(waveform: torch.Tensor) -> torch.Tensor:
    """Match SeamlessM4TFeatureExtractor without the Transformers dependency."""
    waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu").view(1, -1)
    if waveform.shape[-1] < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))

    feature = torchaudio.compliance.kaldi.fbank(
        waveform=waveform,
        sample_frequency=INPUT_SAMPLE_RATE,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        window_type="povey",
        use_energy=False,
        energy_floor=1.192092955078125e-07,
    )
    mean = feature.mean(0, keepdim=True)
    variance = feature.var(0, keepdim=True)
    feature = (feature - mean) / torch.sqrt(variance + 1e-5)

    usable_frames = (feature.shape[0] // 2) * 2
    if usable_frames == 0:
        raise SidonError("The audio segment is too short for feature extraction.")
    return feature[:usable_frames].reshape(1, usable_frames // 2, 160).contiguous()


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _safe_stem(path: str | Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("._")
    return stem[:80] or "audio"


def estimate_output_work_bytes(target_samples: int) -> int:
    """Estimate temporary float WAV plus final PCM-16 WAV disk usage."""
    samples = max(0, int(target_samples))
    return samples * (4 + 2) + OUTPUT_DISK_SAFETY_BYTES


class SidonEngine:
    """Lazy-loading, single-GPU Sidon inference engine."""

    def __init__(self) -> None:
        self._feature_model: torch.jit.ScriptModule | None = None
        self._decoder_model: torch.jit.ScriptModule | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._feature_model is not None and self._decoder_model is not None

    def preload(self) -> Generator[ProgressUpdate, None, None]:
        """Download if needed and load both models on CUDA before serving."""
        yield from self._ensure_models()

    def _cache_dir(self) -> str | None:
        value = os.environ.get("SIDON_MODEL_CACHE", "").strip()
        return str(Path(value).expanduser().resolve()) if value else None

    def _cached_model_file(self, filename: str, token: str | None) -> str | None:
        """Return a cached model path without starting a network download."""
        arguments = {
            "repo_id": MODEL_REPO_ID,
            "filename": filename,
            "token": token,
            "cache_dir": self._cache_dir(),
        }
        try:
            return hf_hub_download(**arguments, local_files_only=True)
        except LocalEntryNotFoundError:
            return None

    def _download_model_file(self, filename: str, token: str | None) -> str:
        """Download a model file that was not found in the local cache."""
        return hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            token=token,
            cache_dir=self._cache_dir(),
        )

    def _ensure_models(
        self,
    ) -> Generator[
        ProgressUpdate, None, tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]
    ]:
        get_cuda_info()
        if self.loaded:
            assert self._feature_model is not None
            assert self._decoder_model is not None
            yield ProgressUpdate(12, "Model already loaded on CUDA.")
            return self._feature_model, self._decoder_model

        with self._load_lock:
            if self.loaded:
                assert self._feature_model is not None
                assert self._decoder_model is not None
                yield ProgressUpdate(12, "Model already loaded on CUDA.")
                return self._feature_model, self._decoder_model

            token = os.environ.get("HF_TOKEN") or None
            feature_path = self._cached_model_file(FEATURE_MODEL_FILE, token)
            if feature_path is None:
                yield ProgressUpdate(
                    3,
                    "Downloading the CUDA feature model "
                    "(first run can take a while)...",
                )
                feature_path = self._download_model_file(FEATURE_MODEL_FILE, token)
            else:
                yield ProgressUpdate(3, "CUDA feature model found in cache.")

            decoder_path = self._cached_model_file(DECODER_MODEL_FILE, token)
            if decoder_path is None:
                yield ProgressUpdate(7, "Downloading the CUDA decoder...")
                decoder_path = self._download_model_file(DECODER_MODEL_FILE, token)
            else:
                yield ProgressUpdate(7, "CUDA decoder found in cache.")

            yield ProgressUpdate(10, "Loading Sidon models into CUDA VRAM...")
            try:
                feature_model = torch.jit.load(
                    feature_path, map_location=torch.device("cuda:0")
                ).eval()
                decoder_model = torch.jit.load(
                    decoder_path, map_location=torch.device("cuda:0")
                ).eval()
            except Exception as exc:
                self._feature_model = None
                self._decoder_model = None
                torch.cuda.empty_cache()
                raise SidonError(
                    f"The CUDA model files could not be loaded: {exc}"
                ) from exc

            self._feature_model = feature_model
            self._decoder_model = decoder_model
            yield ProgressUpdate(15, "Sidon is loaded and ready on CUDA.")
            return feature_model, decoder_model

    @staticmethod
    def _read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
        try:
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise SidonError(
                f"The uploaded audio file could not be read: {exc}"
            ) from exc
        if sample_rate <= 0 or audio.shape[0] == 0:
            raise SidonError("The uploaded audio file is empty.")
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        waveform = torch.from_numpy(audio.mean(axis=1)).view(1, -1)
        if waveform.shape[-1] < max(400, int(sample_rate * 0.1)):
            raise SidonError("The uploaded audio must be at least 0.1 seconds long.")
        if float(waveform.abs().max()) < 1e-7:
            raise SidonError("The uploaded audio is silent.")
        return waveform, int(sample_rate)

    @staticmethod
    def _preprocess(
        waveform: torch.Tensor,
        sample_rate: int,
        options: InferenceOptions,
    ) -> tuple[torch.Tensor, int]:
        target_samples = max(
            1, round(waveform.shape[-1] * OUTPUT_SAMPLE_RATE / sample_rate)
        )
        if options.normalize_input:
            peak = waveform.abs().max().clamp_min(1e-7)
            waveform = waveform * (options.input_peak / peak)
        highpass_hz = min(options.highpass_hz, sample_rate * 0.45)
        if highpass_hz > 0:
            waveform = torchaudio.functional.highpass_biquad(
                waveform, sample_rate, highpass_hz
            )
        if sample_rate != INPUT_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, INPUT_SAMPLE_RATE
            )
        padding = round(DEFAULT_END_PADDING_SECONDS * INPUT_SAMPLE_RATE)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform.contiguous(), target_samples

    def _run_chunks(
        self,
        waveform_16k: torch.Tensor,
        chunk_seconds: float,
        feature_model: torch.jit.ScriptModule,
        decoder_model: torch.jit.ScriptModule,
        temp_output_path: Path,
        target_samples: int,
    ) -> Generator[ProgressUpdate, None, tuple[int, float]]:
        chunk_samples = max(1, round(chunk_seconds * INPUT_SAMPLE_RATE))
        total_samples = waveform_16k.shape[-1]
        num_chunks = max(1, math.ceil(total_samples / chunk_samples))
        feature_cache: torch.Tensor | None = None
        samples_written = 0
        output_peak = 0.0

        with sf.SoundFile(
            str(temp_output_path),
            mode="w",
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            subtype="FLOAT",
            format="WAV",
        ) as temp_output:
            for index, start in enumerate(
                range(0, total_samples, chunk_samples), start=1
            ):
                end = min(start + chunk_samples, total_samples)
                stage_start = 27 + round(64 * (index - 1) / num_chunks)
                stage_mid = 27 + round(64 * (index - 0.5) / num_chunks)
                stage_end = 27 + round(64 * index / num_chunks)
                yield ProgressUpdate(
                    min(stage_start, 90),
                    f"Enhancing chunk {index}/{num_chunks}: extracting speech features...",
                )

                chunk = waveform_16k[0, start:end]
                padded_chunk = torch.nn.functional.pad(chunk, (160, 160))
                input_features = _extract_features(padded_chunk).to(
                    "cuda:0", non_blocking=True
                )

                with torch.inference_mode():
                    feature_output = feature_model(input_features)
                    if isinstance(feature_output, dict):
                        features = feature_output["last_hidden_state"]
                    else:  # TorchScript dictionaries can expose __getitem__ only
                        features = feature_output["last_hidden_state"]
                    if feature_cache is not None:
                        features = torch.cat([feature_cache, features], dim=1)

                    yield ProgressUpdate(
                        min(stage_mid, 91),
                        f"Enhancing chunk {index}/{num_chunks}: reconstructing 48 kHz speech...",
                    )
                    decoded = decoder_model(features.transpose(1, 2)).reshape(-1)
                    if decoded.numel() > DECODER_FRAME_SAMPLES:
                        decoded = decoded[:-DECODER_FRAME_SAMPLES]
                    decoded_cpu = torch.nan_to_num(
                        decoded.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    remaining = max(0, target_samples - samples_written)
                    if remaining:
                        decoded_cpu = decoded_cpu[:remaining]
                        if decoded_cpu.numel():
                            output_peak = max(
                                output_peak, float(decoded_cpu.abs().max())
                            )
                            temp_output.write(decoded_cpu.numpy())
                            samples_written += decoded_cpu.numel()
                    feature_cache = features[:, -1:, :].detach()

                del input_features, feature_output, features, decoded, decoded_cpu
                yield ProgressUpdate(
                    min(stage_end, 92),
                    f"Chunk {index}/{num_chunks} complete.",
                )

            while samples_written < target_samples:
                block_samples = min(
                    OUTPUT_FINALIZE_BLOCK_SAMPLES, target_samples - samples_written
                )
                temp_output.write(np.zeros(block_samples, dtype=np.float32))
                samples_written += block_samples

        if samples_written == 0:
            raise SidonError("No enhanced audio was produced.")
        return num_chunks, output_peak

    @staticmethod
    def _finalize_output(
        temp_output_path: Path,
        partial_output_path: Path,
        output_peak: float,
    ) -> None:
        """Convert the streamed float result to PCM-16 without loading it all."""
        scale = 0.99 / output_peak if output_peak > 0.99 else 1.0
        with (
            sf.SoundFile(str(temp_output_path), mode="r") as source,
            sf.SoundFile(
                str(partial_output_path),
                mode="w",
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=1,
                subtype="PCM_16",
                format="WAV",
            ) as destination,
        ):
            for block in source.blocks(
                blocksize=OUTPUT_FINALIZE_BLOCK_SAMPLES,
                dtype="float32",
                always_2d=False,
            ):
                if scale != 1.0:
                    block *= scale
                destination.write(block)

    def enhance(
        self,
        input_path: str | Path,
        options: InferenceOptions,
        output_dir: str | Path,
    ) -> Generator[ProgressUpdate, None, None]:
        """Enhance one file and yield progress updates suitable for Gradio."""
        started = time.perf_counter()
        options = options.validated()
        input_path = Path(input_path)
        if not input_path.is_file():
            raise SidonError("Please upload an audio file first.")

        with self._inference_lock:
            yield ProgressUpdate(0, "Starting Sidon speech enhancement...")
            feature_model, decoder_model = yield from self._ensure_models()

            yield ProgressUpdate(18, "Reading and validating the audio file...")
            waveform, sample_rate = self._read_audio(input_path)
            input_duration = waveform.shape[-1] / sample_rate

            yield ProgressUpdate(22, "Converting input to mono 16 kHz model audio...")
            waveform_16k, target_samples = self._preprocess(
                waveform, sample_rate, options
            )
            del waveform

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            required_disk_bytes = estimate_output_work_bytes(target_samples)
            free_disk_bytes = shutil.disk_usage(output_dir).free
            if free_disk_bytes < required_disk_bytes:
                required_gb = required_disk_bytes / 1024**3
                free_gb = free_disk_bytes / 1024**3
                raise SidonError(
                    "Not enough free disk space for this audio. "
                    f"About {required_gb:.1f} GB is needed, but only "
                    f"{free_gb:.1f} GB is available."
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = (
                output_dir / f"{_safe_stem(input_path)}_enhanced_{timestamp}.wav"
            )
            temp_output_path = output_dir / f".sidon_{timestamp}.float.wav"
            partial_output_path = output_dir / f".sidon_{timestamp}.partial.wav"

            cuda = get_cuda_info()
            effective_free_gb = effective_free_vram_gb(cuda.free_gb)
            chunk_seconds = (
                recommend_chunk_seconds(cuda.total_gb, effective_free_gb)
                if options.auto_chunk
                else options.chunk_seconds
            )
            chunk_seconds = max(
                MIN_CHUNK_SECONDS, min(MAX_CHUNK_SECONDS, chunk_seconds)
            )
            initial_chunk_seconds = chunk_seconds
            yield ProgressUpdate(
                25,
                f"Using {chunk_seconds:g}-second chunks for "
                f"{cuda.name} ({cuda.total_gb:.1f} GB VRAM).",
            )

            torch.cuda.reset_peak_memory_stats(0)
            retries = 0
            try:
                while True:
                    temp_output_path.unlink(missing_ok=True)
                    try:
                        num_chunks, output_peak = yield from self._run_chunks(
                            waveform_16k,
                            chunk_seconds,
                            feature_model,
                            decoder_model,
                            temp_output_path,
                            target_samples,
                        )
                        break
                    except Exception as exc:
                        if not _is_cuda_oom(exc):
                            raise
                        del exc
                        torch.cuda.empty_cache()
                        if (
                            not options.oom_recovery
                            or chunk_seconds <= MIN_CHUNK_SECONDS
                        ):
                            raise SidonError(
                                "CUDA ran out of memory even at the minimum chunk "
                                "length. Close other GPU applications and try again."
                            )
                        retries += 1
                        new_chunk_seconds = max(
                            MIN_CHUNK_SECONDS,
                            float(max(1, math.floor(chunk_seconds / 2))),
                        )
                        if new_chunk_seconds >= chunk_seconds:
                            new_chunk_seconds = MIN_CHUNK_SECONDS
                        chunk_seconds = new_chunk_seconds
                        yield ProgressUpdate(
                            26,
                            "CUDA memory was insufficient; safely retrying with "
                            f"{chunk_seconds:g}-second chunks...",
                        )

                yield ProgressUpdate(
                    94, "Finalizing waveform and preventing clipping..."
                )
                partial_output_path.unlink(missing_ok=True)
                yield ProgressUpdate(97, "Saving enhanced speech as 48 kHz WAV...")
                self._finalize_output(
                    temp_output_path,
                    partial_output_path,
                    output_peak,
                )
                partial_output_path.replace(output_path)

                elapsed = time.perf_counter() - started
                peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1024**3
                details = (
                    f"**Finished in {elapsed:.1f} s**  \n"
                    f"Input: {input_duration:.1f} s at {sample_rate:,} Hz  \n"
                    f"Output: 48,000 Hz mono WAV  \n"
                    f"GPU: {cuda.name}  \n"
                    f"Chunk length: {chunk_seconds:g} s ({num_chunks} chunks)"
                    + (
                        f" — reduced from {initial_chunk_seconds:g} s after "
                        f"{retries} VRAM retry/retries"
                        if retries
                        else ""
                    )
                    + f"  \nPeak PyTorch VRAM: {peak_vram_gb:.2f} GB"
                )
                yield ProgressUpdate(
                    100,
                    "Enhancement complete.",
                    output_path=str(output_path.resolve()),
                    details=details,
                )
            finally:
                temp_output_path.unlink(missing_ok=True)
                partial_output_path.unlink(missing_ok=True)


ENGINE = SidonEngine()
