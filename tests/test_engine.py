from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf
import torch

from sidon_engine import (
    InferenceOptions,
    SidonEngine,
    SidonError,
    _extract_features,
    effective_free_vram_gb,
    recommend_chunk_seconds,
)


class EngineUnitTests(unittest.TestCase):
    def test_vram_chunk_recommendations_are_monotonic(self) -> None:
        totals = [4, 6, 8, 12, 16, 24, 32]
        recommendations = [recommend_chunk_seconds(total) for total in totals]
        self.assertEqual(recommendations, sorted(recommendations))
        self.assertEqual(recommend_chunk_seconds(8), 60)
        self.assertEqual(recommend_chunk_seconds(32), 96)

    def test_low_free_vram_reduces_chunk_length(self) -> None:
        self.assertEqual(recommend_chunk_seconds(24, free_gb=3.5), 30)
        self.assertEqual(recommend_chunk_seconds(8, free_gb=7.0), 60)

    def test_effective_free_vram_is_never_lower_than_driver_free(self) -> None:
        self.assertGreaterEqual(effective_free_vram_gb(3.25), 3.25)

    def test_options_validation(self) -> None:
        self.assertEqual(InferenceOptions().validated().chunk_seconds, 60)
        with self.assertRaises(SidonError):
            InferenceOptions(chunk_seconds=2).validated()

    def test_feature_shape_matches_w2v_bert_input(self) -> None:
        waveform = torch.randn(16_000)
        features = _extract_features(waveform)
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 160)
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_audio_reader_converts_stereo_to_mono(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            stereo = np.stack(
                [
                    np.linspace(-0.5, 0.5, 16_000),
                    np.linspace(0.5, -0.5, 16_000),
                ],
                axis=1,
            ).astype(np.float32)
            sf.write(str(path), stereo, 16_000)
            waveform, sample_rate = SidonEngine._read_audio(path)
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(tuple(waveform.shape), (1, 16_000))


if __name__ == "__main__":
    unittest.main()
