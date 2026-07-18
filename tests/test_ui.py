from __future__ import annotations

import unittest

from app import _clear_ui, _progress_html, build_demo
from sidon_engine import ProgressUpdate


class AccessibilityUnitTests(unittest.TestCase):
    def test_ready_status_has_screen_reader_semantics(self) -> None:
        markup = _progress_html(ProgressUpdate(0, "Ready. Upload audio to begin."))
        self.assertIn('role="status"', markup)
        self.assertIn('aria-live="polite"', markup)
        self.assertIn('role="progressbar"', markup)
        self.assertIn('aria-valuenow="0"', markup)
        self.assertIn("<span>Ready</span>", markup)

    def test_progress_exposes_numeric_percent_and_escapes_message(self) -> None:
        markup = _progress_html(ProgressUpdate(42, "Chunk <one>"))
        self.assertIn('aria-valuenow="42"', markup)
        self.assertIn("42%", markup)
        self.assertIn("Chunk &lt;one&gt;", markup)
        self.assertNotIn("Chunk <one>", markup)

    def test_error_uses_assertive_alert(self) -> None:
        markup = _progress_html(ProgressUpdate(0, "CUDA failed"), error=True)
        self.assertIn('role="alert"', markup)
        self.assertIn('aria-live="assertive"', markup)
        self.assertIn("<span>Error</span>", markup)

    def test_clear_restores_ready_state(self) -> None:
        input_audio, output_audio, markup, details = _clear_ui()
        self.assertIsNone(input_audio)
        self.assertIsNone(output_audio)
        self.assertIn("<span>Ready</span>", markup)
        self.assertEqual(details, "")

    def test_enhanced_result_uses_read_only_gradio_audio_player(self) -> None:
        config = build_demo().get_config_file()
        audio_components = [
            component
            for component in config["components"]
            if component["type"] == "audio"
        ]
        output = next(
            component
            for component in audio_components
            if component["props"]["label"] == "Step 3 — Enhanced speech result"
        )
        self.assertEqual(output["props"]["type"], "filepath")
        self.assertEqual(output["props"]["format"], "wav")
        self.assertFalse(output["props"]["interactive"])
        self.assertFalse(output["props"]["editable"])


if __name__ == "__main__":
    unittest.main()
