from __future__ import annotations

import unittest

from app import _clear_ui, _progress_html
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


if __name__ == "__main__":
    unittest.main()
