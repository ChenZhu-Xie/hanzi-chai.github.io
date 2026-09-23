from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("unihan-pdf-evidence.py")
SPEC = importlib.util.spec_from_file_location("unihan_pdf_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PDF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PDF)


class ThresholdCalibrationTest(unittest.TestCase):
    def test_calibrates_strict_and_likely_thresholds_from_the_same_labels(self):
        labels = [
            (0.10, True),
            (0.20, True),
            (0.25, False),
            (0.30, True),
            (0.40, True),
        ]

        strict_threshold, strict = PDF.choose_threshold(labels, 0.995)
        likely_threshold, likely = PDF.choose_threshold(labels, 0.80)

        self.assertEqual(strict_threshold, 0.20)
        self.assertEqual(strict["precision"], 1.0)
        self.assertEqual(likely_threshold, 0.40)
        self.assertEqual(likely["precision"], 0.80)
        self.assertGreater(likely["recall"], strict["recall"])

    def test_requires_enough_positive_support_for_a_threshold(self):
        labels = [(0.1, True), (0.2, True), (0.3, False)]

        with self.assertRaisesRegex(RuntimeError, "positive pairs"):
            PDF.choose_threshold(
                labels,
                precision_target=0.95,
                minimum_true_positive=3,
            )


if __name__ == "__main__":
    unittest.main()
