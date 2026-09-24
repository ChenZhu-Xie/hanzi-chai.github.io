import importlib.util
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("unihan-render-match.py")
SPEC = importlib.util.spec_from_file_location("unihan_render_match", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RenderMatchTest(unittest.TestCase):
    def test_acceptance_summary_keeps_held_out_denominator(self):
        held_out = [{"correct": True}, {"correct": False}, {"correct": True}]
        summary = MODULE.acceptance_summary(held_out, [held_out[0]])

        self.assertEqual(summary["evaluated"], 3)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["coverage"], 1 / 3)
        self.assertEqual(summary["accuracyWhenAccepted"], 1)

    def test_unmatched_strokes_use_topology_before_location(self):
        horizontal_fall = {
            "feature": "横撇",
            "start": [20, 20],
            "curveList": [{"command": "h"}, {"command": "c"}],
        }
        fall = {
            "feature": "撇",
            "start": [30, 20],
            "curveList": [{"command": "c"}],
        }
        self.assertEqual(
            MODULE.unmatched_stroke_indices([horizontal_fall], [fall]), {0}
        )

    def test_topology_distance_prefers_matching_direction(self):
        horizontal = np.zeros((64, 64), dtype=bool)
        horizontal[32, 20:45] = True
        vertical = np.zeros((64, 64), dtype=bool)
        vertical[20:45, 32] = True
        strokes = [
            [{"feature": "横", "start": [20, 32], "curveList": [{"command": "h"}]}],
            [{"feature": "竖", "start": [32, 20], "curveList": [{"command": "v"}]}],
        ]
        distances = MODULE.topology_distances(
            horizontal,
            strokes,
            [[horizontal], [vertical]],
            None,
        )
        self.assertLess(distances[0], distances[1])


if __name__ == "__main__":
    unittest.main()
