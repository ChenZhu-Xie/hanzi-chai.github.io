import importlib.util
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("unihan-render-match.py")
SPEC = importlib.util.spec_from_file_location("unihan_render_match", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RenderMatchTest(unittest.TestCase):
    def test_shared_residual_flags_a_branch_moved_to_the_other_side(self):
        candidate = np.zeros((64, 64), dtype=bool)
        candidate[24:51, 32] = True
        candidate[24, 18:33] = True
        pdf = np.zeros((64, 64), dtype=bool)
        pdf[24:51, 32] = True
        pdf[24, 32:47] = True

        report = MODULE.shared_structure_residual(pdf, [candidate, candidate])

        self.assertTrue(report["suspected"])
        self.assertEqual(report["reason"], "directional-corner-mismatch")
        self.assertGreater(len(report["directionalCornerMismatches"]), 0)

    def test_shared_residual_accepts_matching_common_geometry(self):
        candidate = np.zeros((64, 64), dtype=bool)
        candidate[24:51, 32] = True
        candidate[24, 18:33] = True

        report = MODULE.shared_structure_residual(candidate, [candidate, candidate])

        self.assertFalse(report["suspected"])

    def test_shared_residual_ignores_unvalidated_vertical_turn_reversal(self):
        candidate = np.zeros((64, 64), dtype=bool)
        candidate[24:51, 32] = True
        candidate[24, 18:33] = True
        pdf = np.zeros((64, 64), dtype=bool)
        pdf[12:25, 32] = True
        pdf[24, 18:33] = True

        report = MODULE.shared_structure_residual(pdf, [candidate, candidate])

        self.assertFalse(report["suspected"])

    def test_candidate_consensus_tolerates_one_pixel_raster_shift(self):
        first = np.zeros((64, 64), dtype=bool)
        first[24:51, 32] = True
        first[24, 18:33] = True
        shifted = np.zeros((64, 64), dtype=bool)
        shifted[25:52, 33] = True
        shifted[25, 19:34] = True

        consensus = MODULE.candidate_consensus([first, shifted])

        self.assertTrue(np.array_equal(consensus, first))

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
