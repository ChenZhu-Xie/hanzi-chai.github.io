import importlib.util
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("unihan-render-match.py")
SPEC = importlib.util.spec_from_file_location("unihan_render_match", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RenderMatchTest(unittest.TestCase):
    def test_vector_page_crop_changes_only_root_viewbox(self):
        page = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="612pt" '
            'height="792pt" viewBox="0 0 612 792"><defs><path id="g"/></defs>'
            '<use href="#g" x="10" y="20"/></svg>'
        )

        cropped = MODULE.crop_page_svg(page, [100, 200, 120, 230], margin=0)

        self.assertIn('width="30.0000pt"', cropped)
        self.assertIn('height="30.0000pt"', cropped)
        self.assertIn('viewBox="95.0000 200.0000 30.0000 30.0000"', cropped)
        self.assertIn('<use href="#g" x="10" y="20"/>', cropped)

    def test_chart_glyph_bbox_removes_lower_source_reference(self):
        self.assertEqual(
            MODULE.chart_glyph_bbox([10, 20, 30, 120]),
            [10, 20, 30, 98.0],
        )

    def test_graph_signature_distinguishes_line_from_t_junction(self):
        line = np.zeros((64, 64), dtype=bool)
        line[32, 10:55] = True
        tee = line.copy()
        tee[15:33, 32] = True

        line_signature = MODULE.graph_topology_signature(line)
        tee_signature = MODULE.graph_topology_signature(tee)

        self.assertEqual(line_signature["endpoints"], 2)
        self.assertEqual(line_signature["junctions"], 0)
        self.assertEqual(tee_signature["endpoints"], 3)
        self.assertEqual(tee_signature["junctions"], 1)

    def test_principal_axis_separates_horizontal_from_falling_stroke(self):
        horizontal = np.zeros((64, 64), dtype=bool)
        horizontal[32, 10:55] = True
        falling = np.zeros((64, 64), dtype=bool)
        for offset in range(40):
            falling[10 + offset, 50 - offset] = True

        horizontal_angle = MODULE.principal_axis_angle(horizontal)
        falling_angle = MODULE.principal_axis_angle(falling)

        self.assertLess(MODULE.axis_angle_distance(horizontal_angle, 0), 1)
        self.assertGreater(
            MODULE.axis_angle_distance(horizontal_angle, falling_angle), 30
        )

    def test_topology_signature_requires_all_invariants_to_agree(self):
        target = {"components": 6, "endpoints": 16, "junctions": 6}
        decision = MODULE.topology_signature_choice(
            target,
            [
                {"components": 3, "endpoints": 7, "junctions": 1},
                {"components": 5, "endpoints": 12, "junctions": 2},
            ],
        )

        self.assertEqual(decision["candidateIndex"], 1)
        self.assertGreater(decision["margin"], 3)

    def test_topology_signature_abstains_when_invariants_disagree(self):
        self.assertIsNone(
            MODULE.topology_signature_choice(
                {"components": 2, "endpoints": 8, "junctions": 1},
                [
                    {"components": 2, "endpoints": 2, "junctions": 0},
                    {"components": 5, "endpoints": 8, "junctions": 4},
                ],
            )
        )

    def test_closest_component_rejects_larger_neighbouring_noise(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 8] = True
        mask[40:44, 40] = True
        anchor = np.zeros((64, 64), dtype=bool)
        anchor[40:44, 41] = True

        selected = MODULE.closest_component(mask, anchor)

        self.assertEqual(int(selected.sum()), 4)
        self.assertTrue(selected[40, 40])

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

    def test_confidence_threshold_never_splits_an_equal_margin_group(self):
        threshold = MODULE.confidence_threshold(
            [
                {"margin": 5, "correct": True},
                {"margin": 5, "correct": False},
                {"margin": 4, "correct": True},
            ]
        )

        self.assertIsNone(threshold)

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
