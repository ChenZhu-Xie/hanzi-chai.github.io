import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_evaluator():
    path = Path(__file__).with_name("unihan-recursive-leaf-hypothesis-eval.py")
    spec = importlib.util.spec_from_file_location(
        "unihan_recursive_leaf_hypothesis_eval", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_evaluator()


class RecursiveLeafHypothesisTests(unittest.TestCase):
    def test_leaf_chamfer_distinguishes_direction(self):
        horizontal = np.zeros((96, 96), dtype=bool)
        vertical = np.zeros_like(horizontal)
        horizontal[46:51, 15:81] = True
        vertical[15:81, 46:51] = True

        self.assertEqual(EVAL.mask_chamfer(horizontal, horizontal), 0)
        self.assertGreater(
            EVAL.mask_chamfer(horizontal, vertical),
            EVAL.mask_chamfer(horizontal, horizontal),
        )

    def test_unassigned_ink_penalizes_an_incomplete_candidate(self):
        pdf = np.zeros((96, 96), dtype=bool)
        pdf[20:76, 20:25] = True
        pdf[20:76, 70:75] = True
        complete = [np.zeros_like(pdf), np.zeros_like(pdf)]
        complete[0][20:76, 20:25] = True
        complete[1][20:76, 70:75] = True
        incomplete = [complete[0]]

        full = EVAL.aggregate_candidate_features(pdf, complete, complete)
        partial = EVAL.aggregate_candidate_features(pdf, incomplete, incomplete)

        self.assertEqual(full["unassignedRatio"], 0)
        self.assertGreater(partial["unassignedRatio"], full["unassignedRatio"])

    def test_weighted_score_prefers_matching_leaf_topology(self):
        matching = {name: 0.0 for name in EVAL.FEATURE_NAMES}
        mismatch = {**matching, "endpointDelta": 2.0, "unassignedRatio": 0.3}
        weights = (1.0, 0.5, 0.5, 0.5, 0.5, 1.0)

        self.assertLess(
            EVAL.weighted_score(matching, weights),
            EVAL.weighted_score(mismatch, weights),
        )

    def test_baseline_comparison_counts_held_out_rescues_and_breaks(self):
        rows = [
            {"unicode": 1, "source": "T", "split": "test", "correct": True},
            {"unicode": 2, "source": "J", "split": "test", "correct": False},
        ]
        baseline = {
            "results": [
                {"unicode": 1, "source": "T", "correct": False},
                {"unicode": 2, "source": "J", "correct": True},
            ]
        }

        comparison = EVAL.compare_with_baseline(rows, baseline)

        self.assertEqual(comparison["test"]["rescues"], 1)
        self.assertEqual(comparison["test"]["breaks"], 1)


if __name__ == "__main__":
    unittest.main()
