import importlib.util
import unittest
from pathlib import Path


def load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASCADE = load("unihan-topology-cascade.py")
MATCHER = load("unihan-render-match.py")


class TopologyCascadeTest(unittest.TestCase):
    def test_complete_subtree_component_topology_has_first_priority(self):
        predicted, method, margin = CASCADE.choose(
            {
                "recursiveTopologyPredictionGlyphId": 1,
                "recursiveTopology": {
                    "uniqueStrokeIndices": [[0], [0]],
                    "differingSubtreeStrokeIndices": [[0], [0]],
                },
            },
            {"recursiveTopologyPredictionGlyphId": 2},
            {
                "exactComponentPredictionGlyphId": 3,
                "candidateFeatureDistances": {"1": [1], "3": [0]},
            },
        )

        self.assertEqual(predicted, 3)
        self.assertEqual(method, "complete-subtree-component-topology")
        self.assertEqual(margin, 1)

    def test_single_terminal_stroke_substitution_uses_direction_evidence(self):
        predicted, method, margin = CASCADE.choose(
            {
                "recursiveTopologyPredictionGlyphId": 12,
                "recursiveTopology": {
                    "uniqueStrokeIndices": [[4], [4]],
                    "differingSubtreeStrokeIndices": [[4, 5], [4, 5]],
                },
            },
            {"recursiveTopologyPredictionGlyphId": 13},
            {"exactComponentPredictionGlyphId": None},
        )

        self.assertEqual(predicted, 12)
        self.assertEqual(method, "terminal-stroke-direction")
        self.assertEqual(margin, 0)

    def test_extra_or_missing_strokes_use_complete_recursive_subtree(self):
        predicted, method, margin = CASCADE.choose(
            {
                "recursiveTopologyPredictionGlyphId": 20,
                "recursiveTopology": {
                    "uniqueStrokeIndices": [[0, 1, 2], [0, 1, 2, 3]],
                    "differingSubtreeStrokeIndices": [[0, 1, 2], [0, 1, 2, 3]],
                },
            },
            {"recursiveTopologyPredictionGlyphId": 21},
            {"exactComponentPredictionGlyphId": None},
        )

        self.assertEqual(predicted, 21)
        self.assertEqual(method, "recursive-subtree-topology")
        self.assertEqual(margin, 0)

    def test_safe_threshold_requires_zero_training_errors(self):
        thresholds = CASCADE.calibrate_thresholds(
            [
                {"method": "x", "abstained": False, "margin": 4, "correct": True},
                {"method": "x", "abstained": False, "margin": 3, "correct": True},
                {"method": "x", "abstained": False, "margin": 2, "correct": False},
            ]
        )

        self.assertEqual(thresholds, {"x": 3})

    def test_recursive_leaf_diff_discards_common_hook(self):
        horizontal = {
            "feature": "横",
            "start": [20, 20],
            "curveList": [{"command": "h"}],
        }
        falling = {
            "feature": "撇",
            "start": [20, 20],
            "curveList": [{"command": "c"}],
        }
        hook = {
            "feature": "竖弯钩",
            "start": [20, 30],
            "curveList": [{"command": "v"}, {"command": "h"}],
        }

        unique, groups, subtree = MATCHER.recursive_unique_stroke_indices(
            [[falling, hook], [horizontal, hook]], [[133, 133], [1128, 1128]]
        )

        self.assertEqual(unique, [{0}, {0}])
        self.assertEqual(subtree, [{0, 1}, {0, 1}])
        self.assertEqual([group[0]["id"] for group in groups], [133, 1128])

    def test_same_feature_on_opposite_sides_is_not_cancelled(self):
        left = {
            "feature": "撇",
            "start": [20, 5],
            "curveList": [{"command": "c"}],
        }
        right = {
            "feature": "撇",
            "start": [40, 5],
            "curveList": [{"command": "c"}],
        }

        self.assertEqual(MATCHER.unmatched_stroke_indices([left], [right]), {0})


if __name__ == "__main__":
    unittest.main()
