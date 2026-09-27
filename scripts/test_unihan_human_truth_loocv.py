import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("unihan-human-truth-loocv.py")
SPEC = importlib.util.spec_from_file_location("human_truth_loocv", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HumanTruthLeaveOneOutTest(unittest.TestCase):
    def case(self, key, vector):
        return MODULE.FeatureCase(
            key=key,
            candidate_ids=(10, 20),
            vector=np.asarray(vector, dtype=float),
            global_scores={10: 2.0, 20: 1.0},
            focus_scores={10: 2.0, 20: 1.0},
            focus_metric_votes={10: 1, 20: 4},
        )

    def test_prediction_interface_cannot_receive_heldout_truth(self):
        cases = {
            "train-a": self.case("train-a", [1.0, 0.0]),
            "train-b": self.case("train-b", [-1.0, 0.0]),
            "heldout": self.case("heldout", [0.5, 0.0]),
        }
        first = MODULE.predict_fold(cases, {"train-a": 20, "train-b": 10}, "heldout")
        second = MODULE.predict_fold(cases, {"train-a": 20, "train-b": 10}, "heldout")
        self.assertEqual(first["predictedGlyphId"], second["predictedGlyphId"])
        self.assertNotIn("expectedGlyphId", first)

    def test_pairwise_model_is_invariant_to_candidate_dictionary_order(self):
        cases = {
            "a": self.case("a", [1.0, 0.0]),
            "b": self.case("b", [2.0, 0.0]),
        }
        result = MODULE.predict_fold(cases, {"a": 20}, "b")
        self.assertEqual(result["predictedGlyphId"], 20)
        self.assertEqual(result["safePredictionGlyphId"], 20)

    def test_model_has_no_candidate_id_majority_intercept(self):
        cases = {
            "a": self.case("a", [0.0, 0.0]),
            "b": self.case("b", [0.0, 0.0]),
            "heldout": self.case("heldout", [0.0, 0.0]),
        }
        result = MODULE.predict_fold(cases, {"a": 20, "b": 20}, "heldout")
        self.assertEqual(result["decision"], 0.0)
        self.assertEqual(result["predictedGlyphId"], 10)

    def test_feature_loader_discards_embedded_answer_fields(self):
        candidate_metrics = {
            "score": 1,
            **{name: 1 for name in MODULE.FEATURE_NAMES},
        }
        payload = {
            "results": [
                {
                    "key": "U+TEST-X.png",
                    "expectedGlyphId": 999,
                    "predictedGlyphId": 999,
                    "correct": True,
                    "candidates": {"10": candidate_metrics, "20": candidate_metrics},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            path.write_text(json.dumps(payload), "utf-8")
            first = MODULE._measurement_rows(path)
            payload["results"][0]["expectedGlyphId"] = 10
            payload["results"][0]["correct"] = False
            path.write_text(json.dumps(payload), "utf-8")
            second = MODULE._measurement_rows(path)
        self.assertEqual(first, second)

    def test_annotation_key_ignores_candidate_id(self):
        self.assertEqual(
            MODULE.annotation_key(Path("U+66DA-N-57149-annotations.json")),
            "U+66DA-N.png",
        )
        self.assertEqual(
            MODULE.annotation_key(Path("u6418-g-annotations.json")),
            "U+6418-G.png",
        )

    def test_sibling_component_is_normalized_to_selected_candidate_leaf(self):
        components, corrections = MODULE.normalize_component_ids(
            "U+65E8-T.png",
            127685,
            {133, 506},
            {
                0x65E8: {
                    "candidateLeafIds": {127685: {1128, 506}},
                    "focusLeafIds": {133, 1128},
                }
            },
        )
        self.assertEqual(components, [506, 1128])
        self.assertEqual(corrections, [{"from": 133, "to": 1128}])


if __name__ == "__main__":
    unittest.main()
