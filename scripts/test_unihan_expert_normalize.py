import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
FIXTURES = SCRIPT_DIR / "fixtures" / "expert-audits"
sys.path.insert(0, str(SCRIPT_DIR))

from unihan_experts.hashing import sha256_file
from unihan_experts.normalize import (
    normalize_artifacts,
    normalize_audit,
    prediction_fingerprint,
)


class NormalizationTests(unittest.TestCase):
    def load(self, name):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def test_classic_success_preserves_unknown_metrics_as_null(self):
        normalized = normalize_audit(
            self.load("classic-success.json"),
            expert_id="classic-15c",
            case_id="u65e8-t-127685",
        )
        self.assertEqual(normalized["status"], "complete")
        self.assertEqual(normalized["predictedStrokeCount"], 2)
        self.assertIsNone(normalized["expectedStrokeCount"])
        self.assertIsNone(normalized["metrics"]["strokeMacroIoU"])

    def test_evaluation_metrics_and_expected_count_are_normalized(self):
        normalized = normalize_audit(
            self.load("centroid-success.json"),
            expert_id="grass-005",
            case_id="u64ec-t-900021",
        )
        self.assertEqual(normalized["expectedStrokeCount"], 2)
        self.assertEqual(normalized["predictedStrokeCount"], 2)
        self.assertTrue(normalized["complete"])
        self.assertEqual(normalized["metrics"]["strokeMacroIoU"], 0.91)
        self.assertEqual(normalized["metrics"]["componentMacroIoU"], 0.95)
        self.assertEqual(normalized["truthFirstReadPhase"], "evaluation")

    def test_partial_prediction_is_never_promoted_by_good_prefix_metric(self):
        normalized = normalize_audit(
            self.load("partial.json"),
            expert_id="head",
            case_id="partial-case",
        )
        self.assertEqual(normalized["status"], "partial")
        self.assertFalse(normalized["complete"])
        self.assertEqual(normalized["predictedStrokeCount"], 1)
        self.assertEqual(normalized["expectedStrokeCount"], 2)

    def test_native_evidence_is_preserved(self):
        raw = self.load("reservation-success.json")
        normalized = normalize_audit(
            raw, expert_id="reserved-dad", case_id="reserved-case"
        )
        self.assertEqual(normalized["nativeEvidence"]["decision"], raw["decision"])
        self.assertEqual(
            normalized["nativeEvidence"]["residualReview"], raw["residualReview"]
        )

    def test_fingerprint_ignores_evaluation_paths_and_resources(self):
        inference = self.load("classic-success.json")
        evaluated = json.loads(json.dumps(inference))
        evaluated["output"] = r"D:\different\evaluated.html"
        evaluated["evaluation"] = {
            "truthReadAfterPrediction": True,
            "strokeMacroIoU": 0.99,
            "perStrokeIoU": [0.99, 0.99],
        }
        evaluated["resources"] = {"wallSeconds": 999}
        self.assertEqual(
            prediction_fingerprint(inference), prediction_fingerprint(evaluated)
        )

    def test_fingerprint_changes_when_selected_steps_change(self):
        left = self.load("classic-success.json")
        right = json.loads(json.dumps(left))
        right["decision"]["steps"][0]["selectedRoute"] = 99
        self.assertNotEqual(
            prediction_fingerprint(left), prediction_fingerprint(right)
        )

    def test_missing_audit_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = normalize_artifacts(
                root / "missing.json",
                root / "missing.html",
                expert_id="classic-15c",
                case_id="case",
            )
        self.assertEqual(result["status"], "missing-output")

    def test_malformed_audit_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            html = root / "review.html"
            audit.write_text("{bad", encoding="utf-8")
            html.write_text("<html></html>", encoding="utf-8")
            result = normalize_artifacts(
                audit,
                html,
                expert_id="classic-15c",
                case_id="case",
            )
        self.assertEqual(result["status"], "malformed-audit")

    def test_raw_artifacts_are_hashed_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            html = root / "review.html"
            audit.write_bytes((FIXTURES / "classic-success.json").read_bytes())
            html.write_bytes(b"<html>raw</html>\r\n")
            before = (audit.read_bytes(), html.read_bytes())
            normalized = normalize_artifacts(
                audit,
                html,
                expert_id="classic-15c",
                case_id="case",
            )
            self.assertEqual(before, (audit.read_bytes(), html.read_bytes()))
            self.assertEqual(normalized["rawArtifacts"]["auditSha256"], sha256_file(audit))
            self.assertEqual(normalized["rawArtifacts"]["htmlSha256"], sha256_file(html))

    def test_non_object_json_is_malformed(self):
        normalized = normalize_audit(
            self.load("malformed.json"), expert_id="one", case_id="case"
        )
        self.assertEqual(normalized["status"], "malformed-audit")


if __name__ == "__main__":
    unittest.main()
