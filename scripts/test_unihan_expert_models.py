import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from unihan_experts.hashing import canonical_json_bytes, sha256_json
from unihan_experts.models import (
    ArtifactRef,
    CaseManifest,
    EvaluationRecord,
    ExpertManifest,
    PredictionRecord,
    expert_content_identity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GIT_COMMIT = "d" * 40


def make_expert(**overrides):
    values = {
        "expert_id": "classic-15c",
        "schema_version": 1,
        "commit": "15c2b04",
        "adapter": "classic-v1",
        "decoder": "legacy",
        "config": {"beamWidth": 24, "coverageWeight": 12.0},
        "rule_artifacts": (
            ArtifactRef(alias="rules", path=".local/rules.json", sha256=SHA_A),
        ),
        "parents": (),
        "declared_capabilities": ("complete-stroke",),
        "resource_class": "medium",
    }
    values.update(overrides)
    return ExpertManifest(**values)


class CanonicalHashTests(unittest.TestCase):
    def test_canonical_json_hash_is_key_order_independent(self):
        left = {"z": [1, 2], "a": {"two": 2, "one": 1}}
        right = {"a": {"one": 1, "two": 2}, "z": [1, 2]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(sha256_json(left), sha256_json(right))

    def test_canonical_json_rejects_non_finite_numbers(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    canonical_json_bytes({"score": value})

    def test_changing_config_changes_expert_content_identity(self):
        left = make_expert()
        right = make_expert(config={"beamWidth": 48, "coverageWeight": 12.0})
        identity_args = {
            "resolved_commit": GIT_COMMIT,
            "source_tree_hash": SHA_C,
            "interpreter_identity": {"executable": "python", "version": "3.13"},
        }
        self.assertNotEqual(
            expert_content_identity(left, **identity_args),
            expert_content_identity(right, **identity_args),
        )

    def test_changing_rule_hash_changes_expert_content_identity(self):
        left = make_expert()
        right = make_expert(
            rule_artifacts=(
                ArtifactRef(alias="rules", path=".local/rules.json", sha256=SHA_B),
            )
        )
        identity_args = {
            "resolved_commit": GIT_COMMIT,
            "source_tree_hash": SHA_C,
            "interpreter_identity": {"executable": "python", "version": "3.13"},
        }
        self.assertNotEqual(
            expert_content_identity(left, **identity_args),
            expert_content_identity(right, **identity_args),
        )

    def test_expert_identity_accepts_full_git_object_id(self):
        identity = expert_content_identity(
            make_expert(),
            resolved_commit=GIT_COMMIT,
            source_tree_hash=SHA_C,
            interpreter_identity={"version": "3.13"},
        )
        self.assertRegex(identity, r"^[0-9a-f]{64}$")


class ManifestValidationTests(unittest.TestCase):
    def test_case_manifest_accepts_basename(self):
        case = CaseManifest(
            case_id="u6418-g-17973",
            unicode="U+6418",
            source="G",
            glyph_id=17973,
            annotation_basename="U+6418-G-17973-annotations.json",
        )
        self.assertEqual(case.annotation_basename, "U+6418-G-17973-annotations.json")

    def test_case_manifest_rejects_absolute_annotation_path(self):
        with self.assertRaisesRegex(ValueError, "basename"):
            CaseManifest(
                case_id="u6418-g-17973",
                unicode="U+6418",
                source="G",
                glyph_id=17973,
                annotation_basename=r"D:\C2D\Downloads\U+6418.json",
            )

    def test_prediction_record_rejects_truth_during_inference(self):
        with self.assertRaisesRegex(ValueError, "truth"):
            PredictionRecord(
                schema_version=1,
                expert_id="classic-15c",
                expert_content_id=SHA_A,
                case_id="u6418-g-17973",
                input_hashes={"pdf": SHA_B},
                semantic_command_hash=SHA_C,
                status="complete",
                prediction={"routes": []},
                uses_human_truth=True,
                truth_first_read_phase="inference",
            )

    def test_evaluation_record_requires_prediction_fingerprint(self):
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            EvaluationRecord(
                schema_version=1,
                expert_id="classic-15c",
                case_id="u6418-g-17973",
                prediction_fingerprint="",
                metrics={},
            )


if __name__ == "__main__":
    unittest.main()
