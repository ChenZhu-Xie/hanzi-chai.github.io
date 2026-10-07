import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unihan_experts.models import CaseManifest, ExpertManifest
from unihan_experts.registry import (
    load_case_registry,
    load_expert_registry,
    load_shared_artifacts,
    resolve_annotation,
    validate_expert_registry,
)


EXPERTS_PATH = SCRIPT_DIR / "unihan-experts.json"
CASES_PATH = SCRIPT_DIR / "unihan-expert-cases.json"


def make_expert(expert_id, *, parents=(), commit="a1b2c3d", adapter="classic-v1"):
    return ExpertManifest(
        expert_id=expert_id,
        schema_version=1,
        commit=commit,
        adapter=adapter,
        decoder="legacy" if adapter == "classic-v1" else "hybrid",
        config={},
        rule_artifacts=(),
        parents=parents,
        declared_capabilities=(),
        resource_class="small",
    )


class ExpertRegistryTests(unittest.TestCase):
    def test_tracked_registry_contains_expected_historical_experts(self):
        experts = load_expert_registry(EXPERTS_PATH)
        self.assertEqual(
            {expert.expert_id for expert in experts},
            {
                "classic-15c",
                "junction-23b",
                "scan-front-7fc",
                "grass-005",
                "recursive-ids-8c",
                "complete-route-2272",
                "contact-dd",
                "reserved-dad",
            },
        )

    def test_duplicate_expert_ids_are_rejected(self):
        expert = make_expert("same")
        with self.assertRaisesRegex(ValueError, "duplicate expert"):
            validate_expert_registry((expert, expert))

    def test_missing_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing parent"):
            validate_expert_registry((make_expert("child", parents=("parent",)),))

    def test_parent_cycle_is_rejected(self):
        experts = (
            make_expert("left", parents=("right",)),
            make_expert("right", parents=("left",)),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_expert_registry(experts)

    def test_unresolved_commit_is_rejected(self):
        def resolver(_revision):
            raise ValueError("unknown revision")

        with self.assertRaisesRegex(ValueError, "cannot resolve commit"):
            validate_expert_registry((make_expert("one"),), commit_resolver=resolver)

    def test_adapter_decoder_mismatch_is_rejected(self):
        expert = make_expert("wrong", adapter="centroid-v2")
        expert = ExpertManifest(
            **{**expert.__dict__, "decoder": "ensemble"}
        )
        with self.assertRaisesRegex(ValueError, "decoder"):
            validate_expert_registry((expert,))


class CaseRegistryTests(unittest.TestCase):
    def test_shared_artifacts_are_repo_relative(self):
        artifacts = load_shared_artifacts(CASES_PATH)
        self.assertEqual(set(artifacts), {"pdf", "bbox", "candidates"})
        self.assertTrue(all(not path.is_absolute() for path in artifacts.values()))

    def test_tracked_registry_contains_twelve_cases_once(self):
        cases = load_case_registry(CASES_PATH)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertIn("u66da-n-57149", {case.case_id for case in cases})

    def test_duplicate_case_ids_are_rejected(self):
        payload = {
            "schemaVersion": 1,
            "cases": [
                {
                    "caseId": "u6418-g-17973",
                    "unicode": "U+6418",
                    "source": "G",
                    "glyphId": 17973,
                    "annotationBasename": "U+6418-G-17973-annotations.json",
                }
            ]
            * 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case"):
                load_case_registry(path)

    def test_annotation_is_resolved_only_from_explicit_roots(self):
        case = CaseManifest(
            case_id="u6418-g-17973",
            unicode="U+6418",
            source="G",
            glyph_id=17973,
            annotation_basename="U+6418-G-17973-annotations.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / case.annotation_basename
            annotation.write_text("[]", encoding="utf-8")
            self.assertEqual(resolve_annotation(case, (root,)), annotation.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_annotation(case, (root / "elsewhere",))


if __name__ == "__main__":
    unittest.main()
