import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unihan_experts.models import CaseManifest, ExpertManifest
from unihan_experts.runner import CellSpec, merge_matrix_records, run_cell, run_matrix


SHA_A = "a" * 64
SHA_B = "b" * 64


FAKE_EXPERT = r'''
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("bbox-cache", "pdf", "candidates", "unicode", "source", "glyph-id", "canvas", "ordinal-weight", "coverage-weight", "decoder", "beam-width", "output", "audit-output", "annotations"):
    parser.add_argument("--" + name)
args, _unknown = parser.parse_known_args()

candidates = Path(args.candidates)
mode = candidates.read_text(encoding="utf-8").strip()
counter = candidates.with_name("counter.txt")
with counter.open("a", encoding="utf-8") as stream:
    stream.write("evaluation\n" if args.annotations else "inference\n")
if mode == "crash":
    raise SystemExit(7)
if mode == "missing":
    raise SystemExit(0)

selected = 1
evaluation = None
if args.annotations:
    truth = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if truth.get("changePrediction"):
        selected = 99
    evaluation = {
        "truthReadAfterPrediction": True,
        "strokeMacroIoU": 0.9,
        "componentMacroIoU": 0.8,
        "perStrokeIoU": [0.9]
    }
audit = {
    "unicode": args.unicode,
    "source": args.source,
    "glyphId": int(args.glyph_id),
    "output": args.output,
    "decision": {
        "model": "fake",
        "status": "safe-candidate",
        "reviewReasons": [],
        "steps": [{"stroke": 1, "selectedRoute": selected}]
    },
    "evaluation": evaluation
}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text("<html>fake</html>", encoding="utf-8")
if mode == "malformed":
    Path(args.audit_output).write_text("{bad", encoding="utf-8")
else:
    Path(args.audit_output).write_text(json.dumps(audit), encoding="utf-8")
'''


class RunnerFixture:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.worktree = self.root / "worktree"
        script = self.worktree / "scripts" / "unihan-ink-diffusion.py"
        script.parent.mkdir(parents=True)
        script.write_text(textwrap.dedent(FAKE_EXPERT), encoding="utf-8")
        self.shared = {}
        for alias, name in (
            ("pdf", "input.pdf"),
            ("bbox", "bbox.html"),
            ("candidates", "candidates.json"),
        ):
            path = self.root / name
            path.write_text("normal", encoding="utf-8")
            self.shared[alias] = path
        self.annotation = self.root / "truth.json"
        self.annotation.write_text("{}", encoding="utf-8")
        self.expert = ExpertManifest(
            expert_id="fake-expert",
            schema_version=1,
            commit="deadbee",
            adapter="classic-v1",
            decoder="legacy",
            config={"beamWidth": 350},
            rule_artifacts=(),
            parents=(),
            declared_capabilities=(),
            resource_class="small",
        )
        self.case = CaseManifest(
            case_id="u6418-g-17973",
            unicode="U+6418",
            source="G",
            glyph_id=17973,
            annotation_basename="truth.json",
        )
        return self

    def spec(self, case=None):
        return CellSpec(
            expert=self.expert,
            case=case or self.case,
            expert_content_id=SHA_A,
            input_hashes={"pdf": SHA_B, "bbox": SHA_B, "candidates": SHA_B},
            python_executable=Path(sys.executable),
            interpreter_identity={"version": sys.version},
            worktree=self.worktree,
            shared_artifacts=self.shared,
            resolved_rule_artifacts={},
            run_root=self.root / "run",
            timeout_seconds=5,
        )

    def __exit__(self, exc_type, exc, traceback):
        self.temp.cleanup()


class RunnerTests(unittest.TestCase):
    def test_matrix_merge_replaces_same_cell_and_keeps_other_batches(self):
        existing = [
            {"expertId": "alpha", "caseId": "a", "status": "old"},
            {"expertId": "beta", "caseId": "a", "status": "complete"},
        ]
        incoming = [
            {"expertId": "alpha", "caseId": "a", "status": "complete"},
            {"expertId": "alpha", "caseId": "b", "status": "complete"},
        ]
        merged = merge_matrix_records(existing, incoming)
        self.assertEqual(
            [(item["expertId"], item["caseId"], item["status"]) for item in merged],
            [
                ("alpha", "a", "complete"),
                ("alpha", "b", "complete"),
                ("beta", "a", "complete"),
            ],
        )

    def test_prediction_is_persisted_before_annotation_is_resolved(self):
        with RunnerFixture() as fixture:
            spec = fixture.spec()
            observed = []

            def resolver(_case):
                prediction = (
                    spec.run_root
                    / spec.expert.expert_id
                    / spec.case.case_id
                    / "inference"
                    / "prediction.json"
                )
                observed.append(prediction.is_file())
                return fixture.annotation

            result = run_cell(spec, annotation_resolver=resolver)
            self.assertEqual(observed, [True])
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["inference"]["usesHumanTruth"])
            self.assertTrue(result["evaluation"]["usesHumanTruth"])

    def test_truth_changed_prediction_is_a_truth_isolation_failure(self):
        with RunnerFixture() as fixture:
            fixture.annotation.write_text(
                json.dumps({"changePrediction": True}), encoding="utf-8"
            )
            result = run_cell(
                fixture.spec(), annotation_resolver=lambda _case: fixture.annotation
            )
            self.assertEqual(result["status"], "truth-isolation-failure")
            self.assertNotEqual(
                result["inference"]["predictionFingerprint"],
                result["evaluation"]["predictionFingerprint"],
            )

    def test_resume_does_not_repeat_completed_inference(self):
        with RunnerFixture() as fixture:
            spec = fixture.spec()

            def interrupt_after_prediction(_case):
                raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_cell(spec, annotation_resolver=interrupt_after_prediction)
            result = run_cell(
                spec,
                annotation_resolver=lambda _case: fixture.annotation,
                resume=True,
            )
            counter = (fixture.root / "counter.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(counter, ["inference", "evaluation"])
            self.assertEqual(result["status"], "complete")

    def test_process_error_and_missing_output_are_explicit(self):
        for mode, status in (("crash", "process-error"), ("missing", "missing-output"), ("malformed", "malformed-audit")):
            with self.subTest(mode=mode), RunnerFixture() as fixture:
                fixture.shared["candidates"].write_text(mode, encoding="utf-8")
                result = run_cell(
                    fixture.spec(), annotation_resolver=lambda _case: fixture.annotation
                )
                self.assertEqual(result["status"], status)

    def test_matrix_order_is_deterministic_and_keeps_every_cell(self):
        with RunnerFixture() as fixture:
            second = CaseManifest(
                case_id="u6424-j-900005",
                unicode="U+6424",
                source="J",
                glyph_id=900005,
                annotation_basename="truth.json",
            )
            results = run_matrix(
                [fixture.spec(second), fixture.spec()],
                annotation_resolver=lambda _case: fixture.annotation,
            )
            self.assertEqual(
                [item["caseId"] for item in results],
                ["u6418-g-17973", "u6424-j-900005"],
            )


if __name__ == "__main__":
    unittest.main()
