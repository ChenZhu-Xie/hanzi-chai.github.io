import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unihan_experts.adapters import (
    RunContext,
    build_evaluation_command,
    build_inference_command,
    semantic_command_hash,
)
from unihan_experts.models import ArtifactRef, CaseManifest, ExpertManifest
from unihan_experts.registry import load_expert_registry


SHA_A = "a" * 64


def flag_value(argv, flag):
    position = argv.index(flag)
    return argv[position + 1]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name).resolve()
        self.worktree = root / "worktree"
        (self.worktree / "scripts").mkdir(parents=True)
        (self.worktree / "scripts" / "unihan-ink-diffusion.py").touch()
        self.paths = {
            "pdf": root / "U4E00.pdf",
            "bbox": root / "bbox.html",
            "candidates": root / "candidates.json",
        }
        for path in self.paths.values():
            path.touch()
        self.case = CaseManifest(
            case_id="u6418-g-17973",
            unicode="U+6418",
            source="G",
            glyph_id=17973,
            annotation_basename="U+6418-G-17973-annotations.json",
        )
        self.output = root / "review.html"
        self.audit = root / "audit.json"

    def tearDown(self):
        self.temp.cleanup()

    def context(self, expert, **overrides):
        values = {
            "python_executable": Path(sys.executable),
            "worktree": self.worktree,
            "expert": expert,
            "case": self.case,
            "shared_artifacts": self.paths,
            "resolved_rule_artifacts": {},
            "output_path": self.output,
            "audit_path": self.audit,
        }
        values.update(overrides)
        return RunContext(**values)

    def test_every_registered_expert_builds_a_truth_free_absolute_command(self):
        experts = load_expert_registry(SCRIPT_DIR / "unihan-experts.json")
        for expert in experts:
            with self.subTest(expert=expert.expert_id):
                argv = build_inference_command(self.context(expert))
                self.assertNotIn("--annotations", argv)
                for flag in ("--pdf", "--bbox-cache", "--candidates", "--output", "--audit-output"):
                    self.assertTrue(Path(flag_value(argv, flag)).is_absolute())
                self.assertEqual(flag_value(argv, "--decoder"), expert.decoder)
                self.assertEqual(flag_value(argv, "--beam-width"), "350")

    def test_adapter_generations_do_not_receive_newer_flags(self):
        experts = {
            item.expert_id: item
            for item in load_expert_registry(SCRIPT_DIR / "unihan-experts.json")
        }
        classic = build_inference_command(self.context(experts["classic-15c"]))
        centroid = build_inference_command(self.context(experts["grass-005"]))
        reserved = build_inference_command(self.context(experts["reserved-dad"]))
        self.assertNotIn("--leaf-centroid-weight", classic)
        self.assertNotIn("--disable-completed-stroke-reservation", classic)
        self.assertIn("--leaf-centroid-weight", centroid)
        self.assertNotIn("--disable-completed-stroke-reservation", centroid)
        self.assertIn("--leaf-centroid-weight", reserved)
        self.assertNotIn("--disable-completed-stroke-reservation", reserved)

    def test_reservation_adapter_can_disable_reservation_explicitly(self):
        base = next(
            item
            for item in load_expert_registry(SCRIPT_DIR / "unihan-experts.json")
            if item.expert_id == "reserved-dad"
        )
        expert = ExpertManifest(
            **{
                **base.__dict__,
                "expert_id": "reserved-disabled",
                "config": {**base.config, "completedStrokeReservation": False},
            }
        )
        argv = build_inference_command(self.context(expert))
        self.assertIn("--disable-completed-stroke-reservation", argv)

    def test_evaluation_only_adds_annotation_and_phase_outputs(self):
        expert = load_expert_registry(SCRIPT_DIR / "unihan-experts.json")[0]
        inference_context = self.context(expert)
        annotation = self.output.parent / "truth.json"
        annotation.touch()
        evaluation_context = self.context(
            expert,
            output_path=self.output.parent / "evaluated.html",
            audit_path=self.output.parent / "evaluated.json",
        )
        inference = build_inference_command(inference_context)
        evaluation = build_evaluation_command(evaluation_context, annotation)
        self.assertNotIn("--annotations", inference)
        self.assertEqual(flag_value(evaluation, "--annotations"), str(annotation.resolve()))
        self.assertEqual(
            semantic_command_hash(inference), semantic_command_hash(evaluation)
        )

    def test_rule_artifact_is_exactly_the_resolved_manifest_artifact(self):
        expert = ExpertManifest(
            expert_id="rules-expert",
            schema_version=1,
            commit="15c2b04",
            adapter="classic-v1",
            decoder="legacy",
            config={"beamWidth": 350},
            rule_artifacts=(
                ArtifactRef(
                    alias="learnedRules", path=".local/rules.json", sha256=SHA_A
                ),
            ),
            parents=(),
            declared_capabilities=(),
            resource_class="small",
        )
        rules = self.output.parent / "rules.json"
        rules.touch()
        argv = build_inference_command(
            self.context(expert, resolved_rule_artifacts={"learnedRules": rules})
        )
        self.assertEqual(flag_value(argv, "--learned-rules"), str(rules.resolve()))

    def test_unknown_config_key_fails_before_process_spawn(self):
        expert = ExpertManifest(
            expert_id="bad-config",
            schema_version=1,
            commit="15c2b04",
            adapter="classic-v1",
            decoder="legacy",
            config={"futureFlag": True},
            rule_artifacts=(),
            parents=(),
            declared_capabilities=(),
            resource_class="small",
        )
        with self.assertRaisesRegex(ValueError, "unsupported config"):
            build_inference_command(self.context(expert))


if __name__ == "__main__":
    unittest.main()
