import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("unihan-ink-rule-training.py")
SPEC = importlib.util.spec_from_file_location("unihan_ink_rule_training_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def example(case: str, source: str, *, violations=()):
    return {
        "case": case,
        "source": source,
        "key": "220|1|3|横",
        "contextKey": "220:0",
        "constraintViolations": list(violations),
        "chosenHalfEdges": [4, 8],
        "forbiddenHalfEdges": [6, 10],
        "start": {"topologyRole": "endpoint"},
        "junctions": [],
    }


class InkRuleTrainingTest(unittest.TestCase):
    def test_rule_signature_separates_source_conventions(self):
        self.assertTrue(hasattr(MODULE, "build_rule_model"))
        model = MODULE.build_rule_model(
            [example("U+1111-G", "G"), example("U+2222-T", "T")]
        )
        self.assertEqual(set(model["examplesBySignature"]), {
            "G|220|1|3|横",
            "T|220|1|3|横",
        })
        self.assertEqual(model["schemaVersion"], 2)

    def test_training_records_chosen_and_forbidden_junction_half_edges(self):
        model = MODULE.build_rule_model([example("U+1111-G", "G")])
        stored = model["examples"][0]
        self.assertEqual(stored["chosenHalfEdges"], [4, 8])
        self.assertEqual(stored["forbiddenHalfEdges"], [6, 10])

    def test_constraint_failure_is_review_not_negative_training_example(self):
        model = MODULE.build_rule_model(
            [
                example("U+1111-G", "G"),
                example("U+2222-G", "G", violations=("forbidden-branch-leakage",)),
            ]
        )
        self.assertEqual([item["case"] for item in model["examples"]], ["U+1111-G"])
        self.assertEqual(model["rejectedConstraintExampleCount"], 1)
        self.assertEqual(model["reviewCases"], [
            {"case": "U+2222-G", "reasons": ["forbidden-branch-leakage"]}
        ])


if __name__ == "__main__":
    unittest.main()
