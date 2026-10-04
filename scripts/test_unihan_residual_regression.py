import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("unihan-residual-regression.py")
SPEC = importlib.util.spec_from_file_location("unihan_residual_regression_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NamedResidualRegressionTest(unittest.TestCase):
    def test_named_regression_requires_grass_hand_and_old_head_cases(self):
        report = MODULE.build_named_report([])
        missing = {item["key"] for item in report["cases"] if item["status"] == "missing"}
        self.assertEqual(
            missing,
            {"U+66DA-J:228", "U+6726-J:228", "U+6418-G:220", "U+6418-G:439", "U+64CE-T:486"},
        )
        self.assertEqual(report["status"], "failed")

    def test_report_fails_when_horizontal_owns_vertical_half_edge(self):
        observations = [
            {
                "unicode": "U+66DA",
                "source": "J",
                "componentIds": [228],
                "strokes": [
                    {
                        "componentId": 228,
                        "feature": "横",
                        "selectedHalfEdges": [
                            {
                                "edgeId": 9,
                                "startSector": "S",
                                "endSector": "S",
                                "outsideContact": True,
                            }
                        ],
                    }
                ],
            }
        ]
        report = MODULE.build_named_report(observations)
        row = next(item for item in report["cases"] if item["key"] == "U+66DA-J:228")
        self.assertEqual(row["status"], "failed")
        self.assertIn("horizontal-owns-vertical-half-edge", row["failures"])

    def test_later_hard_stop_does_not_retroactively_fail_completed_component(self):
        observations = [
            {
                "unicode": "U+64CE",
                "source": "T",
                "status": "needs-review",
                "hardViolations": ["no-feasible-route-stroke-16"],
                "strokes": [
                    {
                        "componentId": 486,
                        "feature": feature,
                        "selectedHalfEdges": [{"edgeId": index}],
                    }
                    for index, feature in enumerate(("横", "竖", "竖", "横"))
                ],
            }
        ]
        report = MODULE.build_named_report(observations)
        row = next(item for item in report["cases"] if item["key"] == "U+64CE-T:486")
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["decoderStatus"], "needs-review")


if __name__ == "__main__":
    unittest.main()
