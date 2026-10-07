import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from unihan_experts.hashing import sha256_file
from unihan_experts.report import generate_reports


def result(expert, case, stroke, *, status="complete", complete=True):
    return {
        "expertId": expert,
        "caseId": case,
        "status": status,
        "complete": complete,
        "predictionFingerprintMatch": status != "truth-isolation-failure",
        "evaluation": {
            "metrics": {
                "strokeMacroIoU": stroke,
                "componentMacroIoU": stroke - 0.05,
                "meanDirectedSequenceDtwPercent": 2.0,
                "meanStartErrorPercent": 1.0,
                "meanEndErrorPercent": 1.5,
            },
            "rawArtifacts": {
                "htmlPath": f"raw/{expert}-{case}.html",
                "auditPath": f"raw/{expert}-{case}.json",
            },
            "process": {
                "wallSeconds": 2.0,
                "cpuSeconds": 1.5,
                "peakMemoryBytes": 1000,
            },
        },
        "inference": {
            "process": {
                "wallSeconds": 1.0,
                "cpuSeconds": 0.8,
                "peakMemoryBytes": 900,
            }
        },
    }


class ReportTests(unittest.TestCase):
    def test_reports_keep_failures_runners_up_capabilities_and_resources(self):
        cells = [
            result("alpha", "case-a", 0.9),
            result("beta", "case-a", 0.8),
            result("alpha", "case-b", 0.0, status="process-error", complete=False),
            result("beta", "case-b", 0.7),
        ]
        declared = {"alpha": ["grass-head"], "beta": ["closed-box"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = generate_reports(
                cells, root, run_id="run-1", declared_capabilities=declared
            )
            matrix = json.loads(outputs["matrix"].read_text(encoding="utf-8"))
            oracle = json.loads(outputs["oracle"].read_text(encoding="utf-8"))
            capabilities = json.loads(
                outputs["capabilities"].read_text(encoding="utf-8")
            )
            resources = json.loads(outputs["resources"].read_text(encoding="utf-8"))
            html = outputs["html"].read_text(encoding="utf-8")

        self.assertEqual(len(matrix["cells"]), 4)
        self.assertIn("process-error", {item["status"] for item in matrix["cells"]})
        self.assertEqual(len(oracle["cases"]["case-a"]["ranked"]), 2)
        self.assertTrue(oracle["usesHumanTruth"])
        self.assertEqual(capabilities["experts"]["alpha"]["declared"], ["grass-head"])
        self.assertIn("measured", capabilities["experts"]["alpha"])
        self.assertGreater(resources["experts"]["alpha"]["wallSeconds"], 0)
        self.assertIn("raw/alpha-case-a.html", html)
        self.assertIn("not-run", html)

    def test_report_bytes_are_deterministic(self):
        cells = [result("alpha", "case-a", 0.9)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = generate_reports(cells, root, run_id="same")
            hashes = {key: sha256_file(path) for key, path in first.items()}
            second = generate_reports(cells, root, run_id="same")
            self.assertEqual(hashes, {key: sha256_file(path) for key, path in second.items()})


if __name__ == "__main__":
    unittest.main()
