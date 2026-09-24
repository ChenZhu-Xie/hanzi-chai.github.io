import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("unihan-classify-topology-errors.py")
SPEC = importlib.util.spec_from_file_location("classify_topology_errors", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ClassifyTopologyErrorsTest(unittest.TestCase):
    def test_classifies_overlapping_structural_failure_modes(self):
        cascade = {
            "results": [
                {
                    "unicode": 0x4E00,
                    "source": "T",
                    "expectedGlyphId": 1,
                    "predictedGlyphId": 2,
                    "correct": False,
                    "abstained": False,
                }
            ]
        }
        evidence = {
            "results": [
                {
                    "unicode": 0x4E00,
                    "source": "T",
                    "recursiveTopology": {
                        "principalAxis": {"margin": 20},
                        "uniqueStrokeIndices": [[0], [0, 1]],
                        "differingSubtreeStrokeIndices": [[0], [0, 1]],
                        "pdfComponentCount": 8,
                        "candidateComponentCounts": [1, 2],
                        "candidateTopologySignatures": [
                            {"components": 1, "junctions": 0},
                            {"components": 2, "junctions": 1},
                        ],
                    },
                }
            ]
        }

        result = MODULE.classify(cascade, evidence)

        self.assertEqual(result["metadata"]["wrongAttempted"], 1)
        self.assertIn("terminal-stroke-direction", result["rows"][0]["categories"])
        self.assertIn("missing-or-extra-stroke", result["rows"][0]["categories"])
        self.assertIn(
            "local-window-neighbour-contamination", result["rows"][0]["categories"]
        )


if __name__ == "__main__":
    unittest.main()
