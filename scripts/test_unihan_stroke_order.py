import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INK = load_module(ROOT / "unihan-ink-diffusion.py", "unihan_ink_for_order_test")
DIRECTED = load_module(ROOT / "unihan-directed-skeleton.py", "unihan_directed_for_order_test")
RESIDUAL = load_module(ROOT / "unihan-residual-ink.py", "unihan_residual_for_order_test")
TRANSFER = load_module(ROOT / "unihan-stroke-transfer.py", "unihan_transfer_for_order_test")
ORDER = load_module(ROOT / "unihan-stroke-order.py", "unihan_stroke_order_tested")


class StrokeOrderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            ORDER,
            "unihan-stroke-order.py must provide source-aware stroke expectations",
        )

    @staticmethod
    def old_head_fixture():
        skeleton = np.zeros((13, 13), dtype=bool)
        skeleton[3, 3:10] = True
        skeleton[8, 1:12] = True
        skeleton[3:9, 6] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        ledger = RESIDUAL.initial_ledger(skeleton)
        strokes = [
            {
                "componentId": 439,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0]]),
            },
            {
                "componentId": 439,
                "occurrence": 0,
                "feature": "竖",
                "points": np.asarray([[4.0, 0.0], [4.0, 5.0]]),
            },
            {
                "componentId": 439,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 439,
                "occurrence": 0,
                "feature": "撇",
                "points": np.asarray([[7.0, 0.0], [2.0, 8.0]]),
            },
        ]
        expectations = ORDER.compile_stroke_expectations(strokes, "G", 0x6418)
        return skeleton, graph, directed, ledger, expectations

    def test_old_head_selects_upper_short_horizontal_before_lower_long_horizontal(self):
        _skeleton, graph, directed, ledger, expectations = self.old_head_fixture()
        candidates = ORDER.rank_pen_down_candidates(
            expectations[0], ledger, directed, graph
        )
        legal = [candidate for candidate in candidates if not candidate.hard_rejections]
        point = graph.points[legal[0].point_index]
        self.assertEqual(tuple(map(int, point)), (3, 3))

    def test_raster_diagonal_neighbour_is_compatible_with_cardinal_direction(self):
        self.assertTrue(ORDER.sector_compatible("S", "SE"))
        self.assertTrue(ORDER.sector_compatible("S", "SW"))
        self.assertTrue(ORDER.sector_compatible("E", "SE"))
        self.assertFalse(ORDER.sector_compatible("S", "E"))
        self.assertFalse(ORDER.sector_compatible("S", "N"))

    def test_pen_down_sector_uses_initial_arc_not_the_distant_endpoint(self):
        points = np.asarray([[0.0, 0.0], [6.0, 0.0], [6.0, 20.0]])
        self.assertEqual(ORDER.direction_sector(points), "E")

    def test_stroke_feature_supplies_semantic_pen_down_direction(self):
        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "提",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0], [16.0, -8.0]]),
            }
        ]
        expectation = ORDER.compile_stroke_expectations(strokes, "G", 0x6418)[0]
        self.assertEqual(expectation.expected_sector, "NE")

    def test_after_subtracting_first_horizontal_next_horizontal_is_lower_one(self):
        skeleton, graph, directed, ledger, expectations = self.old_head_fixture()
        upper = np.zeros_like(skeleton)
        upper[3, 3:10] = True
        empty = np.zeros_like(skeleton)
        region = RESIDUAL.StrokeRegion(upper, empty, empty)
        ledger = RESIDUAL.advance_ledger(ledger, region, 0, ())
        candidates = ORDER.rank_pen_down_candidates(
            expectations[2], ledger, directed, graph
        )
        legal = [candidate for candidate in candidates if not candidate.hard_rejections]
        point = graph.points[legal[0].point_index]
        self.assertEqual(tuple(map(int, point)), (8, 1))

    def test_exact_verified_leaf_order_precedes_generic_top_to_bottom_rule(self):
        strokes = [
            {"componentId": 486, "occurrence": 0, "feature": feature, "points": points}
            for feature, points in (
                ("横", np.asarray([[0.0, 3.0], [4.0, 3.0]])),
                ("竖", np.asarray([[1.0, 0.0], [1.0, 4.0]])),
                ("竖", np.asarray([[3.0, 0.0], [3.0, 4.0]])),
                ("横", np.asarray([[0.0, 5.0], [4.0, 5.0]])),
            )
        ]
        expectations = ORDER.compile_stroke_expectations(strokes, "T", 0x64CE)
        self.assertEqual([item.feature for item in expectations], ["横", "竖", "竖", "横"])
        self.assertTrue(any(item.level == 2 and item.hard for item in expectations[0].evidence))

    def test_t_source_catalog_does_not_leak_into_g_source(self):
        payload = {
            "schemaVersion": 1,
            "entries": [
                {
                    "source": "T",
                    "unicode": "U+64CE",
                    "features": ["横", "竖"],
                    "provenance": "Taiwan MOE verified fixture",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            catalog = ORDER.load_normative_catalog(path)
        strokes = [
            {"componentId": 10, "occurrence": 0, "feature": "横", "points": np.asarray([[0, 0], [2, 0]])},
            {"componentId": 10, "occurrence": 0, "feature": "竖", "points": np.asarray([[1, 0], [1, 2]])},
        ]
        taiwan = ORDER.compile_stroke_expectations(strokes, "T", 0x64CE, catalog)
        mainland = ORDER.compile_stroke_expectations(strokes, "G", 0x64CE, catalog)
        self.assertTrue(any("Taiwan MOE" in item.provenance for item in taiwan[0].evidence))
        self.assertFalse(any("Taiwan MOE" in item.provenance for item in mainland[0].evidence))

    def test_unknown_source_rule_is_evidence_only_and_forces_review(self):
        strokes = [
            {"componentId": 10, "occurrence": 0, "feature": "横", "points": np.asarray([[0, 0], [2, 0]])}
        ]
        expectation = ORDER.compile_stroke_expectations(strokes, "X", 0x9999)[0]
        unavailable = [item for item in expectation.evidence if item.rule == "source-order-unavailable"]
        self.assertEqual(len(unavailable), 1)
        self.assertFalse(unavailable[0].hard)

    def test_disconnected_grass_head_keeps_verified_horizontal_first_order(self):
        strokes = [
            {"componentId": 486, "occurrence": 0, "feature": feature, "points": np.asarray(points, dtype=float)}
            for feature, points in (
                ("竖", [[0, 0], [0, 3]]),
                ("横", [[0, 1], [4, 1]]),
                ("竖", [[4, 0], [4, 3]]),
                ("横", [[0, 4], [4, 4]]),
            )
        ]
        corrected = TRANSFER.apply_verified_leaf_stroke_orders(strokes)
        expectations = ORDER.compile_stroke_expectations(corrected, "T", 0x64CE)
        self.assertEqual([item.feature for item in expectations], ["横", "竖", "竖", "横"])


if __name__ == "__main__":
    unittest.main()
