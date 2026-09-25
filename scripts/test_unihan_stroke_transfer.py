import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PATH = Path(__file__).with_name("unihan-stroke-transfer.py")
SPEC = importlib.util.spec_from_file_location("stroke_transfer", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StrokeTransferTest(unittest.TestCase):
    def test_annotation_editor_uses_pointerdown_and_smooth_auto_tangents(self):
        script = MODULE.annotation_editor_script(
            {
                "reviewKey": "test",
                "unicode": "U+6418",
                "source": "G",
                "candidateGlyphId": 17973,
            }
        )

        self.assertIn("placeLinearPoint(drawingPoint(event))", script)
        self.assertNotIn("board.addEventListener('click', event =>", script)
        self.assertIn("incomingHandleLength = vectorLength(incoming) / 3", script)
        self.assertIn("outgoingHandleLength = vectorLength(outgoing) / 3", script)
        self.assertIn("polygon: '多边形圈：逐点单击；Enter 或右键自动闭合'", script)
        self.assertIn("['line', 'polyline', 'polygon'].includes(tool)", script)

    def test_directed_stroke_features_change_when_arc_is_reversed(self):
        forward = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 30.0]])
        normal = MODULE.directed_stroke_features(forward, 100)
        reversed_arc = MODULE.directed_stroke_features(forward[::-1], 100)

        self.assertEqual(normal["start"], reversed_arc["end"])
        self.assertEqual(normal["end"], reversed_arc["start"])
        self.assertEqual(normal["startTangent"]["angleDegrees"], 0.0)
        self.assertEqual(reversed_arc["startTangent"]["angleDegrees"], -90.0)
        self.assertEqual(normal["signedTurnsDegrees"], [90.0])
        self.assertEqual(reversed_arc["signedTurnsDegrees"], [-90.0])

    def test_candidate_svg_rebuilds_all_topology_marker_types(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<path d="M 10 50 H 90"/><path d="M 50 10 V 50 H 80"/>'
            '<circle cx="10" cy="50" r="1.5" fill="red"/></svg>'
        )
        row = {
            "candidateSvgs": {"42": svg},
            "candidates": {"42": [{"feature": "横"}, {"feature": "折"}]},
            "candidateLeafSvgs": {
                "42": [
                    {
                        "leafId": 1,
                        "familyKey": "1",
                        "color": "#f59e0b",
                        "occurrence": 0,
                        "strokeIndices": [0],
                        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M 10 50 H 90"/></svg>',
                    },
                    {
                        "leafId": 2,
                        "familyKey": "2",
                        "color": "#7c3aed",
                        "occurrence": 0,
                        "strokeIndices": [1],
                        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M 50 10 V 50 H 80"/></svg>',
                    },
                ]
            },
        }
        output = MODULE.interactive_candidate_svg(row, 42, {}, instance="test")

        self.assertIn('class="node endpoint"', output)
        self.assertIn('class="node contact"', output)
        self.assertIn('class="node bend"', output)
        self.assertNotIn('fill="red"', output)

    def test_human_annotations_normalize_a_sibling_to_selected_leaf(self):
        candidate = [
            {
                "componentId": 1128,
                "familyKey": "133/1128",
                "color": "#db2777",
                "hierarchy": [{"id": 1128}],
                "occurrence": 0,
                "feature": "horizontal",
            },
            {
                "componentId": 1128,
                "familyKey": "133/1128",
                "color": "#db2777",
                "hierarchy": [{"id": 1128}],
                "occurrence": 1,
                "feature": "hook",
            },
        ]
        document = {
            "metadata": {
                "unicode": "U+6418",
                "source": "T",
                "candidateGlyphId": 900019,
            },
            "annotations": [
                {"type": "line", "label": "133", "color": "#000000", "points": [[10, 20], [30, 40]]},
                {"type": "freehand", "label": "133", "color": "#000000", "points": [[50, 60], [70, 80]]},
                {"type": "polygon", "label": "133", "color": "#000000", "points": [[5, 5], [90, 5], [90, 90]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(document), "utf-8")
            strokes, normalized, corrections = MODULE.load_human_annotations(
                path,
                codepoint=0x6418,
                source="T",
                glyph_id=900019,
                candidate=candidate,
                canvas=200,
            )

        self.assertEqual([item["label"] for item in normalized["annotations"]], ["1128"] * 3)
        self.assertEqual([item["color"] for item in normalized["annotations"]], ["#db2777"] * 3)
        self.assertEqual(len(corrections), 3)
        self.assertEqual(
            [item["display"] for item in corrections],
            ["第 1 笔", "第 2 笔", "部件圈"],
        )
        self.assertEqual([item["componentId"] for item in strokes], [1128, 1128])
        np.testing.assert_allclose(strokes[0]["points"], [[20, 40], [60, 80]])

    def test_human_truth_partition_is_complete_and_preserves_component_groups(self):
        target = np.zeros((32, 32), dtype=bool)
        target[3:13, 3:29] = True
        target[19:29, 3:29] = True
        centerlines = [
            np.array([[4, 8], [27, 8]], dtype=float),
            np.array([[4, 24], [27, 24]], dtype=float),
        ]
        strokes = [{"componentId": 10}, {"componentId": 20}]
        annotations = [
            {"type": "lasso", "label": "10", "points": [[0, 0], [100, 0], [100, 45], [0, 45]]},
            {"type": "polygon", "label": "20", "points": [[0, 55], [100, 55], [100, 100], [0, 100]]},
        ]
        masks, _ambiguous, metrics = MODULE.partition_human_truth(
            target, centerlines, strokes, annotations, 32
        )

        self.assertTrue(np.array_equal(np.logical_or.reduce(masks), target))
        self.assertFalse(np.logical_and(masks[0], masks[1]).any())
        self.assertTrue(masks[0][8, 8])
        self.assertTrue(masks[1][24, 8])
        self.assertEqual(metrics["humanComponentIds"], [10, 20])
        self.assertEqual(metrics["humanRegionCount"], 2)
        self.assertEqual(metrics["humanLassoCount"], 1)
        self.assertEqual(metrics["humanPolygonCount"], 1)

    def test_samples_relative_lines_and_cubic_without_control_points(self):
        points = MODULE.sample_svg_centerline("M 1 2 h 3 v 4 c 1 0 2 1 3 2", curve_steps=4)
        np.testing.assert_allclose(points[0], [1, 2])
        np.testing.assert_allclose(points[1], [4, 2])
        np.testing.assert_allclose(points[2], [4, 6])
        np.testing.assert_allclose(points[-1], [7, 8])
        self.assertEqual(len(points), 7)

    def test_samples_hook_arc_to_its_endpoint(self):
        points = MODULE.sample_svg_centerline("M 10 10 a 5 5 0 0 1 5 5")
        np.testing.assert_allclose(points[-1], [15, 15])
        self.assertGreater(len(points), 2)

    def test_structural_bends_only_follow_real_command_boundaries(self):
        points, fractions = MODULE.sample_svg_centerline(
            "M 0 0 h 10 v 10", with_fractions=True
        )
        self.assertEqual(len(points), 3)
        self.assertEqual(len(fractions), 1)
        self.assertAlmostEqual(fractions[0], 0.5)
        curve_points, curve_fractions = MODULE.sample_svg_centerline(
            "M 0 0 c 5 0 10 5 10 10", with_fractions=True
        )
        self.assertGreater(len(curve_points), 3)
        self.assertEqual(curve_fractions, [])

    def test_partition_is_complete_disjoint_and_preserves_stroke_order(self):
        target = np.zeros((24, 24), dtype=bool)
        target[3:21, 3:21] = True
        lines = [np.array([[6, 3], [6, 20]]), np.array([[17, 3], [17, 20]])]
        masks, ambiguous, metrics = MODULE.partition_strokes(target, lines)
        self.assertTrue(np.array_equal(np.logical_or.reduce(masks), target))
        self.assertFalse(np.logical_and(masks[0], masks[1]).any())
        self.assertTrue(ambiguous.any())
        self.assertEqual(metrics["seededStrokes"], 2)
        self.assertGreater(masks[0].sum(), 0)
        self.assertGreater(masks[1].sum(), 0)

    def test_coherent_snap_does_not_jump_backwards_at_a_crossing(self):
        skeleton = np.zeros((31, 31), dtype=bool)
        skeleton[15, 2:29] = True
        skeleton[2:29, 15] = True
        points = np.argwhere(skeleton)
        tree = cKDTree(points[:, ::-1])
        template = np.array([[3.0, 13.0], [27.0, 13.0]])
        snapped, _distance = MODULE.snap_polyline_coherently(
            template, points, tree, max_distance=5
        )
        self.assertGreater(len(snapped), 2)
        self.assertTrue(np.all(np.diff(snapped[:, 0]) >= 0))
        self.assertLessEqual(np.abs(snapped[:, 1] - 15).max(), 1)

    def test_geodesic_labels_do_not_cross_whitespace(self):
        target = np.zeros((16, 24), dtype=bool)
        target[3:6, 2:22] = True
        target[10:13, 2:22] = True
        seeds = np.zeros_like(target, dtype=np.int32)
        seeds[4, 3] = 1
        seeds[11, 20] = 2
        labels = MODULE.geodesic_labels(target, seeds)
        self.assertTrue(np.all(labels[3:6, 2:22] == 1))
        self.assertTrue(np.all(labels[10:13, 2:22] == 2))

    def test_alignment_score_prefers_matching_stroke_direction(self):
        target = np.zeros((64, 64), dtype=bool)
        target[30:34, 8:56] = True
        horizontal = [{"points": np.array([[0.0, 50.0], [100.0, 50.0]])}]
        vertical = [{"points": np.array([[50.0, 0.0], [50.0, 100.0]])}]
        matching = MODULE.candidate_alignment_metrics(horizontal, target, 64)
        different = MODULE.candidate_alignment_metrics(vertical, target, 64)
        self.assertLess(matching["score"], different["score"])

    def test_vectorizes_holes_with_evenodd_compatible_subpaths(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[3:37, 3:37] = True
        mask[12:28, 12:28] = False
        path = MODULE.mask_svg_path(mask, 40)
        self.assertGreaterEqual(path.count("M "), 2)
        self.assertGreaterEqual(path.count("Z"), 2)

    def test_candidate_ranking_exposes_method_prior_and_distance_disagreement(self):
        row = {"candidateSvgs": {"10": "", "20": ""}}
        result = {
            "predictionMethod": "topology-rule",
            "predictedGlyphId": 20,
            "candidates": [
                {"id": 10, "distance": 1.0},
                {"id": 20, "distance": 3.0},
            ],
            "correct": True,
        }
        evidence = {
            "results": [
                result,
                {"predictionMethod": "topology-rule", "correct": True},
                {"predictionMethod": "topology-rule", "correct": False},
            ]
        }
        ranked = MODULE.rank_candidates(row, result, evidence)
        self.assertEqual([item["id"] for item in ranked], [20, 10])
        self.assertAlmostEqual(ranked[0]["relativeWeight"], 0.6)
        self.assertLess(ranked[0]["distanceWeight"], ranked[1]["distanceWeight"])
        self.assertEqual(ranked[0]["methodPrior"]["correct"], 2)
        self.assertEqual(ranked[0]["methodPrior"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
