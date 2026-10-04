import copy
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
    def test_verified_u66da_n_truth_uses_inner_inner_outer_outer_order(self):
        annotations = [{"token": index} for index in range(1, 19)]
        reordered, permutation = MODULE.apply_verified_annotation_stroke_order(
            {"reviewKey": "U+66DA-N-57149"}, annotations
        )

        self.assertEqual(permutation[4:8], [6, 7, 5, 8])
        self.assertEqual([item["token"] for item in reordered[4:8]], [6, 7, 5, 8])

    def test_disconnected_grass_leaf_uses_verified_horizontal_first_order(self):
        strokes = [
            {"componentId": 486, "occurrence": 0, "feature": "竖", "token": "left-v"},
            {"componentId": 486, "occurrence": 0, "feature": "横", "token": "left-h"},
            {"componentId": 486, "occurrence": 0, "feature": "竖", "token": "right-v"},
            {"componentId": 486, "occurrence": 0, "feature": "横", "token": "right-h"},
            {"componentId": 9, "occurrence": 0, "feature": "点", "token": "next"},
        ]

        corrected = MODULE.apply_verified_leaf_stroke_orders(strokes)

        self.assertEqual(
            [stroke["token"] for stroke in corrected],
            ["left-h", "left-v", "right-v", "right-h", "next"],
        )
        self.assertEqual([stroke["strokeIndex"] for stroke in corrected], list(range(5)))

    def test_merge_intersecting_annotations_trims_first_stroke_at_shared_turn(self):
        annotations = [
            {
                "type": "polyline",
                "label": "934",
                "points": [[43.0, 47.0], [43.0, 72.0]],
                "fixedPoints": [[43.0, 47.0], [43.0, 72.0]],
            },
            {
                "type": "polyline",
                "label": "934",
                "points": [[43.0, 66.0], [60.0, 66.0]],
            },
        ]

        merged = MODULE.merge_intersecting_annotation_strokes(annotations, 0, 1)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["type"], "polyline")
        self.assertEqual(merged[0]["label"], "934")
        self.assertEqual(
            merged[0]["points"],
            [[43.0, 47.0], [43.0, 66.0], [60.0, 66.0]],
        )
        self.assertEqual(merged[0]["fixedPoints"], merged[0]["points"])

    def test_human_truth_warning_counts_all_region_tools(self):
        warning = MODULE.human_truth_warning(
            stroke_count=15,
            metrics={
                "humanRegionCount": 5,
                "humanLassoCount": 0,
                "humanPolygonCount": 5,
            },
            correction_count=0,
        )

        self.assertEqual(
            warning,
            "已载入人工真值：15 笔、5 个部件圈；按真值重分区。ID 自动纠正 0 项。",
        )

    def test_stroke_sequence_profile_treats_repeated_leaf_occurrences_as_distinct(self):
        strokes = [
            {"componentId": 117, "occurrence": 0, "feature": "点"},
            {"componentId": 117, "occurrence": 0, "feature": "撇"},
            {"componentId": 1, "occurrence": 0, "feature": "横"},
            {"componentId": 117, "occurrence": 1, "feature": "点"},
            {"componentId": 117, "occurrence": 1, "feature": "撇"},
        ]
        profile = MODULE.stroke_sequence_profile(strokes)
        self.assertTrue(profile["sequentiallyClosed"])
        self.assertEqual(profile["componentInstanceCount"], 3)
        self.assertEqual(profile["completionStrokes"], [2, 3, 5])

    def test_stroke_sequence_profile_rejects_reentering_same_instance(self):
        strokes = [
            {"componentId": 10, "occurrence": 0, "feature": "横"},
            {"componentId": 20, "occurrence": 0, "feature": "竖"},
            {"componentId": 10, "occurrence": 0, "feature": "点"},
        ]
        profile = MODULE.stroke_sequence_profile(strokes)
        self.assertFalse(profile["sequentiallyClosed"])
        self.assertEqual(
            profile["reenteredInstances"],
            [{"componentId": 10, "occurrence": 0}],
        )

    def test_verified_component_transfer_requires_exact_id_and_stroke_count(self):
        candidate = [
            {"componentId": 10, "points": np.array([[0.0, 0.0], [1.0, 0.0]])},
            {"componentId": 10, "points": np.array([[0.0, 1.0], [1.0, 1.0]])},
            {"componentId": 20, "points": np.array([[2.0, 0.0], [2.0, 1.0]])},
        ]
        annotations = {
            "annotations": [
                {"type": "polyline", "label": "10", "points": [[5, 5], [6, 5]]},
                {"type": "polyline", "label": "10", "points": [[5, 6], [6, 6]]},
                # Wrong count for component 20: it must not be transferred.
                {"type": "polyline", "label": "20", "points": [[7, 5], [7, 6]]},
                {"type": "polyline", "label": "20", "points": [[8, 5], [8, 6]]},
                {"type": "polygon", "label": "10", "points": [[0, 0], [1, 0], [1, 1]]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(annotations), "utf-8")
            transferred, details = MODULE.transfer_verified_component_strokes(
                candidate, [path]
            )
        np.testing.assert_array_equal(
            transferred[0]["points"], np.array([[0.0, 0.0], [1.0, 0.0]])
        )
        np.testing.assert_array_equal(transferred[2]["points"], candidate[2]["points"])
        self.assertEqual([item["componentId"] for item in details], [10])

    def test_same_character_transfer_preserves_absolute_component_layout(self):
        candidate = [
            {"componentId": 10, "points": np.array([[0.0, 0.0], [1.0, 0.0]])},
        ]
        annotations = {
            "metadata": {"unicode": "U+64CE", "source": "G"},
            "annotations": [
                {"type": "polyline", "label": "10", "points": [[25, 30], [40, 30]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(annotations), "utf-8")
            transferred, details = MODULE.transfer_verified_component_strokes(
                candidate, [path], target_unicode="U+64CE", target_source="G"
            )
        np.testing.assert_array_equal(
            transferred[0]["points"], np.array([[25.0, 30.0], [40.0, 30.0]])
        )
        self.assertEqual(
            details[0]["transferMode"], "same-character-and-source-absolute"
        )

    def test_same_character_different_source_uses_component_local_layout(self):
        candidate = [
            {"componentId": 10, "points": np.array([[10.0, 10.0], [20.0, 10.0]])},
        ]
        annotations = {
            "metadata": {"unicode": "U+64CE", "source": "T"},
            "annotations": [
                {"type": "polyline", "label": "10", "points": [[25, 30], [40, 30]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(annotations), "utf-8")
            transferred, details = MODULE.transfer_verified_component_strokes(
                candidate, [path], target_unicode="U+64CE", target_source="G"
            )
        np.testing.assert_array_equal(transferred[0]["points"], candidate[0]["points"])
        self.assertEqual(details[0]["transferMode"], "component-local")

    def test_reference_transfer_normalizes_exported_sibling_label_from_candidate_order(self):
        candidate = [
            {
                "componentId": 1128,
                "feature": "横",
                "points": np.array([[10.0, 10.0], [20.0, 10.0]]),
            },
            {
                "componentId": 1128,
                "feature": "竖弯钩",
                "points": np.array([[20.0, 10.0], [20.0, 20.0]]),
            },
        ]
        annotations = {
            "metadata": {
                "candidateGlyphId": 99,
                "candidateStrokeOrder": ["横", "竖弯钩"],
                "source": "T",
            },
            "annotations": [
                {"type": "line", "label": "133", "points": [[30, 30], [50, 30]]},
                {"type": "line", "label": "133", "points": [[50, 30], [50, 50]]},
            ],
        }
        reference_candidates = {99: copy.deepcopy(candidate)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(annotations), "utf-8")
            transferred, details = MODULE.transfer_verified_component_strokes(
                candidate,
                [path],
                reference_candidates=reference_candidates,
            )
            coverage = MODULE.verified_component_coverage(
                candidate,
                [path],
                reference_candidates=reference_candidates,
                target_source="T",
            )
            comparison = MODULE.compare_verified_component_coverage(
                {
                    1: candidate,
                    2: [
                        {**stroke, "componentId": 133}
                        for stroke in copy.deepcopy(candidate)
                    ],
                },
                [path],
                reference_candidates=reference_candidates,
                target_source="T",
            )

        self.assertEqual([item["componentId"] for item in details], [1128])
        self.assertEqual(
            details[0]["annotationLabelCorrections"],
            [
                {
                    "strokeNumber": 1,
                    "from": 133,
                    "to": 1128,
                    "reason": "reference candidate stroke ownership",
                },
                {
                    "strokeNumber": 2,
                    "from": 133,
                    "to": 1128,
                    "reason": "reference candidate stroke ownership",
                },
            ],
        )
        np.testing.assert_array_equal(transferred[0]["points"], candidate[0]["points"])
        self.assertEqual(coverage["verifiedExactIds"], [1128])
        self.assertEqual(coverage["verifiedSameSourceExactIds"], [1128])
        self.assertEqual(coverage["normalizedReferenceLabelCorrections"], 2)
        self.assertEqual(comparison["evidenceGlyphId"], 1)
        self.assertEqual(
            comparison["discriminatingSameSourceVerifiedIds"],
            {"1": [1128], "2": []},
        )

    def test_global_template_search_prefers_better_whole_glyph_reconstruction(self):
        candidate = [
            {"componentId": 10, "points": np.array([[50.0, 10.0], [50.0, 90.0]])},
        ]
        annotations = {
            "metadata": {"unicode": "U+TEST", "source": "G"},
            "annotations": [
                {"type": "polyline", "label": "10", "points": [[10, 50], [90, 50]]},
            ],
        }
        target = np.zeros((64, 64), dtype=bool)
        target[30:34, 8:56] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(annotations), "utf-8")
            selected, search = MODULE.search_component_templates_globally(
                candidate,
                [path],
                target,
                64,
                target_unicode="U+TEST",
                target_source="G",
                beam_width=2,
            )
        self.assertEqual(search["choices"][0]["kind"], "verified-annotation")
        self.assertEqual(search["componentOptionCounts"], {"10": 2})
        self.assertEqual(search["componentsWithoutVerifiedTemplates"], [])
        np.testing.assert_array_equal(
            selected[0]["points"], np.array([[10.0, 50.0], [90.0, 50.0]])
        )

    def test_global_template_search_reports_missing_verified_coverage(self):
        candidate = [
            {"componentId": 10, "points": np.array([[5.0, 5.0], [25.0, 5.0]])},
        ]
        target = np.zeros((32, 32), dtype=bool)
        target[4:7, 5:26] = True
        _selected, search = MODULE.search_component_templates_globally(
            candidate, [], target, 32, beam_width=2
        )
        self.assertEqual(search["componentOptionCounts"], {"10": 1})
        self.assertEqual(search["componentsWithoutVerifiedTemplates"], [10])
        self.assertIsNone(search["margin"])

    def test_annotation_editor_uses_pointerdown_and_smooth_auto_tangents(self):
        script = MODULE.annotation_editor_script(
            {
                "reviewKey": "test",
                "unicode": "U+6418",
                "source": "G",
                "candidateGlyphId": 17973,
                "defaultComponentId": 220,
                "defaultComponentColor": "#f59e0b",
                "defaultComponentSequence": [
                    {"id": 220, "color": "#f59e0b"},
                    {"id": 934, "color": "#db2777"},
                ],
            }
        )

        self.assertIn("placeLinearPoint(\n            drawingPoint(event)", script)
        self.assertNotIn("board.addEventListener('click', event =>", script)
        self.assertIn("incomingHandleLength = vectorLength(incoming) / 3", script)
        self.assertIn("outgoingHandleLength = vectorLength(outgoing) / 3", script)
        self.assertIn("polygon: '多边形圈：逐点单击；空格或右键自动闭合'", script)
        self.assertIn("['line', 'polyline', 'polygon'].includes(tool)", script)
        self.assertIn("reassignRegion(Number(shape.dataset.annotationIndex))", script)
        self.assertIn("event.shiftKey && ['line', 'polyline'].includes(tool)", script)
        self.assertIn("labelFollowupPreferenceKey", script)
        self.assertIn("let pendingEmptyConfirmTool = null", script)
        self.assertIn("const finishDraftOrSwitchTool = () =>", script)
        self.assertIn("completed?.kind === 'stroke'", script)
        self.assertIn("completed?.kind === 'region'", script)
        self.assertIn("board.focus({preventScroll: true})", script)
        self.assertIn("event.code === 'Space' && document.activeElement === board", script)
        self.assertIn("event.preventDefault();\n        board.focus({preventScroll: true})", script)
        self.assertNotIn("event.key === 'Enter'", script)
        self.assertIn("if (!annotations.length && metadata.defaultComponentId != null)", script)
        self.assertIn("labelInput.value = String(metadata.defaultComponentId)", script)
        self.assertIn("colorInput.value = metadata.defaultComponentColor", script)
        self.assertIn("let tool = 'polyline'", script)
        self.assertNotIn("let tool = 'freehand'", script)
        self.assertIn("const advanceDefaultComponent = () =>", script)
        self.assertIn("nextTool === labelFollowupTool && advanceDefaultComponent()", script)

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

    def test_legacy_annotation_order_follows_verified_candidate_permutation(self):
        candidate = [
            {
                "componentId": 486,
                "familyKey": "228/486",
                "color": "#c2410c",
                "hierarchy": [{"id": 486}],
                "occurrence": 0,
                "feature": feature,
            }
            for feature in ("横", "竖", "竖", "横")
        ]
        document = {
            "metadata": {
                "unicode": "U+64CE",
                "source": "T",
                "candidateGlyphId": 900000,
                "candidateStrokeOrder": ["竖", "横", "竖", "横"],
            },
            "annotations": [
                {"type": "line", "label": "486", "points": [[10, 10 + index], [20, 10 + index]]}
                for index in range(4)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(document), "utf-8")
            strokes, _normalized, corrections = MODULE.load_human_annotations(
                path,
                codepoint=0x64CE,
                source="T",
                glyph_id=900000,
                candidate=candidate,
                canvas=100,
            )

        self.assertEqual([stroke["points"][0][1] for stroke in strokes], [11, 10, 12, 13])
        self.assertEqual(corrections[-1]["type"], "verified-stroke-order")
        self.assertEqual(corrections[-1]["to"], [2, 1, 3, 4])

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

    def test_graph_snap_returns_only_connected_skeleton_edges(self):
        skeleton = np.zeros((32, 32), dtype=bool)
        skeleton[5, 5:25] = True
        skeleton[5:25, 24] = True
        template = np.array([[5.0, 5.0], [24.0, 5.0], [24.0, 24.0]])
        snapped, _maximum = MODULE.snap_polyline_on_skeleton_graph(
            template, skeleton
        )
        steps = np.abs(np.diff(snapped, axis=0))
        self.assertTrue(np.all(np.max(steps, axis=1) == 1))
        self.assertTrue(np.all(skeleton[snapped[:, 1].astype(int), snapped[:, 0].astype(int)]))
        np.testing.assert_array_equal(snapped[0], [5.0, 5.0])
        np.testing.assert_array_equal(snapped[-1], [24.0, 24.0])

    def test_sequential_snap_consumes_strokes_in_order(self):
        target = np.zeros((32, 32), dtype=bool)
        target[7:10, 4:28] = True
        target[21:24, 4:28] = True
        strokes = [
            {"componentId": 10, "occurrence": 0, "feature": "横"},
            {"componentId": 20, "occurrence": 0, "feature": "横"},
        ]
        lines = [
            np.array([[4.0, 8.0], [27.0, 8.0]]),
            np.array([[4.0, 22.0], [27.0, 22.0]]),
        ]
        selected, maxima, search = MODULE.snap_centerlines_sequentially(
            strokes, lines, target, beam_width=8
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(maxima), 2)
        self.assertEqual(
            [choice["strokeNumber"] for choice in search["choices"]], [1, 2]
        )
        self.assertTrue(search["strokeSequenceProfile"]["sequentiallyClosed"])
        self.assertLess(search["uncoveredSkeletonRatio"], 0.4)

    def test_hierarchical_partition_closes_component_before_strokes(self):
        target = np.zeros((32, 32), dtype=bool)
        target[5:9, 3:29] = True
        target[21:25, 3:29] = True
        strokes = [
            {"componentId": 10, "occurrence": 0},
            {"componentId": 20, "occurrence": 0},
        ]
        lines = [
            np.array([[3.0, 7.0], [28.0, 7.0]]),
            np.array([[3.0, 23.0], [28.0, 23.0]]),
        ]
        masks, _ambiguous, metrics = MODULE.partition_components_then_strokes(
            target, lines, strokes
        )
        self.assertTrue(np.array_equal(masks[0], target & (np.indices(target.shape)[0] < 16)))
        self.assertTrue(np.array_equal(masks[1], target & (np.indices(target.shape)[0] >= 16)))
        self.assertEqual(metrics["componentInstanceCount"], 2)

    def test_component_closure_score_prefers_matching_closed_components(self):
        target = np.zeros((64, 64), dtype=bool)
        target[14:18, 8:28] = True
        target[46:50, 36:56] = True
        matching = [
            {
                "componentId": 10,
                "occurrence": 0,
                "feature": "横",
                "points": np.array([[12.5, 25.0], [43.5, 25.0]]),
            },
            {
                "componentId": 20,
                "occurrence": 0,
                "feature": "横",
                "points": np.array([[56.0, 75.0], [87.5, 75.0]]),
            },
        ]
        mismatching = copy.deepcopy(matching)
        mismatching[1]["points"] = np.array([[56.0, 25.0], [87.5, 25.0]])
        good = MODULE.component_closure_alignment_metrics(matching, target, 64)
        bad = MODULE.component_closure_alignment_metrics(mismatching, target, 64)
        self.assertLess(good["score"], bad["score"])
        self.assertEqual([item["lastStroke"] for item in good["componentClosures"]], [1, 2])

    def test_shared_focus_window_is_candidate_independent(self):
        first = [
            {"componentId": 99, "points": np.array([[0.0, 0.0], [100.0, 100.0]])},
            {"componentId": 10, "points": np.array([[10.0, 20.0], [30.0, 20.0]])},
        ]
        second = [
            {"componentId": 99, "points": np.array([[0.0, 0.0], [100.0, 100.0]])},
            {"componentId": 20, "points": np.array([[60.0, 70.0], [90.0, 70.0]])},
        ]
        minimum, maximum = MODULE.shared_focus_window(
            [first, second], {10, 20}, 100
        )
        reversed_minimum, reversed_maximum = MODULE.shared_focus_window(
            [second, first], {10, 20}, 100
        )
        np.testing.assert_array_equal(minimum, reversed_minimum)
        np.testing.assert_array_equal(maximum, reversed_maximum)
        first_focus = MODULE.fit_centerlines(first, 100)[1]
        second_focus = MODULE.fit_centerlines(second, 100)[1]
        for point in np.concatenate([first_focus, second_focus]):
            self.assertTrue(np.all(point >= minimum))
            self.assertTrue(np.all(point <= maximum))

    def test_directed_polyline_distortion_preserves_pen_direction(self):
        horizontal = np.array([[0.0, 0.0], [20.0, 0.0]])
        same = np.array([[1.0, 1.0], [21.0, 1.0]])
        reversed_line = same[::-1]
        diagonal = np.array([[1.0, 1.0], [15.0, 15.0]])

        self.assertEqual(
            MODULE.directed_polyline_distortions(horizontal, same),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            MODULE.directed_polyline_distortions(horizontal, reversed_line),
            (180.0, 180.0, 180.0),
        )
        displacement, start, end = MODULE.directed_polyline_distortions(
            horizontal, diagonal
        )
        self.assertAlmostEqual(displacement, 45.0)
        self.assertAlmostEqual(start, 45.0)
        self.assertAlmostEqual(end, 45.0)

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

    def test_candidate_ranking_puts_selected_candidate_first_when_evidence_ties(self):
        row = {"candidateSvgs": {"133": "", "1128": ""}}
        ranked = MODULE.rank_candidates(
            row,
            result=None,
            evidence=None,
            selected_glyph_id=1128,
        )

        self.assertEqual([item["id"] for item in ranked], [1128, 133])


if __name__ == "__main__":
    unittest.main()
