import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).with_name("unihan-ink-diffusion.py")
SPEC = importlib.util.spec_from_file_location("unihan_ink_diffusion_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DirectedInkDiffusionTests(unittest.TestCase):
    def test_cli_exposes_legacy_and_residual_decoder_choices(self):
        self.assertTrue(hasattr(MODULE, "build_argument_parser"))
        parser = MODULE.build_argument_parser()
        decoder = next(action for action in parser._actions if action.dest == "decoder")
        self.assertEqual(tuple(decoder.choices), ("legacy", "residual"))
        self.assertEqual(decoder.default, "legacy")

    def test_residual_decision_payload_exposes_new_safety_metrics(self):
        self.assertTrue(hasattr(MODULE, "residual_decision_payload"))
        unexplained = np.zeros((5, 5), dtype=bool)
        unexplained[0, 0] = True
        leak = np.zeros((5, 5), dtype=bool)
        result = SimpleNamespace(
            status="needs-review",
            review_reasons=("source-order-unavailable",),
            best_score=1.25,
            runner_up_margin=0.5,
            steps=({"stroke": 1},),
            ledger=SimpleNamespace(unexplained=unexplained),
            regions=(SimpleNamespace(forbidden_leak_mask=leak),),
        )
        payload = MODULE.residual_decision_payload(result, np.ones((5, 5), dtype=bool))
        self.assertEqual(payload["decoder"], "residual")
        self.assertEqual(payload["forbiddenBranchLeakagePixels"], 0)
        self.assertEqual(payload["residualUnexplainedInkPixels"], 1)
        self.assertAlmostEqual(payload["residualUnexplainedInkRatio"], 0.04)
        self.assertEqual(payload["reviewReasons"], ["source-order-unavailable"])

    def test_ink_width_follows_skeleton_ownership_across_a_junction(self):
        skeleton = np.zeros((15, 15), dtype=bool)
        skeleton[7, 1:14] = True
        skeleton[2:13, 7] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        target = MODULE.cv2.dilate(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

        def route(points):
            array = np.asarray(points, dtype=float)
            pixels = frozenset((int(x), int(y)) for x, y in points)
            return MODULE.RouteCandidate(0, 0, array, pixels, 1, 0.0, {})

        horizontal = route([(x, 7) for x in range(1, 14)])
        vertical = route([(7, y) for y in range(2, 13)])
        owner, _distance = MODULE.geodesic_owners(
            target, graph, [horizontal, vertical]
        )

        self.assertEqual(owner[7, 3], 0)
        self.assertEqual(owner[3, 7], 1)
        self.assertEqual(owner[11, 7], 1)

    def test_stroke_rule_signature_binds_exact_leaf_and_ordinal(self):
        strokes = [
            {"componentId": 220, "occurrence": 0, "feature": "横"},
            {"componentId": 220, "occurrence": 0, "feature": "竖钩"},
            {"componentId": 220, "occurrence": 0, "feature": "提"},
            {"componentId": 9, "occurrence": 0, "feature": "点"},
        ]
        signatures = MODULE.stroke_rule_signatures(strokes)
        self.assertEqual(signatures[1]["key"], "220|2|3|竖钩")
        self.assertEqual(signatures[3]["key"], "9|1|1|点")

    def test_held_out_learned_rule_requires_two_other_cases_to_score(self):
        signature = {
            "key": "220|1|3|横",
            "componentId": 220,
            "occurrence": 0,
            "ordinal": 1,
            "count": 3,
            "feature": "横",
        }
        model = {
            "examplesBySignature": {
                signature["key"]: [
                    {
                        "case": case,
                        "start": {
                            "topologyRole": "endpoint",
                            "directionSector": "E",
                            "componentNormalized": [0, 0],
                            "glyphNormalized": [0, 0],
                        },
                        "junctionTurnSequence": [],
                        "junctions": [],
                    }
                    for case in ("U+1111-G", "U+2222-G", "U+3333-G")
                ]
            }
        }
        rule = MODULE.learned_rule_for_stroke(model, signature, "U+1111-G")
        self.assertEqual(rule["support"], 2)
        self.assertEqual(rule["cases"], ["U+2222-G", "U+3333-G"])

        skeleton = np.zeros((9, 9), dtype=bool)
        skeleton[4, 1:8] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        endpoint_route = {"points": np.asarray([[1.0, 4.0], [7.0, 4.0]])}
        path_route = {"points": np.asarray([[4.0, 4.0], [7.0, 4.0]])}
        endpoint_cost, endpoint_evidence = MODULE.route_learned_cost(
            endpoint_route, graph, rule, (np.asarray([1.0, 4.0]), np.asarray([7.0, 4.0]))
        )
        path_cost, path_evidence = MODULE.route_learned_cost(
            path_route, graph, rule, (np.asarray([1.0, 4.0]), np.asarray([7.0, 4.0]))
        )
        self.assertEqual(endpoint_cost, 0.0)
        self.assertGreater(path_cost, 0.0)
        self.assertTrue(path_evidence["startRoleMismatch"])
        self.assertFalse(path_evidence["startRoleScoringEnabled"])
        self.assertTrue(endpoint_evidence["learnedScoringEnabled"])

        duplicated = {
            "examplesBySignature": {
                signature["key"]: [
                    {
                        **model["examplesBySignature"][signature["key"]][0],
                        "case": "U+2222-G",
                    }
                    for _index in range(2)
                ]
            }
        }
        duplicate_rule = MODULE.learned_rule_for_stroke(duplicated, signature, None)
        self.assertEqual(duplicate_rule["support"], 1)
        duplicate_cost, duplicate_evidence = MODULE.route_learned_cost(
            path_route,
            graph,
            duplicate_rule,
            (np.asarray([1.0, 4.0]), np.asarray([7.0, 4.0])),
        )
        self.assertEqual(duplicate_cost, 0.0)
        self.assertFalse(duplicate_evidence["learnedScoringEnabled"])

    def test_schema_two_learned_rules_never_mix_source_conventions(self):
        signature = {
            "key": "220|1|3|横",
            "contextKey": "220:0",
        }

        def training_case(case, sector):
            return {
                "case": case,
                "contextKey": "220:0",
                "start": {
                    "topologyRole": "endpoint",
                    "directionSector": sector,
                    "componentNormalized": [0, 0],
                    "glyphNormalized": [0, 0],
                },
                "junctionTurnSequence": [],
                "junctions": [],
            }

        model = {
            "schemaVersion": 2,
            "examplesBySignature": {
                "G|220|1|3|横": [training_case("U+1111-G", "E")],
                "T|220|1|3|横": [training_case("U+2222-T", "S")],
            },
        }
        g_rule = MODULE.learned_rule_for_stroke(model, signature, None, source="G")
        t_rule = MODULE.learned_rule_for_stroke(model, signature, None, source="T")
        self.assertEqual(g_rule["startSector"], "E")
        self.assertEqual(t_rule["startSector"], "S")
        self.assertIsNone(MODULE.learned_rule_for_stroke(model, signature, None))

    def test_turn_landmarks_distinguish_straight_from_right_angle(self):
        straight = np.asarray([[0.0, 0.0], [30.0, 0.0]])
        corner = np.asarray([[0.0, 0.0], [15.0, 0.0], [15.0, 15.0]])
        self.assertEqual(MODULE.turn_landmarks(straight), [])
        turns = MODULE.turn_landmarks(corner)
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(turns[0], 0.5, delta=0.12)

    def test_directed_sequence_dtw_ignores_local_drawing_speed(self):
        uniform = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        delayed_turn = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [9.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
        )
        aligned = MODULE.directed_sequence_dtw(uniform, delayed_turn)
        reversed_arc = MODULE.directed_sequence_dtw(uniform, delayed_turn[::-1])
        self.assertLess(aligned["mean"], reversed_arc["mean"])
        self.assertEqual(aligned["logicProgressMap"][0], 0.0)
        self.assertEqual(aligned["logicProgressMap"][-1], 1.0)

    def test_critical_node_trace_preserves_directed_junction_order(self):
        skeleton = np.zeros((15, 25), dtype=bool)
        skeleton[7, 1:24] = True
        skeleton[3:12, 7] = True
        skeleton[3:12, 17] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        forward = MODULE.critical_node_trace(
            np.asarray([[1.0, 7.0], [23.0, 7.0]]), graph, radius=3.0
        )
        backward = MODULE.critical_node_trace(
            np.asarray([[23.0, 7.0], [1.0, 7.0]]), graph, radius=3.0
        )
        self.assertGreaterEqual(len(forward), 2)
        self.assertEqual(
            [item["node"] for item in forward],
            list(reversed([item["node"] for item in backward])),
        )

    def test_total_turn_ignores_one_pixel_skeleton_stair_steps(self):
        noisy_vertical = np.asarray(
            [[5.0 + (index % 2), float(index)] for index in range(30)]
        )
        right_angle = np.asarray([[0.0, 0.0], [0.0, 20.0], [20.0, 20.0]])
        self.assertLess(MODULE.total_turn(noisy_vertical), 0.2)
        self.assertAlmostEqual(MODULE.total_turn(right_angle), np.pi / 2, delta=0.08)

    def test_more_leaf_components_can_share_their_nearest_pdf_ink_island(self):
        skeleton = np.zeros((12, 12), dtype=bool)
        skeleton[1:11, 1] = True
        skeleton[1:11, 10] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {"componentId": 10, "occurrence": 0, "points": np.asarray([[0.0, 1.0], [0.0, 4.0]])},
            {"componentId": 20, "occurrence": 0, "points": np.asarray([[9.0, 1.0], [9.0, 4.0]])},
            {"componentId": 30, "occurrence": 0, "points": np.asarray([[9.0, 6.0], [9.0, 9.0]])},
        ]
        mapping = MODULE.component_mapping(strokes, graph)
        self.assertNotEqual(mapping[(10, 0)], mapping[(20, 0)])
        self.assertEqual(mapping[(20, 0)], mapping[(30, 0)])

    def test_structural_columns_anchor_first_and_last_leaf_but_leave_middle_open(self):
        skeleton = np.zeros((12, 12), dtype=bool)
        skeleton[1:11, 1] = True
        skeleton[1:5, 10] = True
        skeleton[7:11, 10] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {"componentId": 10, "occurrence": 0, "points": np.asarray([[0.0, 1.0], [0.0, 10.0]])},
            {"componentId": 20, "occurrence": 0, "points": np.asarray([[10.0, 1.0], [10.0, 3.0]])},
            {"componentId": 30, "occurrence": 0, "points": np.asarray([[10.0, 4.0], [10.0, 6.0]])},
            {"componentId": 40, "occurrence": 0, "points": np.asarray([[10.0, 8.0], [10.0, 10.0]])},
        ]
        options = MODULE.component_label_options(strokes, graph)
        target_centres = {
            label: np.argwhere(graph.components == label)[:, ::-1].mean(axis=0)
            for label in set(graph.components[graph.components > 0].tolist())
        }
        left = min(target_centres, key=lambda label: target_centres[label][0])
        right = sorted(
            (label for label in target_centres if label != left),
            key=lambda label: target_centres[label][1],
        )
        self.assertEqual(options[(10, 0)], {left})
        self.assertEqual(options[(20, 0)], {right[0]})
        self.assertEqual(options[(30, 0)], set(right))
        self.assertEqual(options[(40, 0)], {right[1]})

    def test_forward_continuation_detects_a_stroke_stopped_mid_road(self):
        skeleton = np.zeros((11, 11), dtype=bool)
        skeleton[1:10, 5] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        truncated = np.asarray([[5.0, 1.0], [5.0, 5.0]])
        complete = np.asarray([[5.0, 1.0], [5.0, 9.0]])
        self.assertTrue(MODULE.has_forward_continuation(graph, truncated))
        self.assertFalse(MODULE.has_forward_continuation(graph, complete))

    def test_absolute_skeleton_coverage_can_outweigh_a_short_local_match(self):
        stroke = {"points": np.asarray([[0.0, 0.0], [10.0, 0.0]])}
        short = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 0.0], [1.0, 0.0]]), frozenset({(0, 0), (1, 0)}), 1, 0.0, {}
        )
        long = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [8.0, 0.0]]),
            frozenset((x, 0) for x in range(9)),
            1,
            0.5,
            {},
        )
        routes, decision = MODULE.choose_routes(
            [stroke], [[short, long]], total_skeleton_pixels=10
        )
        self.assertIs(routes[0], long)
        self.assertEqual(decision["unexplainedSkeletonPixels"], 1)
        self.assertEqual(decision["status"], "safe-candidate")

    def test_large_unexplained_skeleton_forces_review(self):
        stroke = {"points": np.asarray([[0.0, 0.0], [1.0, 0.0]])}
        route = MODULE.RouteCandidate(
            0, 1, stroke["points"], frozenset({(0, 0), (1, 0)}), 1, 0.0, {}
        )
        _routes, decision = MODULE.choose_routes(
            [stroke], [[route]], total_skeleton_pixels=10
        )
        self.assertEqual(decision["status"], "needs-review")
        self.assertIn("large-unexplained-skeleton", decision["reviewReasons"])

    def test_candidate_contact_is_weak_when_pdf_coverage_contradicts_it(self):
        strokes = [
            {"points": np.asarray([[0.0, 0.0], [4.0, 0.0]])},
            {"points": np.asarray([[4.0, 0.0], [5.0, 0.0]])},
        ]
        first = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [4.0, 0.0]]),
            frozenset((x, 0) for x in range(5)),
            1,
            0.0,
            {},
        )
        touching_but_short = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[4.0, 0.0], [5.0, 0.0]]),
            frozenset({(4, 0), (5, 0)}),
            1,
            0.0,
            {},
        )
        separate_but_complete = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 1.0], [8.0, 1.0]]),
            frozenset((x, 1) for x in range(9)),
            1,
            0.5,
            {},
        )
        routes, _decision = MODULE.choose_routes(
            strokes,
            [[first], [touching_but_short, separate_but_complete]],
            total_skeleton_pixels=14,
        )
        self.assertIs(routes[1], separate_but_complete)

    def test_direct_sibling_order_rejects_a_lower_leaf_above_its_predecessor(self):
        upper = {
            "componentId": 10,
            "occurrence": 0,
            "hierarchy": [{"id": 10}, {"id": 100, "label": "⿱"}],
            "points": np.asarray([[0.0, 2.0], [1.0, 2.0]]),
        }
        lower = {
            "componentId": 20,
            "occurrence": 0,
            "hierarchy": [{"id": 20}, {"id": 100, "label": "⿱"}],
            "points": np.asarray([[0.0, 8.0], [1.0, 8.0]]),
        }
        fixed_upper = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 5.0], [1.0, 5.0]]), frozenset({(0, 5), (1, 5)}), 1, 0.0, {}
        )
        wrong_above = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 1.0], [1.0, 1.0]]), frozenset({(0, 1), (1, 1)}), 1, 0.0, {}
        )
        correct_below = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 9.0], [1.0, 9.0]]), frozenset({(0, 9), (1, 9)}), 1, 0.1, {}
        )
        routes, _decision = MODULE.choose_routes(
            [upper, lower], [[fixed_upper], [wrong_above, correct_below]], 6
        )
        self.assertIs(routes[1], correct_below)

    def test_html_keeps_native_pdf_fill_below_ink_and_outline_above_it(self):
        points = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        route = MODULE.RouteCandidate(0, 1, points, frozenset({(0, 0), (1, 1)}), 1, 0.0, {})
        page = MODULE.build_html(
            record={"unicode": 0x65E8, "source": "T"},
            glyph={
                "definitions": '<path id="pdf-glyph" style="stroke:none" d="M0 0L1 1"/>',
                "useAttributes": {"href": "#pdf-glyph", "x": 174.42, "y": 621.06},
            },
            glyph_id=127685,
            strokes=[
                {
                    "points": points,
                    "feature": "horizontal",
                    "componentId": 1128,
                    "occurrence": 0,
                    "color": "#f59e0b",
                }
            ],
            target=np.ones((2, 2), dtype=bool),
            skeleton=np.eye(2, dtype=bool),
            graph=None,
            ranked=[[]],
            routes=[route],
            owner=np.zeros((2, 2), dtype=np.int16),
            arrival=np.zeros((2, 2), dtype=float),
            events=[
                {
                    "stroke": 1,
                    "start": {"point": [0, 0], "time": 0},
                    "end": {"point": [1, 1], "time": 1},
                    "startTime": 0,
                    "endTime": 1,
                    "stopReason": "test",
                }
            ],
            maximum_time=1.0,
            decision={},
            evaluation=None,
            truth_lines=[],
        )
        self.assertIn('id="pdf-native-fill"', page)
        self.assertIn('id="ink-diffusion-canvas"', page)
        self.assertIn('id="pdf-native-outline"', page)
        self.assertLess(page.index('id="pdf-native-fill"'), page.index('id="ink-diffusion-canvas"'))
        self.assertLess(page.index('id="ink-diffusion-canvas"'), page.index('id="pdf-native-outline"'))
        self.assertIn('id="show-pdf-fill"', page)
        self.assertIn('id="show-pdf-outline"', page)
        self.assertIn('id="outline-opacity"', page)
        self.assertIn('class="pdf-fill-use pdf-source-fit"', page)
        self.assertIn('class="pdf-outline-use pdf-source-fit"', page)
        self.assertIn("document.querySelectorAll('.pdf-source-fit').forEach(fitPdfSource)", page)
        self.assertIn("红点＝模型笔尖", page)
        self.assertIn("青点＝人工笔尖", page)
        self.assertIn("perStrokeTrajectoryAudit", page)
        self.assertIn("function arcLengthPrefix(points,fraction)", page)
        self.assertIn("if(clamped>=1)return points.slice()", page)
        self.assertNotIn("truthCount=Math.max", page)
        self.assertNotIn("ctx.setLineDash([5,4])", page)
        self.assertEqual(page.count('d="M0 0L1 1"'), 2)
        self.assertEqual(page.count('href="#pdf-glyph"'), 1)
        self.assertEqual(page.count('href="#pdf-glyph-outline"'), 1)
        outline = page[page.index('id="pdf-native-outline"') :]
        self.assertIn("fill:none;stroke:#0f172a", outline)

    def test_crossing_number_distinguishes_path_and_t_junction(self):
        skeleton = np.zeros((9, 9), dtype=bool)
        skeleton[1:8, 4] = True
        skeleton[4, 4:8] = True
        _count, crossing = MODULE.crossing_numbers(skeleton)
        self.assertEqual(int(crossing[1, 4]), 1)
        self.assertEqual(int(crossing[2, 4]), 2)
        self.assertGreaterEqual(int(crossing[4, 4]), 3)

    def test_contact_signature_is_scale_and_translation_independent(self):
        horizontal = np.asarray([[0.0, 5.0], [10.0, 5.0]])
        vertical = np.asarray([[4.0, 0.0], [4.0, 10.0]])
        original = MODULE.contact_signature(horizontal, vertical)[:2]
        transformed = MODULE.contact_signature(horizontal * 8 + 37, vertical * 8 + 37)[:2]
        self.assertTrue(np.allclose(original, transformed, atol=0.03))
        self.assertTrue(np.allclose(original, (0.4, 0.5), atol=0.04))

    def test_directed_contact_roles_change_when_a_stroke_is_reversed(self):
        first = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        second = np.asarray([[10.0, 0.0], [10.0, 8.0]])
        forward = MODULE.contact_signature(first, second)[:2]
        reversed_first = MODULE.contact_signature(first[::-1], second)[:2]
        self.assertTrue(np.allclose(forward, (1.0, 0.0), atol=0.03))
        self.assertTrue(np.allclose(reversed_first, (0.0, 0.0), atol=0.03))

    def test_alternate_route_can_be_found_when_shortest_fork_is_closed(self):
        # A square has two topologically distinct paths from top-left to
        # bottom-right.  Closing the first edge must expose the other one.
        adjacency = [
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (2, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (2, 1.0)],
        ]
        first, _ = MODULE.shortest_indices(adjacency, 0, 2)
        alternate, _ = MODULE.shortest_indices(adjacency, 0, 2, (first[0], first[1]))
        self.assertNotEqual(first, alternate)
        self.assertEqual({tuple(first), tuple(alternate)}, {(0, 1, 2), (0, 3, 2)})


if __name__ == "__main__":
    unittest.main()
