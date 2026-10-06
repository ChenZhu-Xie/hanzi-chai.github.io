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
    def test_separated_pen_up_endpoints_are_not_candidate_contacts(self):
        strokes = [
            {"points": np.asarray([[62.0, 45.0], [66.0, 53.0]])},
            {"points": np.asarray([[72.0, 45.0], [69.0, 53.0]])},
        ]
        self.assertFalse(MODULE.stroke_contact_matrix(strokes)[0, 1])

    def test_exact_pen_up_join_remains_a_candidate_contact(self):
        strokes = [
            {"points": np.asarray([[10.0, 2.0], [10.0, 8.0], [15.0, 8.0]])},
            {"points": np.asarray([[15.0, 2.0], [15.0, 8.0]])},
        ]
        self.assertTrue(MODULE.stroke_contact_matrix(strokes)[0, 1])

    def test_completed_leaf_claims_short_terminal_shared_pen_down_cap(self):
        skeleton = np.zeros((14, 14), dtype=bool)
        skeleton[5, 5:11] = True
        skeleton[5:11, 5] = True
        skeleton[2, 2] = True
        skeleton[3, 3] = True
        skeleton[4, 4] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {"points": np.asarray([[5.0, 5.0], [10.0, 5.0]])},
            {"points": np.asarray([[5.0, 5.0], [5.0, 10.0]])},
        ]
        routes = [
            MODULE.RouteCandidate(
                0,
                1,
                strokes[0]["points"],
                frozenset((x, 5) for x in range(5, 11)),
                1,
                0.0,
                {},
            ),
            MODULE.RouteCandidate(
                0,
                1,
                strokes[1]["points"],
                frozenset((5, y) for y in range(5, 11)),
                1,
                0.0,
                {},
            ),
        ]
        closure = MODULE.completed_leaf_residual_closure(strokes, routes, graph)
        self.assertEqual(closure, frozenset({(2, 2), (3, 3), (4, 4)}))

    def test_completed_leaf_claims_bounded_terminal_residual_away_from_pen_down(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[3, 2:13] = True
        skeleton[3:13, 2] = True
        skeleton[3:9, 8] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {"points": np.asarray([[2.0, 3.0], [12.0, 3.0]])},
            {"points": np.asarray([[2.0, 3.0], [2.0, 12.0]])},
        ]
        routes = [
            MODULE.RouteCandidate(
                0, 1, strokes[0]["points"],
                frozenset((x, 3) for x in range(2, 13)), 1, 0.0, {},
            ),
            MODULE.RouteCandidate(
                0, 1, strokes[1]["points"],
                frozenset((2, y) for y in range(3, 13)), 1, 0.0, {},
            ),
        ]
        closure = MODULE.completed_leaf_residual_closure(strokes, routes, graph)
        self.assertTrue(frozenset((8, y) for y in range(4, 9)) <= closure)

    def test_finished_stroke_reserves_short_uncoloured_terminal_detail(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[5, 5:11] = True
        skeleton[2, 2] = True
        skeleton[3, 3] = True
        skeleton[4, 4] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        route = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[5.0, 5.0], [10.0, 5.0]]),
            frozenset((x, 5) for x in range(5, 11)),
            1,
            0.0,
            {},
        )
        reservation = MODULE.completed_stroke_residual_reservation(route, graph)
        self.assertEqual(reservation, frozenset({(2, 2), (3, 3), (4, 4)}))
        self.assertTrue(
            MODULE.pen_down_hits_reservation(
                np.asarray([[3.0, 3.0], [2.0, 2.0]]), reservation
            )
        )

    def test_finished_long_stroke_reserves_a_scaled_terminal_protrusion(self):
        skeleton = np.zeros((80, 32), dtype=bool)
        skeleton[10:71, 5] = True
        skeleton[50, 6:22] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        route = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[5.0, 10.0], [5.0, 70.0]]),
            frozenset((5, y) for y in range(10, 71)),
            1,
            0.0,
            {},
        )
        reservation = MODULE.completed_stroke_residual_reservation(route, graph)
        self.assertTrue(frozenset((x, 50) for x in range(6, 22)) <= reservation)

    def test_later_pen_down_prefers_not_to_steal_reserved_completed_stroke_detail(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[5, 5:11] = True
        skeleton[2, 2] = True
        skeleton[3, 3] = True
        skeleton[4, 4] = True
        skeleton[12, 12:15] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {
                "componentId": 10,
                "occurrence": 0,
                "points": np.asarray([[5.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 20,
                "occurrence": 0,
                "points": np.asarray([[12.0, 12.0], [14.0, 12.0]]),
            },
        ]
        first = MODULE.RouteCandidate(
            0, 1, strokes[0]["points"],
            frozenset((x, 5) for x in range(5, 11)), 1, 0.0, {},
        )
        steals_reserved = MODULE.RouteCandidate(
            2, 3, np.asarray([[3.0, 3.0], [2.0, 2.0]]),
            frozenset({(3, 3), (2, 2)}), 1, -10.0, {},
        )
        safe = MODULE.RouteCandidate(
            4, 5, strokes[1]["points"],
            frozenset({(12, 12), (13, 12), (14, 12)}), 2, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[first], [steals_reserved, safe]],
            total_skeleton_pixels=int(skeleton.sum()),
            graph=graph,
        )
        self.assertIs(routes[1], safe)
        self.assertGreater(
            decision["steps"][0]["reservedCompletedStrokePixels"], 0
        )

    def test_candidate_declared_pen_down_contact_may_enter_a_reservation(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[5, 5:11] = True
        skeleton[2, 2] = True
        skeleton[3, 3] = True
        skeleton[4, 4] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {
                "componentId": 10,
                "occurrence": 0,
                "points": np.asarray([[5.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 20,
                "occurrence": 0,
                # The candidate explicitly puts this pen-down at the first
                # stroke's ink boundary; it is not an accidental theft.
                "points": np.asarray([[4.0, 4.0], [2.0, 2.0]]),
            },
        ]
        first = MODULE.RouteCandidate(
            0, 1, strokes[0]["points"],
            frozenset((x, 5) for x in range(5, 11)), 1, 0.0, {},
        )
        declared_contact = MODULE.RouteCandidate(
            2, 3, np.asarray([[3.0, 3.0], [2.0, 2.0]]),
            frozenset({(3, 3), (2, 2)}), 1, -2.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[first], [declared_contact]],
            total_skeleton_pixels=int(skeleton.sum()),
            graph=graph,
        )
        self.assertIs(routes[1], declared_contact)
        self.assertTrue(
            decision["steps"][1]["reservedPenDownContactException"]
        )

    def test_future_same_leaf_pen_down_is_exempted_from_reservation(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[5, 5:11] = True
        skeleton[2, 2] = True
        skeleton[3, 3] = True
        skeleton[4, 4] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {
                "componentId": 30,
                "occurrence": 0,
                "points": np.asarray([[5.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 30,
                "occurrence": 0,
                "points": np.asarray([[3.0, 3.0], [2.0, 2.0]]),
            },
        ]
        first = MODULE.RouteCandidate(
            0, 1, strokes[0]["points"],
            frozenset((x, 5) for x in range(5, 11)), 1, 0.0, {},
        )
        future_leaf_stroke = MODULE.RouteCandidate(
            2, 3, strokes[1]["points"],
            frozenset({(3, 3), (2, 2)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[first], [future_leaf_stroke]],
            total_skeleton_pixels=int(skeleton.sum()),
            graph=graph,
        )
        self.assertEqual(len(routes), 2)
        np.testing.assert_allclose(routes[0].points, first.points)
        np.testing.assert_allclose(routes[1].points, future_leaf_stroke.points)
        self.assertGreater(
            decision["steps"][0]["singleStrokeReservationPixels"], 0
        )
        self.assertEqual(
            decision["steps"][0]["partialLeafReservationPixels"], 0
        )
        self.assertTrue(
            decision["steps"][0][
                "reservationSuppressedByFutureSameLeafStart"
            ]
        )
        self.assertGreater(
            decision["steps"][0]["futureSameLeafPenDownExemptionPixels"], 0
        )

    def test_future_same_leaf_keeps_unrelated_terminal_reservation(self):
        skeleton = np.zeros((20, 20), dtype=bool)
        skeleton[8, 5:12] = True
        skeleton[5, 2] = True
        skeleton[6, 3] = True
        skeleton[7, 4] = True
        skeleton[15, 15:19] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        strokes = [
            {
                "componentId": 30,
                "occurrence": 0,
                "points": np.asarray([[5.0, 8.0], [11.0, 8.0]]),
            },
            {
                "componentId": 30,
                "occurrence": 0,
                "points": np.asarray([[15.0, 15.0], [18.0, 15.0]]),
            },
        ]
        first = MODULE.RouteCandidate(
            0, 1, strokes[0]["points"],
            frozenset((x, 8) for x in range(5, 12)), 1, 0.0, {},
        )
        future_leaf_stroke = MODULE.RouteCandidate(
            2, 3, strokes[1]["points"],
            frozenset((x, 15) for x in range(15, 19)), 2, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[first], [future_leaf_stroke]],
            total_skeleton_pixels=int(skeleton.sum()),
            graph=graph,
        )
        self.assertEqual(len(routes), 2)
        np.testing.assert_allclose(routes[0].points, first.points)
        np.testing.assert_allclose(routes[1].points, future_leaf_stroke.points)
        self.assertGreater(
            decision["steps"][0]["reservedCompletedStrokePixels"], 0
        )
        self.assertEqual(
            decision["steps"][0]["futureSameLeafPenDownExemptionPixels"], 0
        )

    def test_scan_seed_can_reach_a_distant_terminal_in_the_same_ink_island(self):
        points = np.asarray([[0, index] for index in range(10)], dtype=int)
        rows = []
        columns = []
        weights = []
        for index in range(9):
            rows.extend((index, index + 1))
            columns.extend((index + 1, index))
            weights.extend((1.0, 1.0))
        crossing = np.full((1, 10), 2, dtype=np.uint8)
        crossing[0, 9] = 1
        graph = MODULE.SkeletonGraph(
            points=points,
            point_index={(0, index): index for index in range(10)},
            matrix=MODULE.csr_matrix(
                (weights, (rows, columns)), shape=(10, 10)
            ),
            critical=[(0, index) for index in range(10)],
            components=np.ones((1, 10), dtype=np.int32),
            crossing=crossing,
            scan_seeds=((0, 0),),
        )
        routes = MODULE.all_routes(graph)
        self.assertTrue(
            any(
                np.array_equal(route["points"][0], [0.0, 0.0])
                and np.array_equal(route["points"][-1], [9.0, 0.0])
                for route in routes
            )
        )

    def test_compound_stroke_may_use_a_scan_seed_without_leaf_whitelisting(self):
        key = (569, 0)
        self.assertTrue(
            MODULE.allows_diagonal_seed_route(
                {"feature": "横折钩", "bendFractions": [0.2, 0.8]},
                key,
                set(),
            )
        )
        self.assertFalse(
            MODULE.allows_diagonal_seed_route(
                {"feature": "横", "bendFractions": []},
                key,
                set(),
            )
        )
        self.assertTrue(
            MODULE.allows_diagonal_seed_route(
                {"feature": "捺", "bendFractions": []},
                key,
                set(),
                pen_down_touches_sibling=True,
            )
        )

    def test_compound_diagonal_tail_cannot_collapse_into_a_vertical_tail(self):
        stroke = {
            "feature": "横撇",
            "bendFractions": [0.45],
            "points": np.asarray([[0.0, 0.0], [6.0, 0.0], [3.0, 8.0]]),
        }
        vertical_tail = np.asarray(
            [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [6.0, 8.0]]
        )
        diagonal_tail = np.asarray(
            [[0.0, 0.0], [6.0, 0.0], [5.0, 3.0], [3.0, 8.0]]
        )
        self.assertFalse(
            MODULE.compound_terminal_direction_compatible(stroke, vertical_tail)
        )
        self.assertTrue(
            MODULE.compound_terminal_direction_compatible(stroke, diagonal_tail)
        )

    def test_human_leaf_pen_down_starts_when_it_exits_previous_broad_ink(self):
        previous = np.asarray([[0.0, 0.0], [20.0, 0.0]])
        current = np.asarray(
            [[10.0, 0.0], [10.0, 3.0], [10.0, 6.0], [9.0, 10.0], [4.0, 20.0]]
        )
        trimmed, hidden = MODULE.trim_broad_ink_contact_prefix(
            current, previous, ink_radius=8.0
        )
        self.assertGreater(hidden, 4.0)
        self.assertGreaterEqual(trimmed[0, 1], 8.0)
        self.assertLess(trimmed[-1, 0], trimmed[0, 0])

    def test_contact_start_prunes_a_dominated_suffix_route(self):
        long = MODULE.RouteCandidate(
            0,
            2,
            np.asarray([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]),
            frozenset((index, index) for index in range(11)),
            1,
            0.2,
            {},
        )
        suffix = MODULE.RouteCandidate(
            1,
            2,
            np.asarray([[5.0, 5.0], [10.0, 10.0]]),
            frozenset((index, index) for index in range(5, 11)),
            1,
            0.4,
            {},
        )
        self.assertEqual(
            MODULE.prune_dominated_contact_start_suffixes(
                [suffix, long], pen_down_touches_sibling=True
            ),
            [long],
        )
        self.assertEqual(
            MODULE.prune_dominated_contact_start_suffixes(
                [suffix, long], pen_down_touches_sibling=False
            ),
            [suffix, long],
        )

    def test_free_pen_up_prunes_a_dominated_incomplete_prefix(self):
        short = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [4.0, -2.0]]),
            frozenset((index, 0) for index in range(5)),
            1,
            0.2,
            {
                "candidateLengthFraction": 0.20,
                "routeLengthFraction": 0.11,
            },
        )
        complete = MODULE.RouteCandidate(
            0,
            2,
            np.asarray([[0.0, 0.0], [4.0, -2.0], [10.0, -5.0]]),
            frozenset((index, 0) for index in range(11)),
            1,
            0.55,
            {
                "candidateLengthFraction": 0.20,
                "routeLengthFraction": 0.24,
            },
        )
        self.assertEqual(
            MODULE.prune_dominated_incomplete_prefixes(
                [short, complete], pen_up_touches_sibling=False
            ),
            [complete],
        )
        self.assertEqual(
            MODULE.prune_dominated_incomplete_prefixes(
                [short, complete], pen_up_touches_sibling=True
            ),
            [short, complete],
        )

    def test_route_reuse_detects_reverse_centerline_retracing(self):
        vertical = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[10.0, 0.0], [10.0, 100.0]]),
            frozenset((10, index) for index in range(101)),
            1,
            0.0,
            {},
        )
        reverse_segment = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[10.5, 90.0], [10.5, 30.0]]),
            frozenset((11, index) for index in range(30, 91)),
            1,
            0.0,
            {},
        )
        crossing = MODULE.RouteCandidate(
            4,
            5,
            np.asarray([[0.0, 50.0], [20.0, 50.0]]),
            frozenset((index, 50) for index in range(21)),
            1,
            0.0,
            {},
        )
        self.assertGreater(MODULE.route_reuse_ratio(vertical, reverse_segment), 0.9)
        self.assertLess(MODULE.route_reuse_ratio(vertical, crossing), 0.25)
        self.assertGreater(
            MODULE.centerline_retrace_length(vertical.points, reverse_segment.points),
            40.0,
        )
        self.assertLess(
            MODULE.centerline_retrace_length(vertical.points, crossing.points),
            8.0,
        )

    def test_stable_centerline_retrace_is_not_a_fallback_universe(self):
        first_points = np.asarray([[0.0, 0.0], [20.0, 0.0]])
        strokes = [
            {"componentId": 1, "occurrence": 0, "points": first_points},
            {
                "componentId": 2,
                "occurrence": 0,
                "points": np.asarray([[0.0, 5.0], [20.0, 5.0]]),
            },
        ]
        first = MODULE.RouteCandidate(
            0, 1, first_points, frozenset((x, 0) for x in range(21)), 1, 0.0, {}
        )
        retrace = MODULE.RouteCandidate(
            0,
            1,
            first_points[::-1].copy(),
            frozenset((x, 1) for x in range(21)),
            1,
            -100.0,
            {},
        )
        clean = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[0.0, 5.0], [20.0, 5.0]]),
            frozenset((x, 5) for x in range(21)),
            1,
            0.0,
            {},
        )
        routes, _decision = MODULE.choose_routes(
            strokes, [[first], [retrace, clean]], total_skeleton_pixels=42
        )
        self.assertIs(routes[1], clean)

    def test_candidate_proven_first_stroke_must_reach_leaf_front(self):
        strokes = [
            {"points": np.asarray([[0.0, 0.0], [2.0, 2.0]])},
            {"points": np.asarray([[6.0, 1.0], [8.0, 3.0]])},
            {"points": np.asarray([[1.0, 6.0], [8.0, 6.0]])},
        ]

        def route(points):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(0, 1, points, frozenset(), 1, 0.0, {})

        correct = [
            route([[10.0, 10.0], [12.0, 13.0]]),
            route([[16.0, 11.0], [18.0, 14.0]]),
            route([[10.0, 17.0], [18.0, 17.0]]),
        ]
        wrong = [
            route([[15.0, 15.0], [18.0, 18.0]]),
            route([[16.0, 11.0], [18.0, 14.0]]),
            route([[10.0, 12.0], [18.0, 12.0]]),
        ]
        _cost, evidence = MODULE.leaf_first_stroke_front_cost(strokes, correct)
        self.assertTrue(evidence["candidateProvesFront"])
        self.assertFalse(evidence["hardViolation"])
        _cost, evidence = MODULE.leaf_first_stroke_front_cost(strokes, wrong)
        self.assertTrue(evidence["hardViolation"])

    def test_contact_route_skips_an_already_owned_shared_prefix(self):
        previous = np.asarray([[0.0, 0.0], [0.0, 20.0]])
        merged_then_branches = np.asarray(
            [[0.5, 0.0], [0.5, 5.0], [0.5, 10.0], [4.0, 14.0], [9.0, 18.0]]
        )
        trimmed, hidden_length = MODULE.trim_contact_retrace_prefix(
            merged_then_branches,
            previous,
        )
        self.assertGreaterEqual(hidden_length, 8.0)
        self.assertGreater(trimmed[0, 0], 1.0)

        crossing = np.asarray([[-5.0, 10.0], [0.0, 10.0], [5.0, 10.0]])
        unchanged, hidden_length = MODULE.trim_contact_retrace_prefix(
            crossing,
            previous,
        )
        self.assertEqual(hidden_length, 0.0)
        np.testing.assert_array_equal(unchanged, crossing)

    def test_contact_retrace_variant_is_available_to_global_search(self):
        strokes = [
            {"points": np.asarray([[0.0, 0.0], [0.0, 20.0]])},
            {"points": np.asarray([[0.0, 0.0], [9.0, 18.0]])},
        ]
        previous = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [0.0, 20.0]]),
            frozenset((0, y) for y in range(21)),
            1,
            0.0,
            {},
        )
        merged = MODULE.RouteCandidate(
            0,
            2,
            np.asarray(
                [[0.5, 0.0], [0.5, 5.0], [0.5, 10.0], [4.0, 14.0], [9.0, 18.0]]
            ),
            frozenset(),
            1,
            0.0,
            {},
        )
        augmented = MODULE.augment_contact_retrace_variants(
            strokes, [[previous], [merged]]
        )
        variants = [
            option
            for option in augmented[1]
            if option.evidence.get("trimmedOwnedContactPrefix")
        ]
        self.assertTrue(variants)
        self.assertGreater(variants[0].points[0, 0], 1.0)

    def test_proven_leaf_front_prevents_later_strokes_from_forcing_bad_first_route(self):
        strokes = [
            {
                "componentId": 9,
                "occurrence": 0,
                "points": np.asarray([[0.0, 0.0], [2.0, 2.0]]),
            },
            {
                "componentId": 9,
                "occurrence": 0,
                "points": np.asarray([[5.0, 1.0], [7.0, 3.0]]),
            },
            {
                "componentId": 9,
                "occurrence": 0,
                "points": np.asarray([[1.0, 6.0], [7.0, 6.0]]),
            },
        ]

        def option(points, score, pixels):
            return MODULE.RouteCandidate(
                0, 1, np.asarray(points, dtype=float), frozenset(pixels), 1, score, {}
            )

        good_first = option([[0.0, 0.0], [2.0, 2.0]], 0.2, {(0, 0), (1, 1)})
        globally_tempting_bad_first = option(
            [[8.0, 8.0], [10.0, 10.0]],
            1.0,
            {(x, 8) for x in range(30)},
        )
        second = option([[5.0, 1.0], [7.0, 3.0]], 0.0, {(5, 1), (7, 3)})
        third = option([[1.0, 6.0], [7.0, 6.0]], 0.0, {(1, 6), (7, 6)})
        routes, _decision = MODULE.choose_routes(
            strokes,
            [[good_first, globally_tempting_bad_first], [second], [third]],
            total_skeleton_pixels=40,
        )
        self.assertIs(routes[0], good_first)

    def test_pen_down_contact_can_create_a_virtual_degree_two_start(self):
        skeleton = np.zeros((12, 12), dtype=bool)
        skeleton[1, 1:11] = True
        skeleton[1:11, 1] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        horizontal = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[1.0, 1.0], [10.0, 1.0]]),
            frozenset((x, 1) for x in range(1, 11)),
            1,
            0.0,
            {"strokeOrderExpectedSector": "E"},
        )
        vertical_suffix = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[1.0, 5.0], [1.0, 10.0]]),
            frozenset((1, y) for y in range(5, 11)),
            1,
            0.0,
            {
                "strokeOrderExpectedSector": "S",
                "candidateLengthFraction": 0.7,
                "routeLengthFraction": 0.35,
                "candidatePenDownTouchesSibling": True,
                "candidatePenUpTouchesSibling": False,
            },
        )
        strokes = [
            {
                "feature": "横",
                "points": np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            },
            {
                "feature": "竖",
                "points": np.asarray([[0.0, 0.0], [0.0, 10.0]]),
            },
        ]
        augmented = MODULE.augment_pen_down_contact_routes(
            strokes,
            [[horizontal], [vertical_suffix]],
            graph,
        )
        full = [
            option
            for option in augmented[1]
            if option.evidence.get("contactSeededVirtualStart")
        ]
        self.assertTrue(full)
        self.assertTrue(np.allclose(full[0].points[0], [1.0, 1.0]))
        self.assertTrue(np.allclose(full[0].points[-1], [1.0, 10.0]))

    def test_open_leaf_beam_keeps_each_first_stroke_universe(self):
        states = []
        for first_rank in range(4):
            for continuation in range(120):
                states.append(
                    {
                        "score": float(continuation) + first_rank * 0.01,
                        "routeRanks": [first_rank, continuation],
                    }
                )
        pruned = MODULE.prune_open_leaf_beam(
            states,
            width=350,
            first_stroke_index=0,
        )
        self.assertEqual(len(pruned), 350)
        self.assertEqual({state["routeRanks"][0] for state in pruned}, {0, 1, 2, 3})

    def test_leaf_centroid_order_treats_nearby_rows_as_left_to_right(self):
        strokes = [
            {"feature": "横", "points": np.asarray([[0.0, 0.0], [2.0, 0.0]])},
            {"feature": "竖", "points": np.asarray([[5.0, 0.0], [5.0, 8.0]])},
            {"feature": "横", "points": np.asarray([[7.0, -0.5], [9.0, -0.5]])},
        ]

        def route(points):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(
                0,
                1,
                points,
                frozenset((int(x), int(y)) for x, y in points),
                1,
                0.0,
                {},
            )

        correct = [
            route([[0.0, 4.0], [2.0, 4.0]]),
            route([[5.0, 1.0], [5.0, 9.0]]),
            route([[7.0, 3.2], [9.0, 3.2]]),
        ]
        cost, evidence = MODULE.leaf_centroid_order_cost(strokes, correct)
        self.assertEqual(cost, 0.0)
        self.assertEqual(evidence["preferredStroke"], 1)

        wrong = [
            route([[7.0, 4.0], [9.0, 4.0]]),
            correct[1],
            route([[0.0, 3.2], [2.0, 3.2]]),
        ]
        cost, evidence = MODULE.leaf_centroid_order_cost(strokes, wrong)
        self.assertGreater(cost, 0.0)
        self.assertEqual(evidence["preferredStroke"], 3)

    def test_diagonal_front_adds_a_temporary_seed_at_an_l_corner(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[3, 3:12] = True
        skeleton[3:13, 3] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        self.assertIn((3, 3), graph.critical)

    def test_broken_grass_head_uses_the_diagonal_front_model(self):
        strokes = [
            {
                "componentId": 486,
                "occurrence": 0,
                "feature": feature,
                "points": np.asarray(points, dtype=float),
            }
            for feature, points in (
                ("横", [[0, 2], [4, 2]]),
                ("竖", [[2, 0], [2, 6]]),
                ("竖", [[8, 0], [8, 6]]),
                ("横", [[6, 2], [10, 2]]),
            )
        ]
        self.assertIn((486, 0), MODULE.diagonal_front_leaf_keys(strokes))

        def route(points):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(
                0,
                1,
                points,
                frozenset((int(x), int(y)) for x, y in points),
                1,
                0.0,
                {},
            )

        correct = [route(stroke["points"]) for stroke in strokes]
        self.assertEqual(MODULE.grass_diagonal_recovery_needed(strokes, correct), [])
        wrong = [correct[3], correct[1], correct[2], correct[0]]
        self.assertEqual(
            MODULE.grass_diagonal_recovery_needed(strokes, wrong),
            [(486, 0)],
        )
        bent = list(correct)
        # The first horizontal still wins centroid order, but wrongly turns
        # down into the following vertical road.
        bent[0] = route([[0, -10], [10, -10], [10, 0]])
        self.assertEqual(
            MODULE.grass_diagonal_recovery_needed(strokes, bent),
            [(486, 0)],
        )

    def test_first_leaf_stroke_prefers_the_earliest_compatible_diagonal_front(self):
        strokes = [
            {
                "componentId": 282,
                "occurrence": 0,
                "feature": "竖",
                "points": np.asarray([[0.0, 0.0], [0.0, 10.0]]),
            },
            {
                "componentId": 282,
                "occurrence": 0,
                "feature": "横折",
                "points": np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]),
            },
            {
                "componentId": 282,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 10.0], [10.0, 10.0]]),
            },
        ]

        def vertical(x, score):
            points = np.asarray([[float(x), 5.0], [float(x), 15.0]])
            return MODULE.RouteCandidate(
                0,
                1,
                points,
                frozenset((int(x), y) for y in range(5, 16)),
                1,
                score,
                {},
            )

        left_front = vertical(10, 0.5)
        wrong_right = vertical(20, 0.0)
        enclosure_points = np.asarray([[10.0, 5.0], [30.0, 5.0], [30.0, 15.0]])
        enclosure = MODULE.RouteCandidate(
            2,
            3,
            enclosure_points,
            frozenset({(10, 5), (20, 5), (30, 5), (30, 10), (30, 15)}),
            1,
            0.0,
            {},
        )
        bottom_points = np.asarray([[10.0, 15.0], [30.0, 15.0]])
        bottom = MODULE.RouteCandidate(
            4,
            5,
            bottom_points,
            frozenset({(10, 15), (20, 15), (30, 15)}),
            1,
            0.0,
            {},
        )
        self.assertIn((282, 0), MODULE.diagonal_front_leaf_keys(strokes))
        routes, decision = MODULE.choose_routes(
            strokes,
            [[wrong_right, left_front], [enclosure], [bottom]],
            total_skeleton_pixels=30,
        )
        self.assertIs(routes[0], left_front)
        self.assertEqual(decision["steps"][0]["diagonalFrontCost"], 0.0)

    def test_html_lists_pen_down_acceptance_and_rejection_reasons(self):
        self.assertTrue(hasattr(MODULE, "residual_review_markup"))
        markup = MODULE.residual_review_markup()
        self.assertIn("落笔候选", markup)
        self.assertIn("接受理由", markup)
        self.assertIn("拒绝理由", markup)
        self.assertIn("后续笔画可行性", markup)

    def test_html_draws_selected_and_forbidden_half_edges_distinctly(self):
        markup = MODULE.residual_review_markup()
        self.assertIn("show-selected-half-edges", markup)
        self.assertIn("show-forbidden-half-edges", markup)
        self.assertIn("#22c55e", markup)
        self.assertIn("#ef4444", markup)

    def test_html_animates_residual_ink_after_each_stroke(self):
        markup = MODULE.residual_review_markup()
        self.assertIn("show-residual-before", markup)
        self.assertIn("show-residual-after", markup)
        self.assertIn("D.residualLayers", markup)

    def test_html_uses_later_stroke_colour_at_shared_contact(self):
        self.assertTrue(hasattr(MODULE, "residual_review_payload"))
        target = np.ones((3, 3), dtype=bool)
        first = np.zeros((3, 3), dtype=bool)
        first[1, 0:2] = True
        second = np.zeros((3, 3), dtype=bool)
        second[1, 1:3] = True
        contact = np.zeros((3, 3), dtype=bool)
        contact[1, 1] = True
        empty = np.zeros((3, 3), dtype=bool)
        result = SimpleNamespace(
            regions=(
                SimpleNamespace(mask=first, contact_mask=contact, forbidden_leak_mask=empty),
                SimpleNamespace(mask=second, contact_mask=contact, forbidden_leak_mask=empty),
            ),
            steps=(
                {"stroke": 1, "routeEdgeIds": [2], "forbiddenHalfEdges": [4]},
                {"stroke": 2, "routeEdgeIds": [6], "forbiddenHalfEdges": [8]},
            ),
        )
        payload = MODULE.residual_review_payload(result, target)
        final_owner = {tuple(pixel[:2]): pixel[2] for pixel in payload["visibleOwner"]}
        self.assertEqual(final_owner[(1, 1)], 1)
        self.assertEqual(len(payload["residualLayers"]), 2)
    def test_cli_defaults_to_classic_search_with_order_guard(self):
        self.assertTrue(hasattr(MODULE, "build_argument_parser"))
        parser = MODULE.build_argument_parser()
        decoder = next(action for action in parser._actions if action.dest == "decoder")
        self.assertEqual(tuple(decoder.choices), ("hybrid", "legacy", "residual"))
        self.assertEqual(decoder.default, "hybrid")

    def test_hybrid_order_guard_removes_opposite_pen_direction(self):
        def option(points, score):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(
                0,
                1,
                points,
                frozenset((int(x), int(y)) for x, y in points),
                7,
                score,
                {},
            )

        strokes = [
            {
                "componentId": 7,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0]]),
            }
        ]
        ranked = [[option([[8, 2], [0, 2]], 0.1), option([[0, 2], [8, 2]], 0.2)]]
        guarded, audit = MODULE.apply_stroke_order_guard(
            strokes,
            ranked,
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(guarded[0]), 1)
        self.assertEqual(guarded[0][0].points[0].tolist(), [0.0, 2.0])
        self.assertEqual(audit[0]["rejectedOppositeDirection"], 1)
        self.assertFalse(audit[0]["fallbackToClassicCandidates"])

    def test_hybrid_order_guard_rejects_a_horizontal_road_for_a_falling_left_stroke(self):
        def option(points, score):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(
                0,
                1,
                points,
                frozenset((int(x), int(y)) for x, y in points),
                7,
                score,
                {},
            )

        strokes = [
            {
                "componentId": 117,
                "occurrence": 0,
                "feature": "撇",
                "points": np.asarray([[5.0, 0.0], [0.0, 8.0]]),
            }
        ]
        horizontal = option([[10.0, 2.0], [0.0, 2.0]], 0.0)
        falling_left = option([[8.0, 0.0], [1.0, 9.0]], 1.0)
        guarded, audit = MODULE.apply_stroke_order_guard(
            strokes,
            [[horizontal, falling_left]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(guarded[0]), 1)
        np.testing.assert_array_equal(guarded[0][0].points, falling_left.points)
        self.assertEqual(audit[0]["rejectedOppositeDirection"], 1)

    def test_falling_left_stroke_rejects_a_westbound_initial_trunk(self):
        west_then_falling = np.asarray(
            [[10.0, 0.0], [2.0, 0.0], [1.0, 3.0], [0.0, 20.0]]
        )
        falling_from_pen_down = np.asarray(
            [[10.0, 0.0], [9.0, 2.0], [5.0, 10.0], [0.0, 20.0]]
        )
        self.assertFalse(
            MODULE.semantic_direction_compatible("撇", "SW", west_then_falling)
        )
        self.assertTrue(
            MODULE.semantic_direction_compatible("撇", "SW", falling_from_pen_down)
        )

    def test_diagonal_dot_requires_material_progress_on_both_axes(self):
        nearly_vertical = np.asarray([[10.0, 0.0], [11.0, 30.0]])
        falling_right = np.asarray([[10.0, 0.0], [18.0, 24.0]])
        self.assertFalse(
            MODULE.semantic_direction_compatible("点", "SE", nearly_vertical)
        )
        self.assertTrue(
            MODULE.semantic_direction_compatible("点", "SE", falling_right)
        )

    def test_vertical_turn_rejects_a_straight_diagonal_road(self):
        stroke = {
            "componentId": 1,
            "occurrence": 0,
            "feature": "竖折",
            "bendFractions": [0.5],
            "points": np.asarray(
                [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
            ),
        }
        diagonal = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [5.0, 10.0]]),
            frozenset((index, index) for index in range(6)),
            1,
            0.0,
            {},
        )
        vertical_turn = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [0.0, 10.0], [8.0, 10.0]]),
            frozenset({(0, 0), (0, 10), (8, 10)}),
            1,
            1.0,
            {},
        )
        guarded, _audit = MODULE.apply_stroke_order_guard(
            [stroke], [[diagonal, vertical_turn]], source="G", codepoint=0x4E00
        )
        self.assertEqual(len(guarded[0]), 1)
        np.testing.assert_array_equal(guarded[0][0].points, vertical_turn.points)

    def test_order_guard_trims_a_calligraphic_cap_before_a_falling_left_trunk(self):
        points = np.asarray(
            [[10.0, 0.0], [8.0, 0.0], [6.0, 0.0], [5.0, 2.0], [0.0, 20.0]]
        )
        option = MODULE.RouteCandidate(
            0,
            1,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            7,
            0.0,
            {},
        )
        strokes = [
            {
                "componentId": 7,
                "occurrence": 0,
                "feature": "撇",
                "points": np.asarray([[5.0, 0.0], [0.0, 20.0]]),
            }
        ]
        guarded, _audit = MODULE.apply_stroke_order_guard(
            strokes,
            [[option]],
            source="J",
            codepoint=0x6726,
        )
        self.assertEqual(len(guarded[0]), 1)
        self.assertLessEqual(float(guarded[0][0].points[0, 0]), 6.0)
        self.assertTrue(
            guarded[0][0].evidence.get("trimmedInitialSemanticSpur", False)
        )

    def test_hybrid_order_guard_does_not_restore_an_opposite_hard_direction(self):
        points = np.asarray([[8.0, 2.0], [0.0, 2.0]])
        option = MODULE.RouteCandidate(
            0, 1, points, frozenset((int(x), int(y)) for x, y in points), 7, 0.1, {}
        )
        strokes = [
            {
                "componentId": 7,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0]]),
            }
        ]
        guarded, audit = MODULE.apply_stroke_order_guard(
            strokes,
            [[option]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(guarded, [[]])
        self.assertFalse(audit[0]["fallbackToClassicCandidates"])
        self.assertTrue(audit[0]["hardDirectionConstraint"])

    def test_topology_guard_keeps_complete_rising_stroke_not_junction_fragment(self):
        graph = SimpleNamespace(
            points=np.asarray([[10, 0], [5, 10], [0, 20]], dtype=int),
            crossing=np.asarray(
                [[1] + [0] * 20]
                + [[0] * 21 for _ in range(4)]
                + [[0] * 10 + [4] + [0] * 10]
                + [[0] * 21 for _ in range(4)]
                + [[1] + [0] * 20],
                dtype=np.uint8,
            ),
        )

        def option(points, score):
            points = np.asarray(points, dtype=float)
            return MODULE.RouteCandidate(
                0, 1, points, frozenset((int(x), int(y)) for x, y in points), 1, score, {}
            )

        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "提",
                "points": np.asarray([[0.0, 10.0], [10.0, 5.0], [20.0, 0.0]]),
            },
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "竖钩",
                "points": np.asarray([[10.0, -5.0], [10.0, 15.0]]),
            },
        ]
        fragment = option([[10, 5], [20, 0]], 0.1)
        complete = option([[0, 10], [10, 5], [20, 0]], 0.2)
        vertical = option([[10, 0], [10, 10]], 0.1)
        guarded, audit = MODULE.apply_stroke_topology_guard(
            strokes,
            [[fragment, complete], [vertical]],
            graph,
        )
        self.assertEqual(len(guarded[0]), 1)
        np.testing.assert_array_equal(guarded[0][0].points, complete.points)
        self.assertEqual(audit[0]["rejectedJunctionStarts"], 1)
        self.assertFalse(audit[0]["fallbackToClassicCandidates"])

    def test_topology_guard_keeps_complete_pen_up_at_foreign_junction(self):
        graph = SimpleNamespace(
            points=np.asarray([[0, 0], [0, 10], [10, 10]], dtype=int),
            crossing=np.asarray(
                [[1] + [0] * 10]
                + [[0] * 11 for _ in range(9)]
                + [[0] * 10 + [4]],
                dtype=np.uint8,
            ),
        )
        points = np.asarray([[0.0, 0.0], [10.0, 10.0]])
        complete = MODULE.RouteCandidate(
            0,
            1,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            1,
            0.1,
            {"prematureJunctionStop": False},
        )
        strokes = [
            {
                "componentId": 1,
                "occurrence": 0,
                "points": np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            }
        ]
        guarded, audit = MODULE.apply_stroke_topology_guard(
            strokes, [[complete]], graph
        )
        self.assertEqual(len(guarded[0]), 1)
        np.testing.assert_array_equal(guarded[0][0].points, complete.points)
        self.assertFalse(guarded[0][0].evidence["candidatePenUpTouchesSibling"])
        self.assertEqual(audit[0]["keptCompleteJunctionEnds"], 1)
        self.assertEqual(audit[0]["rejectedJunctionEnds"], 0)

    def test_short_medial_axis_cap_is_trimmed_from_vertical_hook_pen_path(self):
        points = np.asarray([[5.0, 0.0], [0.0, 0.0], [0.0, 50.0], [-3.0, 53.0]])
        option = MODULE.RouteCandidate(
            0,
            1,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            1,
            0.1,
            {},
        )
        trimmed = MODULE.trim_selected_vertical_medial_spur(option, "S")
        self.assertEqual(trimmed.points[0].tolist(), [0.0, 0.0])
        self.assertEqual(trimmed.points[-1].tolist(), [-3.0, 53.0])
        self.assertEqual(trimmed.pixels, option.pixels)
        self.assertTrue(trimmed.evidence["trimmedInitialMedialSpur"])

    def test_vertical_pen_down_ignores_a_long_calligraphic_head_cap(self):
        points = np.asarray(
            [
                [8.0, 0.0],
                [6.0, 0.0],
                [4.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 8.0],
                [0.0, 20.0],
                [0.0, 32.0],
            ]
        )
        option = MODULE.RouteCandidate(
            0,
            1,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            1,
            0.1,
            {},
        )
        trimmed = MODULE.trim_initial_medial_spur(option, "S")
        self.assertEqual(trimmed.points[0].tolist(), [0.0, 0.0])
        self.assertEqual(trimmed.points[-1].tolist(), [0.0, 32.0])
        self.assertEqual(trimmed.pixels, option.pixels)

        strokes = [
            {
                "componentId": 486,
                "occurrence": 0,
                "feature": "竖",
                "points": np.asarray([[0.0, 0.0], [0.0, 20.0]]),
            }
        ]
        guarded, _audit = MODULE.apply_stroke_order_guard(
            strokes,
            [[option]],
            source="T",
            codepoint=0x64CE,
        )
        self.assertEqual(len(guarded[0]), 1)
        self.assertEqual(guarded[0][0].points[0].tolist(), [0.0, 0.0])

    def test_hook_pen_path_uses_longest_terminal_branch_not_short_ink_spur(self):
        skeleton = np.zeros((16, 16), dtype=bool)
        skeleton[1:11, 10] = True
        skeleton[10, 2:11] = True
        skeleton[10:14, 10] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        points = np.asarray([[10.0, y] for y in range(1, 14)])
        option = MODULE.RouteCandidate(
            0,
            1,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            1,
            0.1,
            {},
        )
        normalized = MODULE.normalize_hook_terminal_branch(option, graph, "竖钩")
        self.assertEqual(normalized.points[0].tolist(), [10.0, 1.0])
        self.assertEqual(normalized.points[-1].tolist(), [2.0, 10.0])
        self.assertEqual(normalized.pixels, option.pixels)
        self.assertTrue(normalized.evidence["normalizedTerminalHookBranch"])
        self.assertGreater(
            normalized.evidence["normalizedTerminalHookReplacementLength"],
            normalized.evidence["normalizedTerminalHookOriginalLength"],
        )

    def test_close_t_junction_pair_is_one_logical_contact_zone(self):
        skeleton = np.zeros((22, 25), dtype=bool)
        skeleton[1:21, 12] = True
        skeleton[6, 12:23] = True
        skeleton[14, 2:13] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        route = np.asarray([[2.0, 14.0], [12.0, 14.0], [12.0, 6.0], [22.0, 6.0]])
        trace = MODULE.critical_node_trace(route, graph, radius=4.0)
        self.assertEqual(len(trace), 1)
        self.assertEqual(len(trace[0]["nodes"]), 2)
        self.assertEqual(MODULE.turn_landmarks(route, graph=graph), [])

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

    def test_unclaimed_branch_does_not_inherit_a_finished_strokes_colour(self):
        skeleton = np.zeros((15, 15), dtype=bool)
        skeleton[7, 1:14] = True
        skeleton[2:8, 7] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        target = MODULE.cv2.dilate(
            skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        points = np.asarray([(x, 7) for x in range(1, 8)], dtype=float)
        finished = MODULE.RouteCandidate(
            0,
            0,
            points,
            frozenset((int(x), int(y)) for x, y in points),
            1,
            0.0,
            {},
        )
        owner, _distance = MODULE.geodesic_owners(target, graph, [finished])
        self.assertEqual(owner[7, 3], 0)
        self.assertEqual(owner[7, 13], -1)
        self.assertEqual(owner[2, 7], -1)

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

    def test_total_turn_ignores_small_junction_detours_on_a_long_stroke(self):
        bottom_with_junction_nubs = np.asarray(
            [
                [0.0, 20.0],
                [22.0, 20.0],
                [25.0, 17.0],
                [29.0, 20.0],
                [50.0, 20.0],
                [53.0, 17.0],
                [57.0, 20.0],
                [78.0, 20.0],
                [81.0, 17.0],
                [85.0, 20.0],
                [140.0, 20.0],
            ]
        )
        fold_with_junction_nubs = np.vstack(
            [bottom_with_junction_nubs, np.asarray([[140.0, 75.0]])]
        )
        self.assertLess(MODULE.total_turn(bottom_with_junction_nubs), 0.12)
        self.assertAlmostEqual(
            MODULE.total_turn(fold_with_junction_nubs),
            np.pi / 2,
            delta=0.12,
        )

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

    def test_equal_island_count_respects_recursive_left_right_structure(self):
        skeleton = np.zeros((13, 13), dtype=bool)
        skeleton[1:12, 1] = True
        for y in (1, 4, 7, 10):
            skeleton[y, 7:12] = True
        graph = MODULE.build_skeleton_graph(skeleton)

        def hierarchy(component, parent=None):
            items = [
                {
                    "id": component,
                    "type": "component",
                    "label": "末级部件",
                    "familyKey": str(component),
                }
            ]
            if parent is not None:
                items.append(
                    {
                        "id": parent,
                        "type": "compound",
                        "label": "⿱",
                        "familyKey": str(parent),
                    }
                )
            items.append(
                {
                    "id": 900000,
                    "type": "glyph",
                    "label": "⿰",
                    "familyKey": "900000",
                }
            )
            return items

        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "points": np.asarray([[0.0, 1.0], [0.0, 11.0]]),
                "hierarchy": hierarchy(220),
            }
        ]
        for occurrence, y in enumerate((1.0, 4.0, 7.0, 10.0)):
            strokes.append(
                {
                    "componentId": 117 + occurrence,
                    "occurrence": 0,
                    "points": np.asarray([[7.0, y], [11.0, y]]),
                    "hierarchy": hierarchy(117 + occurrence, 127533),
                }
            )
        mapping = MODULE.component_mapping(strokes, graph)
        target_centres = {
            label: np.argwhere(graph.components == label)[:, ::-1].mean(axis=0)
            for label in set(graph.components[graph.components > 0].tolist())
        }
        leftmost = min(target_centres, key=lambda label: target_centres[label][0])
        self.assertEqual(mapping[(220, 0)], leftmost)
        right_labels = sorted(
            (label for label in target_centres if label != leftmost),
            key=lambda label: target_centres[label][1],
        )
        self.assertEqual(
            [mapping[(117 + index, 0)] for index in range(4)],
            right_labels,
        )

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

    def test_component_island_is_a_soft_prior_and_never_deletes_recovery_roads(self):
        skeleton = np.zeros((9, 12), dtype=bool)
        skeleton[2, 1:10] = True
        skeleton[6, 1:10] = True
        graph = MODULE.build_skeleton_graph(skeleton)
        raw_routes = MODULE.all_routes(graph)
        strokes = [
            {
                "componentId": 506,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0]]),
            }
        ]
        original = MODULE.component_label_options
        kept_component = raw_routes[0]["component"]
        MODULE.component_label_options = lambda _strokes, _graph: {
            (506, 0): {kept_component}
        }
        try:
            legacy = MODULE.rank_routes(strokes, graph, raw_routes)
            residual = MODULE.rank_routes(
                strokes, graph, raw_routes, enforce_component_labels=False
            )
        finally:
            MODULE.component_label_options = original
        self.assertGreater(len(legacy[0]), 0)
        self.assertTrue(any(option.component == kept_component for option in legacy[0]))
        self.assertTrue(any(option.component != kept_component for option in legacy[0]))
        self.assertTrue(
            any(
                option.evidence["componentLabelRecoveryCost"] > 0
                for option in legacy[0]
            )
        )
        self.assertTrue(
            all(
                option.evidence["componentLabelRecoveryCost"] == 0
                for option in residual[0]
            )
        )

    def test_route_compaction_deduplicates_paths_and_keeps_island_diversity(self):
        path = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        best = MODULE.RouteCandidate(
            0, 1, path, frozenset({(0, 0), (10, 0)}), 1, 0.0, {}
        )
        duplicate = MODULE.RouteCandidate(
            2, 3, path.copy(), frozenset({(0, 0), (10, 0)}), 1, 1.0, {}
        )
        other_island = MODULE.RouteCandidate(
            4,
            5,
            np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            frozenset({(0, 5), (10, 5)}),
            2,
            5.0,
            {},
        )
        compacted = MODULE.compact_route_hypotheses(
            [[best, duplicate, other_island]],
            global_limit=1,
            per_component_limit=1,
            total_limit=2,
        )[0]
        self.assertEqual(compacted, [best, other_island])

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

    def test_severely_incomplete_stroke_cannot_win_with_a_better_local_score(self):
        stroke = {"points": np.asarray([[0.0, 0.0], [10.0, 0.0]])}
        short = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [2.0, 0.0]]),
            frozenset({(0, 0), (1, 0), (2, 0)}),
            1,
            -20.0,
            {"candidateLengthFraction": 0.5, "routeLengthFraction": 0.1},
        )
        complete = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 1.0], [9.0, 1.0]]),
            frozenset((x, 1) for x in range(10)),
            1,
            0.0,
            {"candidateLengthFraction": 0.5, "routeLengthFraction": 0.48},
        )
        routes, decision = MODULE.choose_routes(
            [stroke], [[short, complete]], total_skeleton_pixels=13
        )
        self.assertIs(routes[0], complete)
        self.assertFalse(decision["steps"][0]["hardIncompleteStroke"])

    def test_less_than_half_length_route_is_still_severely_incomplete(self):
        stroke = {"points": np.asarray([[0.0, 0.0], [10.0, 0.0]])}
        short = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [4.9, 0.0]]),
            frozenset((x, 0) for x in range(5)),
            1,
            -20.0,
            {"candidateLengthFraction": 0.5, "routeLengthFraction": 0.245},
        )
        complete = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 1.0], [9.0, 1.0]]),
            frozenset((x, 1) for x in range(10)),
            1,
            0.0,
            {"candidateLengthFraction": 0.5, "routeLengthFraction": 0.48},
        )
        routes, _decision = MODULE.choose_routes(
            [stroke], [[short, complete]], total_skeleton_pixels=15
        )
        self.assertIs(routes[0], complete)

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

    def test_same_leaf_requires_the_complete_pairwise_contact_matrix(self):
        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "points": np.asarray([[5.0, 0.0], [5.0, 10.0]]),
            },
            {
                "componentId": 220,
                "occurrence": 0,
                "points": np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 220,
                "occurrence": 0,
                "points": np.asarray([[0.0, 20.0], [10.0, 20.0]]),
            },
        ]
        vertical = MODULE.RouteCandidate(
            0, 1, strokes[0]["points"], frozenset((5, y) for y in range(11)), 1, 0.0, {}
        )
        missing_required_contact = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[0.0, 14.0], [10.0, 14.0]]),
            frozenset((x, 14) for x in range(11)),
            1,
            -10.0,
            {},
        )
        correct_crossing = MODULE.RouteCandidate(
            4, 5, strokes[1]["points"], frozenset((x, 5) for x in range(11)), 1, 1.0, {}
        )
        extra_forbidden_contact = MODULE.RouteCandidate(
            6, 7, strokes[1]["points"], frozenset((x, 5) for x in range(11)), 1, -10.0, {}
        )
        correct_separate = MODULE.RouteCandidate(
            8, 9, strokes[2]["points"], frozenset((x, 20) for x in range(11)), 1, 1.0, {}
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [
                [vertical],
                [missing_required_contact, correct_crossing],
                [extra_forbidden_contact, correct_separate],
            ],
            total_skeleton_pixels=33,
        )
        self.assertIs(routes[1], correct_crossing)
        self.assertIs(routes[2], correct_separate)
        self.assertNotIn("same-leaf-contact-fallback", decision["reviewReasons"])

    def test_relative_stroke_signature_is_translation_and_scale_invariant(self):
        first = np.asarray([[0.0, 0.0], [0.0, 10.0]])
        second = np.asarray([[-5.0, 7.0], [5.0, 3.0]])
        transformed_first = first * 3.0 + np.asarray([40.0, 70.0])
        transformed_second = second * 3.0 + np.asarray([40.0, 70.0])
        np.testing.assert_allclose(
            MODULE.relative_stroke_signature(first, second),
            MODULE.relative_stroke_signature(transformed_first, transformed_second),
        )

    def test_same_leaf_axis_order_backtracks_instead_of_reusing_one_road(self):
        strokes = [
            {
                "componentId": 117,
                "occurrence": 0,
                "feature": "撇",
                "points": np.asarray([[4.0, 0.0], [0.0, 8.0]]),
            },
            {
                "componentId": 117,
                "occurrence": 0,
                "feature": "捺",
                "points": np.asarray([[6.0, 0.0], [10.0, 8.0]]),
            },
        ]
        left = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[5.0, 1.0], [1.0, 9.0]]),
            frozenset({(5, 1), (4, 3), (3, 5), (2, 7), (1, 9)}),
            1,
            2.0,
            {},
        )
        duplicated_right = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[7.0, 1.0], [11.0, 9.0]]),
            frozenset({(7, 1), (8, 3), (9, 5), (10, 7), (11, 9)}),
            1,
            -20.0,
            {},
        )
        right = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[7.0, 1.0], [11.0, 9.0]]),
            duplicated_right.pixels,
            1,
            0.0,
            {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[duplicated_right, left], [right]],
            total_skeleton_pixels=10,
        )
        self.assertIs(routes[0], left)
        self.assertIs(routes[1], right)
        self.assertTrue(decision["steps"][0]["remainingRoutesFeasible"])

    def test_intersecting_same_leaf_uses_relative_position_as_soft_evidence(self):
        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "竖钩",
                "points": np.asarray([[9.0, 0.0], [9.0, 10.0]]),
            },
        ]
        horizontal = MODULE.RouteCandidate(
            0,
            1,
            strokes[0]["points"],
            frozenset((x, 5) for x in range(11)),
            1,
            0.0,
            {},
        )
        source_shifted_vertical = MODULE.RouteCandidate(
            2,
            3,
            np.asarray([[1.0, 0.0], [1.0, 10.0]]),
            frozenset((1, y) for y in range(11)),
            1,
            -10.0,
            {},
        )
        candidate_aligned_vertical = MODULE.RouteCandidate(
            4,
            5,
            strokes[1]["points"],
            frozenset((9, y) for y in range(11)),
            1,
            0.0,
            {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[horizontal], [source_shifted_vertical, candidate_aligned_vertical]],
            total_skeleton_pixels=31,
        )
        self.assertIs(routes[1], source_shifted_vertical)
        self.assertNotIn("hard-geometry-fallback", decision["reviewReasons"])

    def test_future_feasibility_prunes_a_branch_needed_by_a_later_stroke(self):
        strokes = [
            {
                "componentId": 744,
                "occurrence": 0,
                "feature": "横折",
                "points": np.asarray([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0]]),
            },
            {
                "componentId": 744,
                "occurrence": 0,
                "feature": "竖",
                "points": np.asarray([[4.0, 0.0], [4.0, 8.0]]),
            },
        ]
        steals_vertical = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[0.0, 0.0], [4.0, 0.0], [4.0, 8.0]]),
            frozenset({(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (4, 6), (4, 8)}),
            1,
            -20.0,
            {},
        )
        outer_fold = MODULE.RouteCandidate(
            0,
            2,
            np.asarray([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0]]),
            frozenset({(0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (8, 2), (8, 4), (8, 6), (8, 8)}),
            1,
            1.0,
            {},
        )
        inner_vertical = MODULE.RouteCandidate(
            3,
            4,
            np.asarray([[4.0, 0.0], [4.0, 8.0]]),
            frozenset({(4, 0), (4, 2), (4, 4), (4, 6), (4, 8)}),
            1,
            0.0,
            {},
        )
        routes, _decision = MODULE.choose_routes(
            strokes,
            [[steals_vertical, outer_fold], [inner_vertical]],
            total_skeleton_pixels=13,
        )
        self.assertIs(routes[0], outer_fold)

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

    def test_adjacent_ids_children_prefer_candidate_declared_boundary_contact(self):
        upper = {
            "componentId": 10,
            "occurrence": 0,
            "hierarchy": [{"id": 10}, {"id": 100, "label": "⿱"}],
            "points": np.asarray([[5.0, 0.0], [5.0, 5.0]]),
        }
        lower = {
            "componentId": 20,
            "occurrence": 0,
            "hierarchy": [{"id": 20}, {"id": 100, "label": "⿱"}],
            "points": np.asarray([[5.0, 5.0], [5.0, 10.0]]),
        }
        fixed_upper = MODULE.RouteCandidate(
            0, 1, upper["points"], frozenset((5, y) for y in range(6)),
            1, 0.0, {},
        )
        cheap_but_detached = MODULE.RouteCandidate(
            2, 3, np.asarray([[10.0, 7.0], [10.0, 12.0]]),
            frozenset((10, y) for y in range(7, 13)), 1, 0.0, {},
        )
        attached_boundary = MODULE.RouteCandidate(
            4, 5, np.asarray([[5.0, 5.0], [5.0, 11.0]]),
            frozenset((5, y) for y in range(5, 12)), 1, 0.1, {},
        )
        routes, decision = MODULE.choose_routes(
            [upper, lower],
            [[fixed_upper], [cheap_but_detached, attached_boundary]],
            total_skeleton_pixels=18,
        )
        self.assertIs(routes[1], attached_boundary)
        self.assertNotIn("same-leaf-contact-fallback", decision["reviewReasons"])

    def test_left_right_root_forces_first_leaf_onto_left_scan_front(self):
        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 220, "label": "末级部件"},
                    {"id": 900001, "label": "⿰"},
                ],
                "points": np.asarray([[5.0, 5.0], [20.0, 5.0]]),
            }
        ]
        wrong_right = MODULE.RouteCandidate(
            0, 1, np.asarray([[72.0, 5.0], [88.0, 5.0]]),
            frozenset({(72, 5), (80, 5), (88, 5)}), 1, -5.0, {},
        )
        correct_left = MODULE.RouteCandidate(
            0, 1, np.asarray([[4.0, 7.0], [22.0, 7.0]]),
            frozenset({(4, 7), (13, 7), (22, 7)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes, [[wrong_right, correct_left]], total_skeleton_pixels=6
        )
        self.assertIs(routes[0], correct_left)
        self.assertEqual(decision["rootStructureScan"]["operator"], "⿰")
        self.assertEqual(decision["rootStructureScan"]["vector"], [1.0, 0.0])

    def test_upper_right_enclosure_uses_negative_diagonal_scan_front(self):
        strokes = [
            {
                "componentId": 132,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": [
                    {"id": 132, "label": "末级部件"},
                    {"id": 700001, "label": "⿹"},
                ],
                "points": np.asarray([[80.0, 5.0], [60.0, 35.0]]),
            }
        ]
        wrong_lower_left = MODULE.RouteCandidate(
            0, 1, np.asarray([[8.0, 70.0], [30.0, 90.0]]),
            frozenset({(8, 70), (20, 80), (30, 90)}), 1, -5.0, {},
        )
        correct_upper_right = MODULE.RouteCandidate(
            0, 1, np.asarray([[92.0, 8.0], [70.0, 32.0]]),
            frozenset({(92, 8), (80, 20), (70, 32)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes, [[wrong_lower_left, correct_upper_right]], total_skeleton_pixels=6
        )
        self.assertIs(routes[0], correct_upper_right)
        self.assertEqual(decision["rootStructureScan"]["operator"], "⿹")
        self.assertEqual(decision["rootStructureScan"]["vector"], [-1.0, 1.0])

    def test_nested_upper_right_enclosure_scans_outer_leaf_before_inner_leaf(self):
        def hierarchy(component, immediate_parent):
            return [
                {"id": component, "label": "末级部件"},
                {"id": immediate_parent, "label": "⿹"},
                {"id": 900002, "label": "⿰"},
            ]

        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 220, "label": "末级部件"},
                    {"id": 900002, "label": "⿰"},
                ],
                "points": np.asarray([[0.0, 10.0], [15.0, 10.0]]),
            },
            {
                "componentId": 132,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": hierarchy(132, 4274),
                "points": np.asarray([[80.0, 5.0], [60.0, 35.0]]),
            },
            {
                "componentId": 282,
                "occurrence": 0,
                "feature": "竖",
                "hierarchy": hierarchy(282, 4274),
                "points": np.asarray([[55.0, 35.0], [55.0, 70.0]]),
            },
        ]
        left = MODULE.RouteCandidate(
            0, 1, np.asarray([[2.0, 12.0], [16.0, 12.0]]),
            frozenset({(2, 12), (9, 12), (16, 12)}), 1, 0.0, {},
        )
        wrong_outer = MODULE.RouteCandidate(
            0, 1, np.asarray([[30.0, 70.0], [45.0, 88.0]]),
            frozenset({(30, 70), (38, 79), (45, 88)}), 1, -5.0, {},
        )
        correct_outer = MODULE.RouteCandidate(
            0, 1, np.asarray([[92.0, 8.0], [70.0, 32.0]]),
            frozenset({(92, 8), (80, 20), (70, 32)}), 1, 0.0, {},
        )
        inner = MODULE.RouteCandidate(
            0, 1, np.asarray([[55.0, 38.0], [55.0, 72.0]]),
            frozenset({(55, 38), (55, 55), (55, 72)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[left], [wrong_outer, correct_outer], [inner]],
            total_skeleton_pixels=12,
        )
        self.assertIs(routes[1], correct_outer)
        nested = decision["steps"][1]["recursiveStructureScan"]
        self.assertTrue(nested["enabled"])
        self.assertEqual(nested["parentId"], 4274)
        self.assertEqual(nested["operator"], "⿹")

    def test_enclosed_contents_cannot_put_pen_down_outside_upper_left_cover(self):
        strokes = [
            {
                "componentId": 71,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 71, "label": "末级部件"},
                    {"id": 5384, "label": "⿸"},
                ],
                "points": np.asarray([[20.0, 20.0], [80.0, 20.0]]),
            },
            {
                "componentId": 934,
                "occurrence": 0,
                "feature": "点",
                "hierarchy": [
                    {"id": 934, "label": "末级部件"},
                    {"id": 5384, "label": "⿸"},
                ],
                "points": np.asarray([[45.0, 40.0], [50.0, 50.0]]),
            },
        ]
        cover = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[20.0, 20.0], [80.0, 20.0]]),
            frozenset({(20, 20), (50, 20), (80, 20)}),
            1,
            0.0,
            {},
        )
        wrong_above = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[45.0, 8.0], [50.0, 16.0]]),
            frozenset({(45, 8), (48, 12), (50, 16)}),
            1,
            -5.0,
            {},
        )
        correct_inside = MODULE.RouteCandidate(
            0,
            1,
            np.asarray([[45.0, 40.0], [50.0, 50.0]]),
            frozenset({(45, 40), (48, 45), (50, 50)}),
            1,
            0.0,
            {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[cover], [wrong_above, correct_inside]],
            total_skeleton_pixels=6,
        )
        self.assertIs(routes[1], correct_inside)
        domain = decision["steps"][1]["enclosureContentDomain"]
        self.assertTrue(domain["enabled"])
        self.assertEqual(domain["parentId"], 5384)
        self.assertFalse(domain["hardViolation"])

    def test_enclosure_scope_coexists_with_contact_union_groups(self):
        hierarchy = lambda component: [
            {"id": component, "label": "末级部件"},
            {"id": 5384, "label": "⿸"},
        ]
        strokes = [
            {
                "componentId": 71,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": hierarchy(71),
                "points": np.asarray([[20.0, 20.0], [80.0, 20.0]]),
            },
            {
                "componentId": 71,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": hierarchy(71),
                "points": np.asarray([[20.0, 20.0], [15.0, 80.0]]),
            },
            {
                "componentId": 934,
                "occurrence": 0,
                "feature": "点",
                "hierarchy": hierarchy(934),
                "points": np.asarray([[45.0, 40.0], [50.0, 50.0]]),
            },
        ]
        routes = [
            MODULE.RouteCandidate(
                0, 1, stroke["points"], frozenset({tuple(stroke["points"][0].astype(int)), tuple(stroke["points"][-1].astype(int))}), 1, 0.0, {}
            )
            for stroke in strokes
        ]
        selected, _decision = MODULE.choose_routes(
            strokes, [[route] for route in routes], total_skeleton_pixels=6
        )
        self.assertEqual(selected, routes)

    def test_nested_vertical_structure_rejects_lower_child_above_upper_child(self):
        strokes = [
            {
                "componentId": 486,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 486, "label": "末级部件"},
                    {"id": 25376, "label": "⿱"},
                    {"id": 900003, "label": "⿰"},
                ],
                "points": np.asarray([[40.0, 10.0], [70.0, 10.0]]),
            },
            {
                "componentId": 132,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": [
                    {"id": 132, "label": "末级部件"},
                    {"id": 4274, "label": "⿹"},
                    {"id": 25376, "label": "⿱"},
                    {"id": 900003, "label": "⿰"},
                ],
                "points": np.asarray([[60.0, 50.0], [45.0, 75.0]]),
            },
        ]
        upper = MODULE.RouteCandidate(
            0, 1, np.asarray([[40.0, 20.0], [70.0, 20.0]]),
            frozenset({(40, 20), (55, 20), (70, 20)}), 1, 0.0, {},
        )
        wrong_above = MODULE.RouteCandidate(
            0, 1, np.asarray([[62.0, 8.0], [48.0, 28.0]]),
            frozenset({(62, 8), (55, 18), (48, 28)}), 1, -5.0, {},
        )
        correct_below = MODULE.RouteCandidate(
            0, 1, np.asarray([[62.0, 50.0], [48.0, 75.0]]),
            frozenset({(62, 50), (55, 62), (48, 75)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes, [[upper], [wrong_above, correct_below]], total_skeleton_pixels=9
        )
        self.assertIs(routes[1], correct_below)
        constraint = decision["steps"][1]["recursiveSiblingOrder"]
        self.assertTrue(constraint["enabled"])
        self.assertEqual(constraint["parentId"], 25376)
        self.assertEqual(constraint["operator"], "⿱")

    def test_nested_vertical_structure_keeps_lower_child_horizontally_aligned(self):
        strokes = [
            {
                "componentId": 486,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 486, "label": "末级部件"},
                    {"id": 25376, "label": "⿱"},
                    {"id": 900003, "label": "⿰"},
                ],
                "points": np.asarray([[40.0, 10.0], [70.0, 10.0]]),
            },
            {
                "componentId": 132,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": [
                    {"id": 132, "label": "末级部件"},
                    {"id": 25376, "label": "⿱"},
                    {"id": 900003, "label": "⿰"},
                ],
                "points": np.asarray([[60.0, 50.0], [45.0, 75.0]]),
            },
        ]
        upper = MODULE.RouteCandidate(
            0, 1, np.asarray([[40.0, 20.0], [70.0, 20.0]]),
            frozenset({(40, 20), (55, 20), (70, 20)}), 1, 0.0, {},
        )
        wrong_below_but_sideways = MODULE.RouteCandidate(
            0, 1, np.asarray([[82.0, 50.0], [96.0, 75.0]]),
            frozenset({(82, 50), (89, 62), (96, 75)}), 1, -50.0, {},
        )
        correct_below = MODULE.RouteCandidate(
            0, 1, np.asarray([[62.0, 50.0], [48.0, 75.0]]),
            frozenset({(62, 50), (55, 62), (48, 75)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[upper], [wrong_below_but_sideways, correct_below]],
            total_skeleton_pixels=9,
        )
        self.assertIs(routes[1], correct_below)
        constraint = decision["steps"][1]["recursiveSiblingOrder"]
        self.assertGreater(constraint["crossAxisOverlap"], 0.02)
        self.assertFalse(constraint["hardCrossAxisMisalignment"])

    def test_same_leaf_strokes_cannot_retrace_half_of_an_existing_trunk(self):
        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "横",
                "points": np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "竖钩",
                "points": np.asarray([[5.0, 0.0], [5.0, 10.0]]),
            },
        ]
        horizontal = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            frozenset((x, 5) for x in range(11)), 1, 0.0, {},
        )
        retraces_half = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 5.0], [5.0, 5.0], [5.0, 10.0]]),
            frozenset(
                [(x, 5) for x in range(6)] + [(5, y) for y in range(6, 11)]
            ),
            1,
            -50.0,
            {},
        )
        distinct_vertical = MODULE.RouteCandidate(
            0, 1, np.asarray([[5.0, 0.0], [5.0, 10.0]]),
            frozenset((5, y) for y in range(11)), 1, 0.0, {},
        )
        routes, _decision = MODULE.choose_routes(
            strokes,
            [[horizontal], [retraces_half, distinct_vertical]],
            total_skeleton_pixels=21,
        )
        self.assertIs(routes[1], distinct_vertical)

    def test_root_left_right_scope_propagates_to_every_leaf_in_right_subtree(self):
        def right_hierarchy(component):
            return [
                {"id": component, "label": "末级部件"},
                {"id": 700010, "label": "⿱"},
                {"id": 900010, "label": "⿰"},
            ]

        strokes = [
            {
                "componentId": 220,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 220, "label": "末级部件"},
                    {"id": 900010, "label": "⿰"},
                ],
                "points": np.asarray([[0.0, 5.0], [15.0, 5.0]]),
            },
            {
                "componentId": 486,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": right_hierarchy(486),
                "points": np.asarray([[55.0, 5.0], [75.0, 5.0]]),
            },
            {
                "componentId": 132,
                "occurrence": 0,
                "feature": "撇",
                "hierarchy": right_hierarchy(132),
                "points": np.asarray([[65.0, 40.0], [55.0, 70.0]]),
            },
        ]
        left = MODULE.RouteCandidate(
            0, 1, np.asarray([[2.0, 5.0], [18.0, 5.0]]),
            frozenset({(2, 5), (10, 5), (18, 5)}), 1, 0.0, {},
        )
        right_upper = MODULE.RouteCandidate(
            0, 1, np.asarray([[55.0, 8.0], [78.0, 8.0]]),
            frozenset({(55, 8), (66, 8), (78, 8)}), 1, 0.0, {},
        )
        leaks_back_left = MODULE.RouteCandidate(
            0, 1, np.asarray([[18.0, 35.0], [5.0, 70.0]]),
            frozenset({(18, 35), (12, 52), (5, 70)}), 1, -50.0, {},
        )
        stays_right = MODULE.RouteCandidate(
            0, 1, np.asarray([[68.0, 35.0], [55.0, 70.0]]),
            frozenset({(68, 35), (62, 52), (55, 70)}), 1, 0.0, {},
        )
        routes, decision = MODULE.choose_routes(
            strokes,
            [[left], [right_upper], [leaks_back_left, stays_right]],
            total_skeleton_pixels=12,
        )
        self.assertIs(routes[2], stays_right)
        constraints = decision["steps"][2]["recursiveSiblingOrder"]["constraints"]
        self.assertTrue(any(item["parentId"] == 900010 for item in constraints))

    def test_future_structural_sibling_keeps_an_earlier_universe_alive(self):
        strokes = [
            {
                "componentId": 1,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 1, "label": "末级部件"},
                    {"id": 900020, "label": "⿰"},
                ],
                "points": np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            },
            {
                "componentId": 2,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 2, "label": "末级部件"},
                    {"id": 700019, "label": "⿹"},
                    {"id": 700020, "label": "⿰"},
                    {"id": 900020, "label": "⿰"},
                ],
                "points": np.asarray([[40.0, 5.0], [55.0, 5.0]]),
            },
            {
                "componentId": 3,
                "occurrence": 0,
                "feature": "横",
                "hierarchy": [
                    {"id": 3, "label": "末级部件"},
                    {"id": 700020, "label": "⿰"},
                    {"id": 900020, "label": "⿰"},
                ],
                "points": np.asarray([[75.0, 5.0], [90.0, 5.0]]),
            },
        ]
        root_left = MODULE.RouteCandidate(
            0, 1, np.asarray([[0.0, 5.0], [10.0, 5.0]]),
            frozenset({(0, 5), (5, 5), (10, 5)}), 1, 0.0, {},
        )
        steals_future_space = MODULE.RouteCandidate(
            0, 1, np.asarray([[78.0, 5.0], [94.0, 5.0]]),
            frozenset({(78, 5), (86, 5), (94, 5)}), 1, -50.0, {},
        )
        leaves_future_space = MODULE.RouteCandidate(
            0, 1, np.asarray([[38.0, 5.0], [54.0, 5.0]]),
            frozenset({(38, 5), (46, 5), (54, 5)}), 1, 0.0, {},
        )
        future_right = MODULE.RouteCandidate(
            0, 1, np.asarray([[76.0, 5.0], [92.0, 5.0]]),
            frozenset({(76, 5), (84, 5), (92, 5)}), 1, 0.0, {},
        )
        routes, _decision = MODULE.choose_routes(
            strokes,
            [[root_left], [steals_future_space, leaves_future_space], [future_right]],
            total_skeleton_pixels=12,
        )
        self.assertIs(routes[1], leaves_future_space)

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
        self.assertIn("Shift＋滚轮＝全页正倒放", page)
        self.assertIn("function scrubTimeline(event)", page)
        self.assertIn(
            "if(!event.shiftKey||event.ctrlKey||event.altKey||event.metaKey)return",
            page,
        )
        self.assertIn(
            "document.addEventListener('wheel',scrubTimeline,{passive:false})",
            page,
        )
        self.assertIn("event.preventDefault();stopPlayback()", page)
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
