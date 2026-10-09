import numpy as np
import pytest

from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.leaf_alignment import LeafAlignment
from glyph_decomposer.primitive_fit import FittedPrimitive, FittedStroke
from glyph_decomposer.primitive_route import (
    PrimitiveRoute,
    _endpoint_candidates,
    _single_curve_direction_compatible,
    _single_curve_turn_compatible,
    enumerate_primitive_routes,
    select_joint_routes,
)
from glyph_decomposer.stroke_grammar import canonical_stroke_grammar


def test_endpoint_candidates_reserve_spatially_diverse_modes():
    component = frozenset(
        (y, x)
        for x, y in (
            (61, 70),
            (64, 69),
            (58, 72),
            (56, 74),
            (61, 62),
            (56, 64),
            (65, 78),
            (58, 61),
            (52, 66),
            (54, 76),
            (71, 64),
            (73, 67),
            (50, 68),
            (58, 58),
        )
    )

    selected = _endpoint_candidates(
        component,
        np.asarray((62.0, 70.0)),
        1.0,
        count=7,
        diverse=True,
    )

    assert len(selected) == 7
    assert any(x >= 70 for _y, x in selected)


def test_single_curve_direction_is_a_hard_stroke_grammar_constraint():
    grammar = canonical_stroke_grammar("点", ("c",))
    candidate = ((10, 10), (10, 20))

    assert _single_curve_direction_compatible(
        ((10, 10), (7, 20)), grammar, candidate
    )
    assert not _single_curve_direction_compatible(
        ((17, 10), (10, 15)), grammar, candidate
    )
    assert not _single_curve_direction_compatible(
        ((10, 10), (2, 20)), grammar, candidate, minimum_cosine=0.85
    )


def _cubic_fit(controls):
    primitive = FittedPrimitive(
        "c",
        "cubic",
        tuple(controls),
        (controls[0], controls[-1]),
        0,
        1,
        0.0,
    )
    return FittedStroke(
        ("c",),
        (primitive,),
        (controls[0], controls[-1]),
        (controls[0], controls[-1]),
        0.0,
        0.0,
        4,
        1.0,
    )


def test_nak_stroke_rejects_an_s_shaped_cubic_but_accepts_one_way_turn():
    grammar = canonical_stroke_grammar("捺", ("c",))
    one_way = _cubic_fit(((0, 0), (0, 4), (4, 9), (10, 10)))
    numerical_wobble = _cubic_fit(((0, 0), (3, 3.2), (7, 6.8), (10, 10)))
    inflected = _cubic_fit(((0, 0), (10, 0), (0, 10), (10, 10)))

    assert _single_curve_turn_compatible(one_way, grammar)
    assert _single_curve_turn_compatible(numerical_wobble, grammar)
    assert not _single_curve_turn_compatible(inflected, grammar)


def test_single_curve_turn_rule_is_specific_to_nak_family():
    grammar = canonical_stroke_grammar("点", ("c",))
    inflected = _cubic_fit(((0, 0), (10, 0), (0, 10), (10, 10)))

    assert _single_curve_turn_compatible(inflected, grammar)


def test_primitive_route_enumeration_is_ranked_and_truth_free():
    records = [
        {
            "id": 1,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 50],
                    "curveList": [{"command": "h", "parameterList": [80]}],
                }
            ],
        }
    ]
    skeleton = np.zeros((101, 101), dtype=bool)
    skeleton[50, 10:91] = True
    skeleton[48:53, 30] = True  # a small nuisance spur
    graph = compile_candidate_graph(GlyphRepository(records), 1)

    routes = enumerate_primitive_routes(skeleton, graph, graph.strokes[0])

    assert routes
    assert [route.score for route in routes] == sorted(route.score for route in routes)
    assert routes[0].fit.control_point_count == 2
    assert routes[0].alternative_count == len(routes)
    assert all(route.stroke_index == 0 for route in routes)


def test_primitive_route_keeps_a_close_second_skeleton_component():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [30, 50],
                            "curveList": [{"command": "h", "parameterList": [40]}],
                        }
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((101, 101), dtype=bool)
    skeleton[48, 10:91] = True  # closer but overlong decoy
    skeleton[51, 30:71] = True  # close runner-up, shape-compatible component

    alignment = LeafAlignment((), (30, 48, 70, 49), 0, 1, 1, 0, 0, ())
    routes = enumerate_primitive_routes(
        skeleton, graph, graph.strokes[0], leaf_alignment=alignment
    )

    route_rows = {pixel[0] for route in routes for pixel in route.pixels}
    assert 48 in route_rows
    assert 51 in route_rows


def test_primitive_route_rejects_alignment_for_another_leaf():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [10, 50],
                            "curveList": [{"command": "h", "parameterList": [80]}],
                        }
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((101, 101), dtype=bool)
    skeleton[50, 10:91] = True
    wrong = LeafAlignment((1,), (10, 50, 90, 50), 0, 1, 1, 0, 0, ())

    with pytest.raises(ValueError, match="does not own"):
        enumerate_primitive_routes(
            skeleton, graph, graph.strokes[0], leaf_alignment=wrong
        )


def test_joint_selection_avoids_reusing_a_whole_route():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [0, 0],
                            "curveList": [{"command": "h", "parameterList": [10]}],
                        }
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[5, 1:11] = True
    skeleton[12, 1:11] = True
    first = enumerate_primitive_routes(skeleton, graph, graph.strokes[0])[:1]
    assert len(first) == 1
    duplicated = PrimitiveRoute(
        1,
        first[0].pixels,
        first[0].fit,
        first[0].score,
        first[0].endpoint_cost,
        first[0].guide_cost,
        first[0].length_cost,
        first[0].direction_cost,
        first[0].region_cost,
        first[0].placement_confidence,
        2,
    )
    alternate = PrimitiveRoute(
        1,
        tuple((12, x) for x in range(1, 11)),
        first[0].fit,
        first[0].score + 0.1,
        first[0].endpoint_cost,
        first[0].guide_cost,
        first[0].length_cost,
        first[0].direction_cost,
        first[0].region_cost,
        first[0].placement_confidence,
        2,
    )

    solution = select_joint_routes(
        (first[:1], (duplicated, alternate)),
        int(skeleton.sum()),
        free_contact_pixels=0,
    )

    assert solution is not None
    assert solution.routes[1].pixels == alternate.pixels
    assert solution.repeated_pixel_count == 0
    assert solution.stroke_relation_cost == 0.0


def test_joint_selection_rejects_globally_useful_but_locally_bad_route():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [0, 0],
                            "curveList": [{"command": "h", "parameterList": [10]}],
                        }
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[5, 1:11] = True
    skeleton[12, 1:11] = True
    best = enumerate_primitive_routes(skeleton, graph, graph.strokes[0])[:1]
    locally_bad = PrimitiveRoute(
        1,
        tuple((12, x) for x in range(1, 11)),
        best[0].fit,
        best[0].score + 3.0,
        best[0].endpoint_cost,
        best[0].guide_cost,
        best[0].length_cost,
        best[0].direction_cost,
        best[0].region_cost,
        best[0].placement_confidence,
        2,
    )
    duplicate = PrimitiveRoute(
        1,
        best[0].pixels,
        best[0].fit,
        best[0].score,
        best[0].endpoint_cost,
        best[0].guide_cost,
        best[0].length_cost,
        best[0].direction_cost,
        best[0].region_cost,
        best[0].placement_confidence,
        2,
    )

    solution = select_joint_routes(
        (best, (duplicate, locally_bad)),
        int(skeleton.sum()),
        free_contact_pixels=0,
    )

    assert solution is not None
    assert solution.routes[1].pixels == duplicate.pixels


def test_joint_selection_locks_local_winner_at_full_placement_confidence():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [0, 0],
                            "curveList": [{"command": "h", "parameterList": [10]}],
                        }
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[5, 1:11] = True
    skeleton[12, 1:11] = True
    template = enumerate_primitive_routes(skeleton, graph, graph.strokes[0])[0]

    def route(index, pixels, score):
        return PrimitiveRoute(
            index,
            tuple(pixels),
            template.fit,
            score,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2,
        )

    first = route(0, tuple((5, x) for x in range(1, 11)), 1.0)
    duplicate = route(1, first.pixels, 1.0)
    globally_useful = route(1, tuple((12, x) for x in range(1, 11)), 1.1)
    solution = select_joint_routes(
        ((first,), (duplicate, globally_useful)),
        int(skeleton.sum()),
        free_contact_pixels=0,
    )

    assert solution is not None
    assert solution.routes[1].pixels == duplicate.pixels


def test_joint_selection_respects_recursive_ids_order():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [10, 50],
                            "curveList": [{"command": "h", "parameterList": [20]}],
                        }
                    ],
                },
                {
                    "id": 2,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [10, 50],
                            "curveList": [{"command": "h", "parameterList": [20]}],
                        }
                    ],
                },
                {
                    "id": 3,
                    "type": "compound",
                    "operator": "⿰",
                    "references": [{"id": 1}, {"id": 2}],
                },
            ]
        ),
        3,
    )
    skeleton = np.zeros((30, 30), dtype=bool)
    skeleton[10, 2:10] = True
    skeleton[10, 20:28] = True
    template = enumerate_primitive_routes(
        skeleton, graph, graph.strokes[0], endpoint_count=2
    )[0]

    def route(index, pixels):
        return PrimitiveRoute(
            index,
            tuple(pixels),
            template.fit,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2,
        )

    left = tuple((10, x) for x in range(2, 10))
    right = tuple((10, x) for x in range(20, 28))
    solution = select_joint_routes(
        ((route(0, left), route(0, right)), (route(1, left), route(1, right))),
        int(skeleton.sum()),
        candidate=graph,
        free_contact_pixels=0,
    )

    assert solution is not None
    assert solution.routes[0].pixels == left
    assert solution.routes[1].pixels == right
    assert solution.ids_structure_cost == 0.0


def test_joint_selection_preserves_same_leaf_intersection_position():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "竖",
                            "start": [50, 10],
                            "curveList": [{"command": "v", "parameterList": [80]}],
                        },
                        {
                            "feature": "横",
                            "start": [10, 50],
                            "curveList": [{"command": "h", "parameterList": [80]}],
                        },
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((101, 101), dtype=bool)
    skeleton[10:91, 50] = True
    skeleton[50, 10:91] = True
    skeleton[85, 10:51] = True
    first = enumerate_primitive_routes(skeleton, graph, graph.strokes[0])[0]
    template = enumerate_primitive_routes(skeleton, graph, graph.strokes[1])[0]

    def horizontal(y, score=1.0):
        return PrimitiveRoute(
            1,
            tuple((y, x) for x in range(10, 91)),
            template.fit,
            score,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            2,
        )

    solution = select_joint_routes(
        ((first,), (horizontal(85), horizontal(50))),
        int(skeleton.sum()),
        candidate=graph,
        free_contact_pixels=0,
    )

    assert solution is not None
    assert solution.routes[1].pixels == horizontal(50).pixels
    assert solution.stroke_relation_cost < 0.1
