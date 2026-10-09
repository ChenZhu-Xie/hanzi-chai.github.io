import numpy as np

from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.leaf_alignment import LeafAlignment
from glyph_decomposer.leaf_route_selection import (
    _avoids_locked_segments,
    _needs_route_refinement,
    _refinement_shortlist,
    _select_refinement_trial,
    refine_low_confidence_leaf_routes,
)
from glyph_decomposer.primitive_fit import FittedStroke
from glyph_decomposer.primitive_route import JointRouteSolution, PrimitiveRoute


def _alignment(scale_x, scale_y, x, score):
    return LeafAlignment(
        (),
        (x, 0, x + 10, 10),
        score,
        scale_x,
        scale_y,
        0,
        0,
        (),
    )


def test_refinement_shortlist_keeps_two_spatial_modes_per_natural_scale():
    current = _alignment(0.6, 0.6, 0, 1)
    pool = (
        current,
        _alignment(1.0, 1.0, 60, 2),
        _alignment(1.0, 1.0, 10, 3),
        _alignment(1.0, 1.0, 30, 4),
        _alignment(2.2, 2.2, 10, 1.5),
    )

    shortlist = _refinement_shortlist(pool, current)

    assert shortlist == pool[:3]


def test_two_stroke_leaf_is_refined_only_for_extreme_enlargement():
    ordinary = _alignment(1.3, 0.8, 0, 1)
    extreme = _alignment(1.7, 2.8, 0, 1)

    assert not _needs_route_refinement(ordinary, 2)
    assert _needs_route_refinement(extreme, 2)
    assert _needs_route_refinement(ordinary, 3)


def test_high_leaf_alignment_error_triggers_route_refinement():
    poor_fit = LeafAlignment((), (0, 0, 10, 10), 3.1, 1, 1, 0, 0, ())

    assert _needs_route_refinement(poor_fit, 3)


def _joint(score, repeated, ids_cost):
    return JointRouteSolution((), score, 0, repeated, 0, ids_cost, 0.0)


def test_locked_leaf_allows_a_contact_but_rejects_a_shared_segment():
    locked = frozenset((5, x) for x in range(2, 8))

    assert _avoids_locked_segments(((4, 4), (5, 4), (6, 4)), locked)
    assert not _avoids_locked_segments(
        ((5, 1), (5, 2), (5, 3), (5, 4), (5, 5)),
        locked,
        maximum_shared_run=2,
    )


def test_refinement_requires_whole_glyph_improvement_without_regressions():
    original = _alignment(1.0, 0.8, 0, 3.5)
    improved = _alignment(1.0, 1.3, 0, 2.5)
    lower_score_but_more_reuse = _alignment(1.0, 1.0, 0, 2.0)
    lower_score_but_broken_ids = _alignment(1.0, 1.1, 0, 2.1)
    baseline = _joint(100.0, 4, 0.10)

    selected_alignment, selected_solution = _select_refinement_trial(
        original,
        baseline,
        (
            (lower_score_but_more_reuse, _joint(70.0, 5, 0.10)),
            (lower_score_but_broken_ids, _joint(60.0, 4, 0.25)),
            (improved, _joint(80.0, 4, 0.12)),
        ),
        maximum_ids_regression=0.05,
    )

    assert selected_alignment == improved
    assert selected_solution.score == 80.0


def test_low_confidence_leaf_is_refit_while_other_leaf_routes_stay_locked():
    repository = GlyphRepository(
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
            },
            {
                "id": 2,
                "type": "component",
                "strokes": [
                    {
                        "feature": "横",
                        "start": [10, 25],
                        "curveList": [{"command": "h", "parameterList": [80]}],
                    },
                    {
                        "feature": "横",
                        "start": [10, 75],
                        "curveList": [{"command": "h", "parameterList": [80]}],
                    },
                ],
            },
            {
                "id": 3,
                "type": "compound",
                "operator": "⿱",
                "references": [{"id": 1}, {"id": 2}],
            },
        ]
    )
    candidate = compile_candidate_graph(repository, 3)
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[2, 1:11] = True
    skeleton[10, 1:11] = True
    skeleton[12, 1:11] = True
    skeleton[15, 1:11] = True
    fit = FittedStroke((), (), (), (), 0.0, 0.0, 0, 1.0)

    def route(index, row, score):
        return PrimitiveRoute(
            index,
            tuple((row, x) for x in range(1, 11)),
            fit,
            score,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
        )

    locked = route(0, 2, 1.0)
    old_first = route(1, 10, 20.0)
    old_second = route(2, 12, 20.0)
    new_first = route(1, 10, 1.0)
    new_second = route(2, 15, 1.0)
    top = LeafAlignment((0,), (1, 2, 10, 2), 1.0, 1, 1, 0, 0, ())
    old = LeafAlignment((1,), (1, 10, 10, 12), 3.5, 1, 0.8, 0, 0, ())
    expanded = LeafAlignment((1,), (1, 10, 10, 15), 2.0, 1, 1.3, 0, 0, ())
    baseline = JointRouteSolution(
        (locked, old_first, old_second), 100.0, 30, 0, 10, 0.0, 20.0
    )

    def enumerate_routes(_skeleton, _candidate, stroke, *, leaf_alignment, **_):
        if leaf_alignment == expanded:
            return (new_first,) if stroke.index == 1 else (new_second,)
        return (old_first,) if stroke.index == 1 else (old_second,)

    alignments, alternatives, joint, refinements = refine_low_confidence_leaf_routes(
        skeleton,
        candidate,
        {(0,): (top,), (1,): (old, expanded)},
        {(0,): top, (1,): old},
        ((locked,), (old_first,), (old_second,)),
        baseline,
        route_enumerator=enumerate_routes,
    )

    assert alignments[(1,)] == expanded
    assert joint.score < baseline.score
    assert joint.routes[0] == locked
    assert alternatives[1] == (new_first,)
    assert alternatives[2] == (new_second,)
    assert refinements[0].changed


def test_healthy_joint_solution_skips_expensive_alignment_evaluations():
    repository = GlyphRepository(
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
    )
    candidate = compile_candidate_graph(repository, 1)
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[10, 1:11] = True
    fit = FittedStroke((), (), (), (), 0.0, 0.0, 0, 1.0)
    route = PrimitiveRoute(
        0,
        tuple((10, x) for x in range(1, 11)),
        fit,
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1,
    )
    original = LeafAlignment((), (1, 10, 10, 10), 3.5, 1, 1, 0, 0, ())
    pool = (original,) + tuple(
        LeafAlignment((), (1, y, 10, y + 1), 4 + y, 1, 1, 0, y, ()) for y in range(20)
    )
    calls = 0

    def enumerate_routes(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (route,)

    refine_low_confidence_leaf_routes(
        skeleton,
        candidate,
        {(): pool},
        {(): original},
        ((route,),),
        JointRouteSolution((route,), 20.0, 10, 0, 0, 0.0, 0.0),
        route_enumerator=enumerate_routes,
    )

    assert calls == 0
