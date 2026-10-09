from glyph_decomposer.leaf_alignment import LeafAlignment
from glyph_decomposer.leaf_route_selection import (
    _needs_route_refinement,
    _refinement_shortlist,
)


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
