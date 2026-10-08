from shapely.geometry import Point, box
from shapely.ops import unary_union

from glyph_decomposer.atoms import assign_vector_atoms
from glyph_decomposer.candidate import StrokeSeed


def test_vector_atoms_cover_ink_once_and_preserve_leaf_ownership():
    geometry = unary_union([box(2, 2, 98, 24), box(2, 24, 24, 98), box(46, 46, 94, 94)])
    seeds = (
        StrokeSeed(1, 0, "横", ((2, 13), (98, 13))),
        StrokeSeed(1, 0, "竖", ((13, 2), (13, 98))),
        StrokeSeed(2, 0, "横", ((46, 70), (94, 70))),
        StrokeSeed(2, 0, "竖", ((70, 46), (70, 94))),
    )

    result = assign_vector_atoms(geometry, seeds, (frozenset({1}), frozenset({2})))

    first, second = result.children
    assert first.union(second).symmetric_difference(geometry).area < 1e-6
    assert first.intersection(second).area < 1e-6
    assert first.contains(box(4, 4, 20, 20).centroid)
    assert second.contains(box(50, 50, 90, 90).centroid)
    assert result.evidence.solver_status == "VECTOR_VORONOI"
    assert result.evidence.seed_coverage is not None
    assert result.evidence.alignment_iou is not None
    assert result.evidence.atom_count > 0


def test_crossing_strokes_keep_identity_and_later_stroke_owns_junction():
    geometry = unary_union([box(5, 43, 95, 57), box(43, 5, 57, 95)])
    seeds = (
        StrokeSeed(1, 0, "横", ((5, 50), (95, 50))),
        StrokeSeed(2, 0, "竖", ((50, 5), (50, 95))),
    )

    result = assign_vector_atoms(geometry, seeds, (frozenset({1}), frozenset({2})))

    assert result.children[1].covers(Point(50, 50))
    assert result.children[0].intersection(result.children[1]).area < 1e-6
    assert result.evidence.stroke_count == 2
    assert result.evidence.junction_count == 1
    assert result.evidence.stroke_independence == 1
    assert result.evidence.stroke_continuity > 0.95
    assert result.evidence.stroke_inertia > 0.99
    assert result.evidence.orphan_ink_ratio < 0.01


def test_disconnected_voronoi_fragments_grow_from_whole_stroke_cores():
    geometry = unary_union(
        [
            box(4, 43, 96, 57),
            box(43, 4, 57, 96),
            # A remote serif-like island must not inherit a colour merely
            # because it shares a point site's unbounded Voronoi cell.
            box(62, 62, 68, 68),
        ]
    )
    seeds = (
        StrokeSeed(1, 0, "横", ((4, 50), (96, 50))),
        StrokeSeed(2, 0, "竖", ((50, 4), (50, 96))),
    )

    result = assign_vector_atoms(geometry, seeds, (frozenset({1}), frozenset({2})))

    assert result.children[0].intersection(result.children[1]).area < 1e-6
    assert (
        result.children[0].union(result.children[1]).symmetric_difference(geometry).area
        < 1e-6
    )
    assert result.evidence.stroke_inertia > 0.9
    assert result.evidence.orphan_ink_ratio > 0
