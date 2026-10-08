from shapely.geometry import box
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
