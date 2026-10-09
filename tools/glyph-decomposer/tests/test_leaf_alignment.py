import numpy as np

from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.leaf_alignment import (
    LeafAlignment,
    _alignment_ids_cost,
    _alignment_ids_specs,
    _hypothesis_bounds,
    search_leaf_alignments,
    select_joint_leaf_alignments,
)


def _alignment(path, bounds, score=0.0):
    return LeafAlignment(path, bounds, score, 1.0, 1.0, 0.0, 0.0, ())


def test_leaf_alignment_recovers_translation_and_anisotropic_scale():
    graph = compile_candidate_graph(
        GlyphRepository(
            [
                {
                    "id": 1,
                    "type": "component",
                    "strokes": [
                        {
                            "feature": "横",
                            "start": [10, 20],
                            "curveList": [{"command": "h", "parameterList": [30]}],
                        },
                        {
                            "feature": "竖",
                            "start": [10, 20],
                            "curveList": [{"command": "v", "parameterList": [40]}],
                        },
                    ],
                }
            ]
        ),
        1,
    )
    skeleton = np.zeros((100, 100), dtype=bool)
    skeleton[30, 45:76] = True
    skeleton[30:81, 45] = True

    alignments = search_leaf_alignments(skeleton, graph, (), count=3)

    assert alignments
    x0, y0, x1, y1 = alignments[0].bounds
    assert abs(x0 - 45) <= 3
    assert abs(y0 - 30) <= 3
    assert abs(x1 - 75) <= 4
    assert abs(y1 - 80) <= 4


def test_joint_leaf_alignment_uses_both_ids_sides_exclusively():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 30],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                },
                {
                    "feature": "竖",
                    "start": [10, 30],
                    "curveList": [{"command": "v", "parameterList": [30]}],
                },
            ],
        }
        for identifier in (1, 2)
    ]
    records.append(
        {
            "id": 3,
            "type": "compound",
            "operator": "⿰",
            "references": [{"id": 1}, {"id": 2}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 3)
    skeleton = np.zeros((100, 100), dtype=bool)
    for x in (20, 70):
        skeleton[25, x : x + 21] = True
        skeleton[25:66, x] = True
    by_path = {
        path: search_leaf_alignments(skeleton, graph, path, count=12)
        for path in ((0,), (1,))
    }

    solution = select_joint_leaf_alignments(by_path, graph)

    assert solution is not None
    centers = [
        (item.bounds[0] + item.bounds[2]) / 2 for item in solution.alignments
    ]
    assert centers[0] < centers[1]
    assert solution.repeated_claim_count == 0


def test_joint_leaf_alignment_covers_the_whole_skeleton_envelope():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 20],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                }
            ],
        }
        for identifier in (1, 2)
    ]
    records.append(
        {
            "id": 3,
            "type": "compound",
            "operator": "⿱",
            "references": [{"id": 1}, {"id": 2}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 3)
    skeleton = np.zeros((100, 100), dtype=bool)
    skeleton[10, 20:81] = True
    skeleton[80, 20:81] = True
    by_path = {
        (0,): (
            _alignment((0,), (20, 10, 80, 20)),
            _alignment((0,), (20, 35, 80, 45)),
        ),
        (1,): (_alignment((1,), (20, 70, 80, 80)),),
    }

    solution = select_joint_leaf_alignments(by_path, graph, skeleton=skeleton)

    assert solution is not None
    assert solution.alignments[0].bounds == (20, 10, 80, 20)


def test_joint_leaf_alignment_does_not_reorder_an_adequate_envelope():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 20],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                }
            ],
        }
        for identifier in (1, 2)
    ]
    records.append(
        {
            "id": 3,
            "type": "compound",
            "operator": "⿱",
            "references": [{"id": 1}, {"id": 2}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 3)
    skeleton = np.zeros((100, 100), dtype=bool)
    skeleton[10, 20:81] = True
    skeleton[80, 20:81] = True
    locally_better = _alignment((0,), (20, 20, 80, 30), score=0.0)
    exact_envelope = _alignment((0,), (20, 10, 80, 20), score=1.0)
    by_path = {
        (0,): (locally_better, exact_envelope),
        (1,): (_alignment((1,), (20, 70, 80, 80)),),
    }

    solution = select_joint_leaf_alignments(by_path, graph, skeleton=skeleton)

    assert solution is not None
    assert solution.alignments[0] == locally_better


def test_hypothesis_bounds_preserve_center_while_scaling():
    bounds = _hypothesis_bounds((10, 20, 30, 60), 2, 0.5, 5, -5)

    assert bounds == (5.0, 25.0, 45.0, 45.0)


def test_top_bottom_ids_rejects_side_by_side_leaf_placements():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 20],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                }
            ],
        }
        for identifier in (1, 2)
    ]
    records.append(
        {
            "id": 3,
            "type": "compound",
            "operator": "⿱",
            "references": [{"id": 1}, {"id": 2}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 3)
    specs = _alignment_ids_specs(graph, ((0,), (1,)))
    aligned = (
        _alignment((0,), (20, 10, 80, 30)),
        _alignment((1,), (25, 50, 85, 80)),
    )
    side_by_side = (
        _alignment((0,), (0, 10, 35, 30)),
        _alignment((1,), (65, 12, 100, 80)),
    )

    assert _alignment_ids_cost(specs, aligned) < _alignment_ids_cost(
        specs, side_by_side
    )


def test_top_bottom_ids_rejects_nearly_coincident_vertical_bands():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 20],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                }
            ],
        }
        for identifier in (1, 2)
    ]
    records.append(
        {
            "id": 3,
            "type": "compound",
            "operator": "⿱",
            "references": [{"id": 1}, {"id": 2}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 3)
    specs = _alignment_ids_specs(graph, ((0,), (1,)))
    separated = (
        _alignment((0,), (20, 10, 80, 35)),
        _alignment((1,), (20, 40, 80, 80)),
    )
    coincident = (
        _alignment((0,), (20, 35, 80, 60)),
        _alignment((1,), (20, 38, 80, 80)),
    )

    assert _alignment_ids_cost(specs, separated) < _alignment_ids_cost(
        specs, coincident
    )


def test_ids_cost_waits_for_complete_recursive_subtrees():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 20],
                    "curveList": [{"command": "h", "parameterList": [30]}],
                }
            ],
        }
        for identifier in (1, 2, 3)
    ]
    records.extend(
        [
            {
                "id": 4,
                "type": "compound",
                "operator": "⿱",
                "references": [{"id": 2}, {"id": 3}],
            },
            {
                "id": 5,
                "type": "compound",
                "operator": "⿰",
                "references": [{"id": 1}, {"id": 4}],
            },
        ]
    )
    graph = compile_candidate_graph(GlyphRepository(records), 5)
    all_paths = ((0,), (1, 0), (1, 1))
    specs = _alignment_ids_specs(graph, all_paths)
    partial = (
        _alignment((0,), (70, 10, 90, 90)),
        _alignment((1, 0), (10, 10, 30, 30)),
    )

    assert _alignment_ids_cost(specs, partial) == 0.0


def test_ternary_top_middle_bottom_ids_preserves_all_three_levels():
    records = [
        {
            "id": identifier,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [10, 50],
                    "curveList": [{"command": "h", "parameterList": [80]}],
                }
            ],
        }
        for identifier in (1, 2, 3)
    ]
    records.append(
        {
            "id": 4,
            "type": "compound",
            "operator": "⿳",
            "references": [{"id": 1}, {"id": 2}, {"id": 3}],
        }
    )
    graph = compile_candidate_graph(GlyphRepository(records), 4)
    specs = _alignment_ids_specs(graph, ((0,), (1,), (2,)))
    ordered = (
        _alignment((0,), (10, 5, 90, 25)),
        _alignment((1,), (10, 40, 90, 60)),
        _alignment((2,), (10, 75, 90, 95)),
    )
    swapped = (ordered[2], ordered[1], ordered[0])
    swapped = tuple(
        LeafAlignment(path, item.bounds, 0, 1, 1, 0, 0, ())
        for path, item in zip(((0,), (1,), (2,)), swapped)
    )

    assert len(specs) == 2
    assert _alignment_ids_cost(specs, ordered) < _alignment_ids_cost(specs, swapped)

