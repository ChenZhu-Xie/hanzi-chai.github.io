import numpy as np

from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.cut_hypotheses import generate_root_cut_hypotheses
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.topology import compress_skeleton
from glyph_decomposer.topology_features import describe_topology


def _candidate(operator: str):
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
        },
        {
            "id": 2,
            "type": "compound",
            "operator": operator,
            "references": [{"id": 1}, {"id": 1}],
        },
    ]
    return compile_candidate_graph(GlyphRepository(records), 2)


def test_left_right_ids_proposes_central_two_side_junction_but_does_not_accept():
    mask = np.zeros((101, 101), dtype=bool)
    mask[50, 20:81] = True
    mask[25:76, 50] = True
    topology = compress_skeleton(mask)
    descriptors = describe_topology(topology, 101)

    hypotheses = generate_root_cut_hypotheses(
        topology, descriptors, _candidate("⿰"), 101
    )

    assert hypotheses
    assert hypotheses[0].first_edges
    assert hypotheses[0].second_edges
    assert hypotheses[0].auto_accepted is False
    assert hypotheses[0].structural_gain_over_no_cut > 0.5


def test_left_right_ids_ignores_junction_with_branches_only_on_one_side():
    mask = np.zeros((101, 101), dtype=bool)
    mask[50, 5:31] = True
    mask[35:66, 20] = True
    topology = compress_skeleton(mask)
    descriptors = describe_topology(topology, 101)

    hypotheses = generate_root_cut_hypotheses(
        topology, descriptors, _candidate("⿰"), 101
    )

    assert hypotheses == ()


def test_top_bottom_ids_uses_vertical_branch_sides():
    mask = np.zeros((101, 101), dtype=bool)
    mask[20:81, 50] = True
    mask[50, 25:76] = True
    topology = compress_skeleton(mask)
    descriptors = describe_topology(topology, 101)

    hypotheses = generate_root_cut_hypotheses(
        topology, descriptors, _candidate("⿱"), 101
    )

    assert hypotheses
    assert hypotheses[0].axis == "y"
