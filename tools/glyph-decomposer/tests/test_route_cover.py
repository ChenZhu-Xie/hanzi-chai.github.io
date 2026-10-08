import numpy as np

from glyph_decomposer.candidate_graph import (
    compile_candidate_graph,
    compile_catalog_candidate_graph,
)
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.route_cover import (
    StrokeRoute,
    assign_topology_edges,
    find_partition_cuts,
    find_semantic_cuts,
    match_candidate_routes,
)
from glyph_decomposer.topology import compress_skeleton


def _graph():
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
            "type": "component",
            "strokes": [
                {
                    "feature": "竖",
                    "start": [50, 10],
                    "curveList": [{"command": "v", "parameterList": [80]}],
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
    return compile_candidate_graph(GlyphRepository(records), 3)


def _cross():
    mask = np.zeros((101, 101), dtype=bool)
    mask[50, 10:91] = True
    mask[10:91, 50] = True
    return mask


def test_cross_leaf_routes_request_semantic_cut_at_shared_junction():
    skeleton = _cross()
    routes = (
        StrokeRoute(
            0, "横", ((0,), 1), tuple((50, x) for x in range(10, 91)), (), 0, 0, 80, 80
        ),
        StrokeRoute(
            1, "竖", ((1,), 2), tuple((y, 50) for y in range(10, 91)), (), 0, 0, 80, 80
        ),
    )
    cuts = find_semantic_cuts(compress_skeleton(skeleton), routes, 101)

    assert len(routes) == 2
    assert cuts
    assert any(len(cut.leaf_keys) == 2 for cut in cuts)
    assert all(cut.auto_accepted is False for cut in cuts)


def test_one_leaf_owner_does_not_request_component_cut():
    skeleton = _cross()
    same_owner = (
        StrokeRoute(
            0, "横", ((0,), 1), tuple((50, x) for x in range(10, 91)), (), 0, 0, 80, 80
        ),
        StrokeRoute(
            1, "竖", ((0,), 1), tuple((y, 50) for y in range(10, 91)), (), 0, 0, 80, 80
        ),
    )

    assert find_semantic_cuts(compress_skeleton(skeleton), same_owner, 101) == ()


def test_candidate_routes_are_matched_without_truth():
    routes, unmatched = match_candidate_routes(_cross(), _graph())

    assert unmatched == ()
    assert len(routes) == 2


def test_review_catalog_compiles_synthetic_root_without_annotation_truth():
    catalog = {
        "rows": [
            {
                "unicode": 0x64CF,
                "candidates": {
                    "900001": [
                        {
                            "feature": "横",
                            "start": [10, 20],
                            "curveList": [{"command": "h", "parameterList": [30]}],
                        },
                        {
                            "feature": "竖",
                            "start": [70, 20],
                            "curveList": [{"command": "v", "parameterList": [30]}],
                        },
                    ]
                },
                "candidateLeafSvgs": {
                    "900001": [
                        {
                            "leafId": 10,
                            "strokeIndices": [0],
                            "hierarchy": [
                                {"id": 10, "label": "末级部件"},
                                {"id": 900001, "label": "⿰"},
                            ],
                        },
                        {
                            "leafId": 11,
                            "strokeIndices": [1],
                            "hierarchy": [
                                {"id": 11, "label": "末级部件"},
                                {"id": 900001, "label": "⿰"},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    graph = compile_catalog_candidate_graph(catalog, 0x64CF, 900001)

    assert graph.glyph_id == 900001
    assert graph.root.operator == "⿰"
    assert [stroke.leaf_id for stroke in graph.strokes] == [10, 11]
    assert [stroke.component_path for stroke in graph.strokes] == [(0,), (1,)]


def test_review_catalog_preserves_recursive_ids_paths():
    catalog = {
        "rows": [
            {
                "unicode": 0x64EC,
                "candidates": {
                    "900021": [
                        {
                            "feature": "横",
                            "start": [5, 10],
                            "curveList": [{"command": "h", "parameterList": [20]}],
                        },
                        {
                            "feature": "竖",
                            "start": [60, 10],
                            "curveList": [{"command": "v", "parameterList": [20]}],
                        },
                        {
                            "feature": "点",
                            "start": [60, 60],
                            "curveList": [{"command": "l", "parameterList": [10, 10]}],
                        },
                    ]
                },
                "candidateLeafSvgs": {
                    "900021": [
                        {
                            "leafId": 220,
                            "strokeIndices": [0],
                            "hierarchy": [
                                {"id": 220, "type": "component", "label": "末级部件"},
                                {"id": 900021, "type": "glyph", "label": "⿰"},
                            ],
                        },
                        {
                            "leafId": 1128,
                            "strokeIndices": [1],
                            "hierarchy": [
                                {"id": 1128, "type": "component", "label": "末级部件"},
                                {"id": 128253, "type": "compound", "label": "⿱"},
                                {"id": 900021, "type": "glyph", "label": "⿰"},
                            ],
                        },
                        {
                            "leafId": 758,
                            "strokeIndices": [2],
                            "hierarchy": [
                                {"id": 758, "type": "component", "label": "末级部件"},
                                {"id": 128253, "type": "compound", "label": "⿱"},
                                {"id": 900021, "type": "glyph", "label": "⿰"},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    graph = compile_catalog_candidate_graph(catalog, 0x64EC, 900021)

    assert graph.root.operator == "⿰"
    assert graph.root.children[1].operator == "⿱"
    assert [stroke.component_path for stroke in graph.strokes] == [
        (0,),
        (1, 0),
        (1, 1),
    ]


def test_review_catalog_distinguishes_repeated_leaf_occurrences():
    catalog = {
        "rows": [
            {
                "unicode": 0x6424,
                "candidates": {
                    "900005": [
                        {
                            "feature": "撇",
                            "start": [30, 20],
                            "curveList": [{"command": "l", "parameterList": [-10, 20]}],
                        },
                        {
                            "feature": "捺",
                            "start": [50, 60],
                            "curveList": [{"command": "l", "parameterList": [10, 20]}],
                        },
                    ]
                },
                "candidateLeafSvgs": {
                    "900005": [
                        {
                            "leafId": 117,
                            "occurrence": 0,
                            "strokeIndices": [0],
                            "hierarchy": [
                                {"id": 117, "label": "末级部件"},
                                {"id": 127531, "label": "⿱"},
                                {"id": 900005, "label": "⿰"},
                            ],
                        },
                        {
                            "leafId": 117,
                            "occurrence": 1,
                            "strokeIndices": [1],
                            "hierarchy": [
                                {"id": 117, "label": "末级部件"},
                                {"id": 127531, "label": "⿱"},
                                {"id": 900005, "label": "⿰"},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    graph = compile_catalog_candidate_graph(catalog, 0x6424, 900005)

    assert [stroke.occurrence for stroke in graph.strokes] == [0, 1]
    assert [stroke.component_path for stroke in graph.strokes] == [(0, 0), (0, 1)]


def test_maximal_chains_receive_one_owner_and_partition_cuts_are_explicit():
    skeleton = _cross()
    topology = compress_skeleton(skeleton)
    assignments = assign_topology_edges(topology, _graph(), 101)

    assert len(assignments) == len(topology.edges)
    assert {item.edge_id for item in assignments} == {
        edge.id for edge in topology.edges
    }
    assert len({item.leaf_key for item in assignments}) == 2
    assert find_partition_cuts(topology, assignments, 101)
