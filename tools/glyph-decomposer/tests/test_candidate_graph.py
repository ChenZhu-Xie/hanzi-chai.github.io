from glyph_decomposer.candidate_graph import (
    compile_candidate_graph,
    compile_catalog_candidate_graph,
)
from glyph_decomposer.grammar import GlyphRepository


def test_compiles_candidate_strokes_into_recursive_ids_graph():
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
                    },
                    {
                        "feature": "竖",
                        "start": [50, 10],
                        "curveList": [{"command": "v", "parameterList": [80]}],
                    },
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
    )

    graph = compile_candidate_graph(repository, 3)

    assert graph.root.operator == "⿰"
    assert graph.root.stroke_indices == (0, 1, 2)
    assert graph.root.children[0].bounds == (0.0, 0.0, 50.0, 100.0)
    assert graph.root.children[1].bounds == (50.0, 0.0, 100.0, 100.0)
    assert [stroke.component_path for stroke in graph.strokes] == [(0,), (0,), (1,)]
    first_contact = next(
        relation
        for relation in graph.relations
        if (relation.first_stroke, relation.second_stroke) == (0, 1)
    )
    assert first_contact.exact_intersection is True
    assert first_contact.same_leaf_occurrence is True


def test_separated_ids_children_remain_non_intersecting():
    repository = GlyphRepository(
        [
            {
                "id": 1,
                "type": "component",
                "strokes": [
                    {
                        "feature": "竖",
                        "start": [20, 10],
                        "curveList": [{"command": "v", "parameterList": [80]}],
                    }
                ],
            },
            {
                "id": 2,
                "type": "compound",
                "operator": "⿰",
                "references": [{"id": 1}, {"id": 1}],
            },
        ]
    )

    graph = compile_candidate_graph(repository, 2)

    assert len(graph.strokes) == 2
    assert graph.relations[0].distance > 0
    assert graph.relations[0].exact_intersection is False
    assert graph.relations[0].same_leaf_occurrence is False


def test_compiles_ternary_top_middle_bottom_ids():
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

    assert graph.root.operator == "⿳"
    assert [stroke.component_path for stroke in graph.strokes] == [
        (0,),
        (1,),
        (2,),
    ]
    assert graph.root.children[0].bounds == (0.0, 0.0, 100.0, 33.0)
    assert graph.root.children[2].bounds == (0.0, 67.0, 100.0, 100.0)


def test_catalog_compiler_applies_confirmed_leaf_order_correction():
    strokes = [
        {
            "feature": feature,
            "start": start,
            "curveList": [{"command": command, "parameterList": [length]}],
        }
        for feature, start, command, length in (
            ("竖", [37, 6], "v", 88),
            ("横", [7, 50], "h", 30),
            ("竖", [63, 6], "v", 88),
            ("横", [63, 50], "h", 30),
        )
    ]
    catalog = {
        "rows": [
            {
                "unicode": 0x64CE,
                "candidates": {"900000": strokes},
                "candidateLeafSvgs": {
                    "900000": [
                        {
                            "leafId": 486,
                            "occurrence": 0,
                            "strokeIndices": [0, 1, 2, 3],
                            "hierarchy": [
                                {"id": 486, "type": "component", "label": "末级部件"},
                                {"id": 900000, "type": "glyph", "label": ""},
                            ],
                        }
                    ]
                },
            }
        ]
    }

    graph = compile_catalog_candidate_graph(catalog, 0x64CE, 900000)

    assert [stroke.feature for stroke in graph.strokes] == ["横", "竖", "竖", "横"]
    assert graph.strokes[0].points[0] == (7.0, 50.0)


def test_catalog_compiler_preserves_upper_right_enclosure_operator():
    catalog = {
        "rows": [
            {
                "unicode": 0x53E5,
                "candidates": {
                    "900100": [
                        {
                            "feature": "撇",
                            "start": [70, 10],
                            "curveList": [{"command": "l", "parameterList": [-30, 70]}],
                        },
                        {
                            "feature": "竖",
                            "start": [40, 45],
                            "curveList": [{"command": "v", "parameterList": [35]}],
                        },
                    ]
                },
                "candidateLeafSvgs": {
                    "900100": [
                        {
                            "leafId": 1,
                            "strokeIndices": [0],
                            "hierarchy": [
                                {"id": 1, "type": "component", "label": ""},
                                {"id": 3, "type": "compound", "label": "⿹"},
                                {"id": 900100, "type": "glyph", "label": ""},
                            ],
                        },
                        {
                            "leafId": 2,
                            "strokeIndices": [1],
                            "hierarchy": [
                                {"id": 2, "type": "component", "label": ""},
                                {"id": 3, "type": "compound", "label": "⿹"},
                                {"id": 900100, "type": "glyph", "label": ""},
                            ],
                        },
                    ]
                },
            }
        ]
    }

    graph = compile_catalog_candidate_graph(catalog, 0x53E5, 900100)

    assert graph.root.children[0].operator == "⿹"
