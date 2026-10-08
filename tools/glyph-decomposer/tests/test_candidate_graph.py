from glyph_decomposer.candidate_graph import compile_candidate_graph
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
