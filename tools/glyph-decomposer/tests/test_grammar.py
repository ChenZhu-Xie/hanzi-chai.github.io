from glyph_decomposer.grammar import GlyphRepository


def test_compile_recursive_coordinate_free_program():
    repository = GlyphRepository(
        [
            {"id": 1, "type": "component", "strokes": [{"feature": "横"}]},
            {"id": 2, "type": "component", "strokes": [{"feature": "竖"}]},
            {
                "id": 3,
                "type": "compound",
                "operator": "⿰",
                "references": [{"id": 1}, {"id": 2}],
            },
        ]
    )

    program = repository.compile(3)

    assert program.operator == "⿰"
    assert program.leaf_ids() == (1, 2)
    assert program.stroke_count() == 2
    assert program.children[0].stroke_features == ("横",)


def test_compile_rejects_cycles():
    repository = GlyphRepository(
        [
            {
                "id": 3,
                "type": "compound",
                "operator": "⿰",
                "references": [{"id": 3}, {"id": 3}],
            }
        ]
    )

    try:
        repository.compile(3)
    except ValueError as error:
        assert "cyclic glyph references" in str(error)
    else:
        raise AssertionError("cycle was accepted")
