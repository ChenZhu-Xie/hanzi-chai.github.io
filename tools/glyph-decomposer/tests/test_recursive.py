from shapely.geometry import box
from shapely.ops import unary_union

from glyph_decomposer.domain import ComponentProgram
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.recursive import decompose_recursive, terminal_nodes


def component(glyph_id: int, strokes: int) -> ComponentProgram:
    return ComponentProgram(
        glyphId=glyph_id,
        kind="component",
        strokeFeatures=tuple("横" for _ in range(strokes)),
    )


def test_recursive_partition_stops_honestly_at_unsupported_operator():
    unsupported = ComponentProgram(
        glyphId=4,
        kind="compound",
        operator="⿻",
        children=(component(5, 2), component(6, 2)),
    )
    right = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿱",
        children=(unsupported, component(7, 3)),
    )
    root = ComponentProgram(
        glyphId=1,
        kind="compound",
        operator="⿰",
        children=(component(2, 3), right),
    )
    geometry = unary_union(
        [
            box(3, 8, 28, 92),
            box(48, 5, 96, 40),
            box(49, 61, 95, 95),
        ]
    )

    result = decompose_recursive(geometry, root)
    terminals = terminal_nodes(result)

    assert result.evidence.status == "partitioned"
    assert result.children[1].evidence.status == "partitioned"
    assert [node.program.glyph_id for node in terminals] == [2, 4, 7]
    assert terminals[1].evidence.status == "unsupported-operator"
    assert unary_union([node.geometry for node in terminals]).equals(geometry)
    assert sum(node.geometry.area for node in terminals) == geometry.area


def test_recursive_partition_descends_through_upper_left_surround():
    wrapper = unary_union([box(4, 4, 96, 28), box(4, 28, 32, 96)])
    enclosed = box(48, 47, 94, 94)
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿸",
        children=(component(1, 4), component(2, 2)),
    )

    result = decompose_recursive(unary_union([wrapper, enclosed]), program)

    assert [node.program.glyph_id for node in terminal_nodes(result)] == [1, 2]
    assert result.evidence.status == "partitioned"


def test_crossing_axis_partition_uses_stroke_identity_when_repository_is_available():
    records = [
        {
            "id": 1,
            "type": "component",
            "strokes": [
                {
                    "feature": "竖",
                    "start": [35, 5],
                    "curveList": [{"command": "v", "parameterList": [90]}],
                }
            ],
        },
        {
            "id": 2,
            "type": "component",
            "strokes": [
                {
                    "feature": "竖",
                    "start": [65, 5],
                    "curveList": [{"command": "v", "parameterList": [90]}],
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
    repository = GlyphRepository(records)
    program = repository.compile(3)
    geometry = unary_union([box(24, 4, 46, 96), box(44, 4, 76, 96)])

    result = decompose_recursive(geometry, program, repository)

    assert result.evidence.partition.solver_status == "VECTOR_VORONOI"
    assert result.evidence.partition.stroke_continuity >= 0.95


def test_clean_axis_partition_keeps_exact_enumeration_with_repository():
    records = [
        {
            "id": 1,
            "type": "component",
            "strokes": [
                {
                    "feature": "竖",
                    "start": [25, 5],
                    "curveList": [{"command": "v", "parameterList": [90]}],
                }
            ],
        },
        {
            "id": 2,
            "type": "component",
            "strokes": [
                {
                    "feature": "竖",
                    "start": [75, 5],
                    "curveList": [{"command": "v", "parameterList": [90]}],
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
    repository = GlyphRepository(records)
    program = repository.compile(3)
    geometry = unary_union([box(15, 4, 35, 96), box(65, 4, 85, 96)])

    result = decompose_recursive(geometry, program, repository)

    assert result.evidence.partition.solver_status == "EXACT_ENUMERATION"
    assert result.evidence.partition.crossing_ratio == 0
    assert result.evidence.partition.stroke_count is None
