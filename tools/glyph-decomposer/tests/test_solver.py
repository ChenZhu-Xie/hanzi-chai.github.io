from shapely.geometry import box
from shapely.ops import unary_union

from glyph_decomposer.domain import ComponentProgram
from glyph_decomposer.solver import solve_root_partition


def component(glyph_id: int, strokes: int) -> ComponentProgram:
    return ComponentProgram(
        glyphId=glyph_id,
        kind="component",
        strokeFeatures=tuple("横" for _ in range(strokes)),
    )


def test_root_solver_chooses_gap_and_reconstructs_all_ink():
    geometry = unary_union([box(3, 10, 39, 90), box(55, 5, 97, 95)])
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿰",
        children=(component(1, 3), component(2, 6)),
    )

    result = solve_root_partition(geometry, program)

    assert 39 <= result.evidence.cut <= 55
    assert result.evidence.solver_status == "EXACT_ENUMERATION"
    assert result.evidence.crossing_ratio == 0
    assert result.evidence.reconstruction_error == 0
    assert result.children[0].union(result.children[1]).equals(geometry)


def test_root_solver_rejects_unsupported_operator():
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿴",
        children=(component(1, 1), component(2, 1)),
    )

    try:
        solve_root_partition(box(0, 0, 100, 100), program)
    except ValueError as error:
        assert "supports ⿰ and ⿱" in str(error)
    else:
        raise AssertionError("unsupported operator was accepted")
