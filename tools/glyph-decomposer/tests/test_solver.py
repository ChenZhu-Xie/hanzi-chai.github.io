from shapely.geometry import box
from shapely.ops import unary_union

from glyph_decomposer.domain import ComponentProgram
from glyph_decomposer.solver import solve_root_partition, solve_surround_partition


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


def test_root_solver_can_search_inside_a_child_local_extent():
    geometry = unary_union([box(58, 8, 94, 38), box(57, 55, 95, 94)])
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿱",
        children=(component(1, 2), component(2, 3)),
    )

    result = solve_root_partition(
        geometry,
        program,
        minimum=geometry.bounds[1] + 5,
        maximum=geometry.bounds[3] - 5,
    )

    assert 38 <= result.evidence.cut <= 55
    assert result.evidence.crossing_ratio == 0


def test_upper_left_surround_solver_assigns_the_lower_right_quadrant():
    wrapper = unary_union([box(4, 4, 96, 28), box(4, 28, 32, 96)])
    enclosed = box(48, 47, 94, 94)
    geometry = unary_union([wrapper, enclosed])
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿸",
        children=(component(1, 4), component(2, 2)),
    )

    result = solve_surround_partition(geometry, program)

    assert result.evidence.axis == "xy"
    assert result.evidence.secondary_cut is not None
    assert result.children[0].equals(wrapper)
    assert result.children[1].equals(enclosed)
    assert min(child.area for child in result.children) / geometry.area >= 0.10
    assert result.evidence.reconstruction_error == 0


def test_upper_left_surround_solver_rejects_a_boundary_through_solid_ink():
    program = ComponentProgram(
        glyphId=3,
        kind="compound",
        operator="⿸",
        children=(component(1, 4), component(2, 2)),
    )

    try:
        solve_surround_partition(box(2, 2, 98, 98), program)
    except ValueError as error:
        assert "stroke-level assignment is required" in str(error)
    else:
        raise AssertionError("a high-crossing surround partition was accepted")
