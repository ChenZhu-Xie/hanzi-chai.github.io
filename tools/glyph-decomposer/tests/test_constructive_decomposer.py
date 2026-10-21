import numpy as np
import pytest
from glyph_decomposer.constructive_decomposer import (
    fit_stroke_primitive_strict,
    evaluate_glyph_primitives,
    ConstructivePrimitive,
)


def test_fit_stroke_primitive_line():
    route = np.array([[10.0, 20.0], [20.0, 20.0], [30.0, 20.0], [40.0, 20.0]])
    prim = fit_stroke_primitive_strict(0, "横", "横", ("h",), (0,), route)
    assert prim.tool == "line"
    assert prim.point_count == 2
    assert len(prim.controls) == 2
    assert prim.controls[0] == (10.0, 20.0)
    assert prim.controls[1] == (40.0, 20.0)


def test_fit_stroke_primitive_bezier():
    # Arc-like route
    t = np.linspace(0, 1, 20)
    xs = 50.0 - 20.0 * (t ** 2)
    ys = 15.0 + 70.0 * t
    route = np.column_stack([xs, ys])
    prim = fit_stroke_primitive_strict(0, "竖撇", "竖撇", ("q",), (0,), route)
    assert prim.tool == "bezier"
    assert prim.point_count == 3
    assert len(prim.controls) == 3
    # Check Rule 1 vertical alignment of P1
    assert abs(prim.controls[1][0] - prim.controls[0][0]) < 1e-4


def test_fit_stroke_primitive_simple_turn():
    # Horizontal then vertical
    h_part = np.column_stack([np.linspace(10, 40, 10), np.full(10, 20.0)])
    v_part = np.column_stack([np.full(10, 40.0), np.linspace(20, 60, 10)])
    route = np.vstack([h_part, v_part[1:]])
    prim = fit_stroke_primitive_strict(0, "横折", "横折", ("h", "v"), (0,), route)
    assert prim.tool == "polyline"
    assert prim.point_count == 3
    assert len(prim.controls) == 3


def test_fit_stroke_primitive_compound_hook():
    # Horizontal, vertical down, hook up/left
    h_part = np.column_stack([np.linspace(10, 40, 10), np.full(10, 20.0)])
    v_part = np.column_stack([np.full(15, 40.0), np.linspace(20, 80, 15)])
    hook_part = np.column_stack([np.linspace(40, 25, 6), np.linspace(80, 75, 6)])
    route = np.vstack([h_part, v_part[1:], hook_part[1:]])
    prim = fit_stroke_primitive_strict(0, "横折钩", "横折钩", ("h", "v", "l"), (0,), route)
    assert prim.tool == "polyline"
    assert prim.point_count == 4
    assert len(prim.controls) == 4
