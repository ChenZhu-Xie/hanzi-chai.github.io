import numpy as np

from glyph_decomposer.primitive_fit import fit_stroke_primitives


def _cubic(controls, count=17):
    p0, p1, p2, p3 = np.asarray(controls, dtype=float)
    t = np.linspace(0, 1, count)
    u = 1 - t
    return (
        u[:, None] ** 3 * p0
        + 3 * u[:, None] ** 2 * t[:, None] * p1
        + 3 * u[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )


def test_horizontal_uses_one_line_and_two_points():
    route = [(10, 20), (30, 20), (60, 20), (90, 20)]

    fit = fit_stroke_primitives(route, ("h",), [(0, 0), (100, 0)])

    assert [item.kind for item in fit.primitives] == ["line"]
    assert fit.control_point_count == 2
    assert fit.rmse < 1e-9


def test_cubic_uses_exactly_four_control_points():
    route = _cubic(((5, 5), (10, 60), (70, 90), (95, 20)))

    fit = fit_stroke_primitives(route, ("c",), route)

    assert [item.kind for item in fit.primitives] == ["cubic"]
    assert len(fit.primitives[0].controls) == 4
    assert fit.control_point_count == 4
    assert fit.rmse < 0.6


def test_fold_has_one_connected_polyline_with_three_points():
    route = [(10, 10), (50, 10), (90, 10), (90, 45), (90, 90)]
    expected = [(0, 0), (80, 0), (80, 80)]

    fit = fit_stroke_primitives(route, ("h", "v"), expected)

    assert [item.kind for item in fit.primitives] == ["line", "line"]
    assert fit.primitives[0].controls[-1] == fit.primitives[1].controls[0]
    assert fit.control_point_count == 3
    assert fit.rmse < 1e-9


def test_two_cubics_use_seven_unique_control_points():
    first = _cubic(((5, 5), (20, 0), (30, 45), (50, 50)))
    second = _cubic(((50, 50), (65, 55), (75, 100), (95, 90)))
    route = np.vstack((first, second[1:]))

    fit = fit_stroke_primitives(route, ("c", "c"), route)

    assert [item.kind for item in fit.primitives] == ["cubic", "cubic"]
    assert fit.primitives[0].controls[-1] == fit.primitives[1].controls[0]
    assert fit.control_point_count == 7
    assert fit.rmse < 0.5


def test_quadratic_vertical_sweep_uses_sunken_control_point():
    # Vertical sweep route: drops straight down from (17.6, 15.5) then sweeps to (9.0, 87.5)
    t = np.linspace(0, 1, 17)
    p0 = np.array([17.6, 15.5])
    p1 = np.array([17.6, 78.0])
    p2 = np.array([9.0, 87.5])
    route = (1 - t[:, None]) ** 2 * p0 + 2 * (1 - t[:, None]) * t[:, None] * p1 + t[:, None] ** 2 * p2

    fit = fit_stroke_primitives(route, ("q",), route)

    assert [item.kind for item in fit.primitives] == ["quadratic"]
    assert fit.control_point_count == 3
    ctrls = fit.primitives[0].controls
    assert len(ctrls) == 3
    # P1 horizontal locked to P0 horizontal
    assert abs(ctrls[1][0] - ctrls[0][0]) < 1e-3
    # P1 vertical sunken deeply
    assert ctrls[1][1] > 60.0
    assert fit.rmse < 0.6


def test_endpoint_spur_stripping_projects_to_true_trunk():
    # Route with a lateral spur at the start perpendicular to the southwest trunk
    trunk_t = np.linspace(0, 1, 15)
    trunk_x = 50.0 - 12.5 * trunk_t
    trunk_y = 15.0 + 11.2 * trunk_t
    trunk = np.column_stack((trunk_x, trunk_y))
    # Lateral spur at index 0 sticking out perpendicular to trunk direction
    spur_p0 = trunk[0] + 3.0 * np.array([0.67, 0.74])
    route = np.vstack(([spur_p0], trunk))

    fit = fit_stroke_primitives(route, ("h",), [(0, 0), (100, 0)])
    p0 = fit.primitives[0].controls[0]
    # The lateral spur should be stripped, projecting P0 back near trunk[0]
    assert np.linalg.norm(np.asarray(p0) - trunk[0]) < 0.5
