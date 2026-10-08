import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from glyph_decomposer.skeleton import (
    evaluate_skeleton,
    generate_skeleton,
    rasterize_geometry,
    zhang_suen,
)


def test_rasterize_geometry_is_deterministic_and_truth_free():
    geometry = unary_union([box(20, 45, 80, 55), box(45, 20, 55, 80)])

    first = rasterize_geometry(geometry, 100)
    second = rasterize_geometry(geometry, 100)

    assert np.array_equal(first, second)
    assert first.sum() == 1100


def test_zhang_suen_preserves_cross_connectivity_and_one_pixel_width():
    binary = np.zeros((31, 31), dtype=bool)
    binary[13:18, 3:28] = True
    binary[3:28, 13:18] = True

    skeleton = zhang_suen(binary)

    assert skeleton[15, 15]
    assert skeleton[:, 15].sum() >= 20
    assert skeleton[15, :].sum() >= 20
    assert skeleton.sum() < binary.sum() / 3


def test_generated_skeleton_matches_independent_centerline_truth():
    geometry = unary_union([box(10, 47, 90, 53), box(47, 10, 53, 90)])
    _ink, skeleton = generate_skeleton(geometry, 200)
    horizontal = np.asarray([[10.0, 50.0], [90.0, 50.0]])
    vertical = np.asarray([[50.0, 10.0], [50.0, 90.0]])

    metrics = evaluate_skeleton(skeleton, (horizontal, vertical), tolerance=1.0)

    assert metrics["truthReadAfterGeneration"] is True
    assert metrics["truthCoverage"] > 0.92
    assert metrics["endpointP95Distance"] < 3.6
