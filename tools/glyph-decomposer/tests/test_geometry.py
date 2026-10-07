import pytest
from shapely.geometry import box

from glyph_decomposer.geometry import normalize_geometry, svg_path_geometry


def test_nonzero_path_keeps_outer_ring_and_removes_hole():
    geometry = svg_path_geometry(
        "M0 0 L10 0 L10 10 L0 10 Z M2 2 L2 8 L8 8 L8 2 Z",
        tolerance=0.2,
    )

    assert geometry.is_valid
    assert geometry.area == pytest.approx(64)
    assert len(geometry.interiors) == 1


def test_normalize_preserves_aspect_ratio_and_area_scale():
    source = box(10, 20, 30, 60)
    normalized = normalize_geometry(source)

    assert normalized.bounds == (26.0, 2.0, 74.0, 98.0)
    assert normalized.area == pytest.approx(source.area * 2.4**2)
