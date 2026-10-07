from shapely.geometry import box

from glyph_decomposer.review import geometry_path_data


def test_vector_review_path_preserves_rings_without_raster_data():
    path = geometry_path_data(box(1, 2, 3, 4).difference(box(1.5, 2.5, 2.5, 3.5)))

    assert path.count("M ") == 2
    assert "1.0000 2.0000" in path
    assert "data:image" not in path
