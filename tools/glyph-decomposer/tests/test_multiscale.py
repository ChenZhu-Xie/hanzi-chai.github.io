from shapely.geometry import Point, box

from glyph_decomposer.multiscale import certify_multiscale


def test_simple_geometry_is_stable_across_resolutions():
    geometry = box(10, 45, 90, 55).union(box(45, 10, 55, 90))

    certificate = certify_multiscale(geometry, (96, 128, 192))

    assert certificate.certified is True
    assert certificate.same_topology_signature is True
    assert certificate.all_individually_certified is True
    assert certificate.maximum_p95_distance <= 1.25


def test_thin_hole_that_disappears_across_scales_is_rejected():
    geometry = Point(50, 50).buffer(30).difference(Point(50, 50).buffer(0.3))

    certificate = certify_multiscale(geometry, (32, 64, 128))

    assert certificate.certified is False
    assert certificate.same_topology_signature is False
