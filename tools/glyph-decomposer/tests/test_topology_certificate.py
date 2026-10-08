import numpy as np

from glyph_decomposer.skeleton import zhang_suen
from glyph_decomposer.topology_certificate import certify_skeleton


def test_thinning_preserves_components_and_holes():
    ink = np.zeros((40, 40), dtype=bool)
    ink[4:17, 4:17] = True
    ink[8:13, 8:13] = False
    ink[24:35, 20:35] = True

    skeleton = zhang_suen(ink)
    certificate = certify_skeleton(ink, skeleton)

    assert certificate.certified is True
    assert certificate.ink_components == certificate.skeleton_components == 2
    assert certificate.ink_holes == certificate.skeleton_holes == 1
    assert certificate.ink_euler == certificate.skeleton_euler == 1


def test_certificate_rejects_a_missing_component():
    ink = np.zeros((20, 20), dtype=bool)
    ink[2:7, 2:7] = True
    ink[12:17, 12:17] = True
    skeleton = np.zeros_like(ink)
    skeleton[4, 2:7] = True

    certificate = certify_skeleton(ink, skeleton)

    assert certificate.certified is False
    assert certificate.every_ink_component_represented is False
    assert certificate.ink_components == 2
    assert certificate.skeleton_components == 1


def test_certificate_rejects_skeleton_pixels_outside_ink():
    ink = np.zeros((20, 20), dtype=bool)
    ink[5:15, 5:15] = True
    skeleton = zhang_suen(ink)
    skeleton[0, 0] = True

    certificate = certify_skeleton(ink, skeleton)

    assert certificate.certified is False
    assert certificate.skeleton_inside_ink is False
