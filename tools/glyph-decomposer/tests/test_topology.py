import numpy as np

from glyph_decomposer.topology import compress_skeleton


def _assert_conserved(mask, topology):
    node_pixels = {pixel for node in topology.nodes for pixel in node.pixels}
    edge_pixels = {pixel for edge in topology.edges for pixel in edge.pixels}
    assert node_pixels.isdisjoint(edge_pixels)
    assert node_pixels | edge_pixels == {tuple(point) for point in np.argwhere(mask)}
    assert topology.skeleton_pixel_count == int(mask.sum())


def test_compresses_straight_line_to_one_chain():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 2:9] = True

    topology = compress_skeleton(mask)

    assert len(topology.nodes) == 2
    assert len(topology.edges) == 1
    assert topology.endpoint_count == 2
    assert topology.junction_count == 0
    _assert_conserved(mask, topology)


def test_compresses_t_junction_without_losing_pixels():
    mask = np.zeros((13, 13), dtype=bool)
    mask[2:11, 6] = True
    mask[2, 3:10] = True

    topology = compress_skeleton(mask)

    assert topology.endpoint_count == 3
    assert topology.junction_count == 1
    assert len(topology.edges) == 3
    _assert_conserved(mask, topology)


def test_compresses_x_junction_into_four_chains():
    mask = np.zeros((13, 13), dtype=bool)
    for index in range(2, 11):
        mask[index, index] = True
        mask[index, 12 - index] = True

    topology = compress_skeleton(mask)

    assert topology.endpoint_count == 4
    assert topology.junction_count == 1
    assert len(topology.edges) == 4
    _assert_conserved(mask, topology)


def test_pure_cycle_gets_a_deterministic_anchor():
    mask = np.zeros((12, 12), dtype=bool)
    points = [(3, 5), (4, 4), (5, 3), (6, 4), (7, 5), (6, 6), (5, 7), (4, 6)]
    for point in points:
        mask[point] = True

    topology = compress_skeleton(mask)

    assert len(topology.nodes) == 1
    assert len(topology.edges) == 1
    assert topology.edges[0].start_node == topology.edges[0].end_node == 0
    _assert_conserved(mask, topology)
