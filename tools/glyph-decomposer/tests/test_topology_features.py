import math

import numpy as np

from glyph_decomposer.topology import compress_skeleton
from glyph_decomposer.topology_features import describe_topology, simplify_path


def test_simplify_path_collapses_redundant_straight_samples():
    points = tuple((float(index), 4.0) for index in range(30))

    simplified = simplify_path(points)

    assert simplified == (points[0], points[-1])


def test_simplify_path_retains_a_decisive_corner():
    points = ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (8.0, 8.0))

    simplified = simplify_path(points, tolerance=0.25)

    assert (8.0, 0.0) in simplified
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]


def test_t_junction_reports_the_straight_through_pair():
    mask = np.zeros((13, 13), dtype=bool)
    mask[2:11, 6] = True
    mask[2, 3:10] = True
    topology = compress_skeleton(mask)

    descriptors = describe_topology(topology, 13)

    junction = next(node for node in descriptors.nodes if node.kind == "junction")
    best = junction.through_pairs[0]
    assert best.straightness > 0.9
    edge_by_id = {edge.id: edge for edge in topology.edges}
    first = edge_by_id[best.first_edge]
    second = edge_by_id[best.second_edge]
    other_nodes = {
        first.end_node if first.start_node == junction.id else first.start_node,
        second.end_node if second.start_node == junction.id else second.start_node,
    }
    positions = [topology.nodes[node_id].centre for node_id in other_nodes]
    assert positions[0][0] != positions[1][0]
    assert math.isclose(positions[0][1], positions[1][1])


def test_descriptors_reduce_points_without_changing_topology():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 3:37] = True
    topology = compress_skeleton(mask)

    descriptors = describe_topology(topology, 40)

    assert len(descriptors.nodes) == len(topology.nodes)
    assert len(descriptors.edges) == len(topology.edges)
    assert descriptors.original_path_points > descriptors.simplified_path_points
    assert descriptors.reduction_ratio > 0.8
