"""Compact, truth-free descriptors for recursive skeleton reasoning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np

from .topology import SkeletonEdge, SkeletonTopology


@dataclass(frozen=True)
class ThroughPair:
    first_edge: int
    second_edge: int
    angular_deviation: float
    straightness: float


@dataclass(frozen=True)
class NodeDescriptor:
    id: int
    kind: str
    degree: int
    position: tuple[float, float]
    outward_angles: tuple[tuple[int, float], ...]
    through_pairs: tuple[ThroughPair, ...]


@dataclass(frozen=True)
class EdgeDescriptor:
    id: int
    start_node: int
    end_node: int
    simplified_path: tuple[tuple[float, float], ...]
    original_point_count: int
    simplified_point_count: int
    reduction_ratio: float
    normalized_length: float
    chord_ratio: float
    total_turn: float
    start_tangent: float
    end_tangent: float
    bounds: tuple[float, float, float, float]
    endpoint_count: int
    terminal_shortness: float


@dataclass(frozen=True)
class TopologyDescriptors:
    nodes: tuple[NodeDescriptor, ...]
    edges: tuple[EdgeDescriptor, ...]
    original_path_points: int
    simplified_path_points: int

    @property
    def reduction_ratio(self) -> float:
        if not self.original_path_points:
            return 1.0
        return 1.0 - self.simplified_path_points / self.original_path_points


def _point_segment_distance(point, start, end) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator == 0:
        return float(np.linalg.norm(point - start))
    position = float(np.clip(np.dot(point - start, delta) / denominator, 0, 1))
    return float(np.linalg.norm(point - (start + position * delta)))


def simplify_path(
    points: tuple[tuple[float, float], ...], tolerance: float = 0.75
) -> tuple[tuple[float, float], ...]:
    """Deterministic Ramer-Douglas-Peucker simplification."""
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    distances = [_point_segment_distance(point, start, end) for point in points[1:-1]]
    maximum = max(distances, default=0.0)
    if maximum <= tolerance:
        return (start, end)
    split = 1 + distances.index(maximum)
    left = simplify_path(points[: split + 1], tolerance)
    right = simplify_path(points[split:], tolerance)
    return left[:-1] + right


def _angle(vector) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _angular_distance(first: float, second: float) -> float:
    difference = abs(first - second) % (2 * math.pi)
    return min(difference, 2 * math.pi - difference)


def _edge_outward_angle(edge: SkeletonEdge, node_id: int) -> float:
    points = np.asarray(edge.path, dtype=float)
    if len(points) < 2:
        return 0.0
    if edge.start_node == node_id:
        vector = points[min(2, len(points) - 1)] - points[0]
    elif edge.end_node == node_id:
        vector = points[max(0, len(points) - 3)] - points[-1]
    else:
        raise ValueError(f"edge {edge.id} is not incident to node {node_id}")
    return _angle(vector)


def _edge_descriptor(
    edge: SkeletonEdge,
    topology: SkeletonTopology,
    canvas_size: int,
    tolerance: float,
) -> EdgeDescriptor:
    simplified = simplify_path(edge.path, tolerance)
    chord = float(np.linalg.norm(np.asarray(edge.path[-1]) - np.asarray(edge.path[0])))
    directions = [
        _angle(np.asarray(end) - np.asarray(start))
        for start, end in pairwise(simplified)
        if start != end
    ]
    turn = sum(
        _angular_distance(first, second) for first, second in pairwise(directions)
    )
    coordinates = np.asarray(edge.path, dtype=float)
    minimum = coordinates.min(axis=0) / canvas_size
    maximum = coordinates.max(axis=0) / canvas_size
    endpoint_count = sum(
        topology.nodes[node_id].kind == "endpoint"
        for node_id in {edge.start_node, edge.end_node}
    )
    normalized_length = edge.length / canvas_size
    # Continuous evidence only. A short terminal edge may still be a real dot,
    # hook or rising stroke and must not be pruned without candidate support.
    terminal_shortness = endpoint_count * math.exp(-normalized_length / 0.035) / 2
    return EdgeDescriptor(
        id=edge.id,
        start_node=edge.start_node,
        end_node=edge.end_node,
        simplified_path=simplified,
        original_point_count=len(edge.path),
        simplified_point_count=len(simplified),
        reduction_ratio=1 - len(simplified) / len(edge.path),
        normalized_length=normalized_length,
        chord_ratio=chord / edge.length if edge.length else 1.0,
        total_turn=turn,
        start_tangent=directions[0] if directions else 0.0,
        end_tangent=directions[-1] if directions else 0.0,
        bounds=(
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ),
        endpoint_count=endpoint_count,
        terminal_shortness=terminal_shortness,
    )


def describe_topology(
    topology: SkeletonTopology,
    canvas_size: int,
    *,
    tolerance: float = 0.75,
) -> TopologyDescriptors:
    """Describe shape and junction alternatives without reading candidates/truth."""
    edge_by_id = {edge.id: edge for edge in topology.edges}
    edges = tuple(
        _edge_descriptor(edge, topology, canvas_size, tolerance)
        for edge in topology.edges
    )
    nodes = []
    for node in topology.nodes:
        angles = tuple(
            sorted(
                (
                    (edge_id, _edge_outward_angle(edge_by_id[edge_id], node.id))
                    for edge_id in node.incident_edges
                ),
                key=lambda item: item[1],
            )
        )
        pairs = []
        for (first_id, first_angle), (second_id, second_angle) in combinations(
            angles, 2
        ):
            deviation = abs(math.pi - _angular_distance(first_angle, second_angle))
            pairs.append(
                ThroughPair(
                    first_edge=first_id,
                    second_edge=second_id,
                    angular_deviation=deviation,
                    straightness=max(0.0, 1.0 - deviation / math.pi),
                )
            )
        pairs.sort(
            key=lambda pair: (-pair.straightness, pair.first_edge, pair.second_edge)
        )
        nodes.append(
            NodeDescriptor(
                id=node.id,
                kind=node.kind,
                degree=node.degree,
                position=(
                    node.centre[0] / canvas_size,
                    node.centre[1] / canvas_size,
                ),
                outward_angles=angles,
                through_pairs=tuple(pairs),
            )
        )
    return TopologyDescriptors(
        nodes=tuple(nodes),
        edges=edges,
        original_path_points=sum(len(edge.path) for edge in topology.edges),
        simplified_path_points=sum(len(edge.simplified_path) for edge in edges),
    )
