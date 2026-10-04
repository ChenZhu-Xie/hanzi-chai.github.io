"""Directed half-edge view of a raster glyph skeleton.

The ordinary pixel graph is useful for measuring distance, but it does not
represent the decision made at a junction: arriving on one road must not grant
permission to continue along every touching road.  This module collapses each
junction cluster into a gate and exposes every road in both directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


NEIGHBOURS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)

SECTORS = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")


@dataclass(frozen=True)
class DirectedHalfEdge:
    edge_id: int
    road_id: int
    start_gate: int
    end_gate: int
    point_indices: tuple[int, ...]
    start_sector: str
    end_sector: str


@dataclass(frozen=True)
class JunctionGate:
    gate_id: int
    point_indices: frozenset[int]
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]


@dataclass(frozen=True)
class DirectedSkeleton:
    edges: tuple[DirectedHalfEdge, ...]
    gates: tuple[JunctionGate, ...]
    pixel_to_edges: dict[int, tuple[int, ...]]


def _sector(vector: np.ndarray) -> str:
    dy, dx = (float(value) for value in vector)
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    return SECTORS[int((angle + 22.5) // 45.0) % 8]


def _sector_vector(sector: str) -> np.ndarray:
    angle = math.radians(SECTORS.index(sector) * 45.0)
    return np.asarray([math.sin(angle), math.cos(angle)], dtype=float)


def _turn(incoming_sector: str, outgoing_sector: str) -> str:
    incoming = _sector_vector(incoming_sector)
    outgoing = _sector_vector(outgoing_sector)
    signed = math.degrees(
        math.atan2(
            float(incoming[1] * outgoing[0] - incoming[0] * outgoing[1]),
            float(np.dot(incoming, outgoing)),
        )
    )
    if abs(signed) <= 28.0:
        return "straight"
    if abs(signed) >= 152.0:
        return "reverse"
    return "clockwise" if signed > 0 else "counterclockwise"


def _pixel_adjacency(graph) -> list[list[int]]:
    """Return 8-neighbour adjacency without corner-cutting shortcuts.

    A raster plus sign has diagonal neighbours around its centre under a plain
    8-neighbour graph.  Those diagonals are sampling artefacts, not roads that
    let a pen bypass the actual junction.  Keep a diagonal only when neither
    orthogonal intermediate pixel exists.
    """

    adjacency = [[] for _point in graph.points]
    for index, (y, x) in enumerate(graph.points):
        for dy, dx in NEIGHBOURS:
            other = graph.point_index.get((int(y + dy), int(x + dx)))
            if other is None or other <= index:
                continue
            if dy and dx:
                horizontal = graph.point_index.get((int(y), int(x + dx)))
                vertical = graph.point_index.get((int(y + dy), int(x)))
                if horizontal is not None or vertical is not None:
                    continue
            adjacency[index].append(other)
            adjacency[other].append(index)
    return adjacency


def _critical_clusters(adjacency: list[list[int]]) -> list[frozenset[int]]:
    critical = {index for index, neighbours in enumerate(adjacency) if len(neighbours) != 2}
    if not critical and adjacency:
        critical.add(0)
    clusters = []
    unseen = set(critical)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        cluster = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    cluster.add(neighbour)
                    queue.append(neighbour)
        clusters.append(frozenset(cluster))
    return clusters


def build_directed_skeleton(graph) -> DirectedSkeleton:
    adjacency = _pixel_adjacency(graph)
    clusters = _critical_clusters(adjacency)
    point_to_gate = {
        point_index: gate_id
        for gate_id, members in enumerate(clusters)
        for point_index in members
    }
    visited: set[tuple[int, int]] = set()
    roads: list[tuple[int, int, tuple[int, ...]]] = []

    def edge_key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left < right else (right, left)

    for start_gate, members in enumerate(clusters):
        for start in sorted(members):
            for neighbour in adjacency[start]:
                if point_to_gate.get(neighbour) == start_gate:
                    continue
                key = edge_key(start, neighbour)
                if key in visited:
                    continue
                visited.add(key)
                path = [start, neighbour]
                previous, current = start, neighbour
                while current not in point_to_gate:
                    choices = [item for item in adjacency[current] if item != previous]
                    if len(choices) != 1:
                        raise ValueError("non-gate skeleton branch encountered while tracing road")
                    following = choices[0]
                    visited.add(edge_key(current, following))
                    path.append(following)
                    previous, current = current, following
                end_gate = point_to_gate[current]
                roads.append((start_gate, end_gate, tuple(path)))

    edges: list[DirectedHalfEdge] = []
    for road_id, (start_gate, end_gate, path) in enumerate(roads):
        forward_points = graph.points[np.asarray(path)]
        reverse_path = tuple(reversed(path))
        reverse_points = forward_points[::-1]
        edges.append(
            DirectedHalfEdge(
                edge_id=len(edges),
                road_id=road_id,
                start_gate=start_gate,
                end_gate=end_gate,
                point_indices=path,
                start_sector=_sector(forward_points[1] - forward_points[0]),
                end_sector=_sector(forward_points[-1] - forward_points[-2]),
            )
        )
        edges.append(
            DirectedHalfEdge(
                edge_id=len(edges),
                road_id=road_id,
                start_gate=end_gate,
                end_gate=start_gate,
                point_indices=reverse_path,
                start_sector=_sector(reverse_points[1] - reverse_points[0]),
                end_sector=_sector(reverse_points[-1] - reverse_points[-2]),
            )
        )

    incoming: list[list[int]] = [[] for _cluster in clusters]
    outgoing: list[list[int]] = [[] for _cluster in clusters]
    pixel_to_edges: dict[int, list[int]] = {}
    for edge in edges:
        outgoing[edge.start_gate].append(edge.edge_id)
        incoming[edge.end_gate].append(edge.edge_id)
        for point_index in edge.point_indices:
            pixel_to_edges.setdefault(point_index, []).append(edge.edge_id)

    gates = tuple(
        JunctionGate(
            gate_id=gate_id,
            point_indices=members,
            incoming=tuple(sorted(incoming[gate_id])),
            outgoing=tuple(sorted(outgoing[gate_id])),
        )
        for gate_id, members in enumerate(clusters)
    )
    return DirectedSkeleton(
        edges=tuple(edges),
        gates=gates,
        pixel_to_edges={key: tuple(sorted(value)) for key, value in pixel_to_edges.items()},
    )


def trace_route_edges(directed: DirectedSkeleton, graph, route_points: np.ndarray) -> tuple[int, ...]:
    if not len(route_points):
        return ()
    target = np.asarray(route_points, dtype=float)
    graph_xy = graph.points[:, ::-1].astype(float)
    nearest = np.argmin(np.linalg.norm(target[:, None, :] - graph_xy[None, :, :], axis=2), axis=1)
    compressed = [int(nearest[0])]
    for point_index in nearest[1:]:
        if int(point_index) != compressed[-1]:
            compressed.append(int(point_index))
    pair_to_edge: dict[tuple[int, int], int] = {}
    for edge in directed.edges:
        for left, right in zip(edge.point_indices, edge.point_indices[1:]):
            pair_to_edge[left, right] = edge.edge_id
    output = []
    for pair in zip(compressed, compressed[1:]):
        edge_id = pair_to_edge.get(pair)
        if edge_id is not None and (not output or output[-1] != edge_id):
            output.append(edge_id)
    return tuple(output)


def legal_exit_edges(
    directed: DirectedSkeleton,
    incoming_edge_id: int,
    expected_sector: str,
    expected_turns: tuple[str, ...],
) -> tuple[int, ...]:
    incoming = directed.edges[incoming_edge_id]
    gate = directed.gates[incoming.end_gate]
    expected_turn = expected_turns[0] if expected_turns else None
    output = []
    for edge_id in gate.outgoing:
        edge = directed.edges[edge_id]
        if edge.road_id == incoming.road_id:
            continue
        if expected_sector and edge.start_sector != expected_sector:
            continue
        if expected_turn and _turn(incoming.end_sector, edge.start_sector) != expected_turn:
            continue
        output.append(edge_id)
    return tuple(output)
