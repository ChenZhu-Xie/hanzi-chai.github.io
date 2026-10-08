"""Lossless compression of a pixel skeleton into nodes and maximal chains."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

_OFFSETS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
)


@dataclass(frozen=True)
class SkeletonNode:
    id: int
    pixels: tuple[tuple[int, int], ...]
    centre: tuple[float, float]
    incident_edges: tuple[int, ...]

    @property
    def degree(self) -> int:
        return len(self.incident_edges)

    @property
    def kind(self) -> str:
        if self.degree == 1:
            return "endpoint"
        if self.degree >= 3:
            return "junction"
        return "anchor"


@dataclass(frozen=True)
class SkeletonEdge:
    id: int
    start_node: int
    end_node: int
    pixels: tuple[tuple[int, int], ...]
    path: tuple[tuple[float, float], ...]
    length: float


@dataclass(frozen=True)
class SkeletonTopology:
    nodes: tuple[SkeletonNode, ...]
    edges: tuple[SkeletonEdge, ...]
    skeleton_pixel_count: int

    @property
    def endpoint_count(self) -> int:
        return sum(node.kind == "endpoint" for node in self.nodes)

    @property
    def junction_count(self) -> int:
        return sum(node.kind == "junction" for node in self.nodes)


def _neighbours(point: tuple[int, int], shape: tuple[int, int]):
    y, x = point
    height, width = shape
    for dy, dx in _OFFSETS:
        neighbour = (y + dy, x + dx)
        if 0 <= neighbour[0] < height and 0 <= neighbour[1] < width:
            yield neighbour


def _components(points: set[tuple[int, int]], shape: tuple[int, int]):
    remaining = set(points)
    output = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = [seed]
        queue = deque([seed])
        while queue:
            point = queue.popleft()
            for neighbour in _neighbours(point, shape):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.append(neighbour)
                    queue.append(neighbour)
        output.append(tuple(sorted(component)))
    return output


def _order_chain(
    pixels: tuple[tuple[int, int], ...],
    node_by_pixel: dict[tuple[int, int], int],
    start_node: int,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    available = set(pixels)
    start_candidates = [
        point
        for point in pixels
        if any(
            node_by_pixel.get(neighbour) == start_node
            for neighbour in _neighbours(point, shape)
        )
    ]
    current = min(start_candidates or pixels)
    ordered = [current]
    available.remove(current)
    previous = None
    while available:
        candidates = [
            point for point in _neighbours(current, shape) if point in available
        ]
        if not candidates:
            break
        if previous is None or len(candidates) == 1:
            following = min(candidates)
        else:
            incoming = np.asarray(current, dtype=float) - np.asarray(
                previous, dtype=float
            )
            following = max(
                candidates,
                key=lambda point: float(
                    np.dot(np.asarray(point, dtype=float) - current, incoming)
                ),
            )
        previous, current = current, following
        available.remove(current)
        ordered.append(current)
    if available:
        raise ValueError("degree-two skeleton component is not a single chain")
    return tuple(ordered)


def _path_length(points: tuple[tuple[float, float], ...]) -> float:
    return float(
        sum(
            np.linalg.norm(np.asarray(end) - np.asarray(start))
            for start, end in pairwise(points)
        )
    )


def compress_skeleton(skeleton: np.ndarray) -> SkeletonTopology:
    """Collapse degree-two pixels while preserving every skeleton pixel once."""
    mask = np.asarray(skeleton, dtype=bool)
    skeleton_pixels = {tuple(point) for point in np.argwhere(mask)}
    if not skeleton_pixels:
        raise ValueError("cannot compress an empty skeleton")
    shape = mask.shape
    degree = {
        point: sum(
            neighbour in skeleton_pixels for neighbour in _neighbours(point, shape)
        )
        for point in skeleton_pixels
    }
    critical = {point for point, value in degree.items() if value != 2}
    node_components = _components(critical, shape) if critical else []
    node_by_pixel = {
        pixel: node_id
        for node_id, component in enumerate(node_components)
        for pixel in component
    }
    chain_components = _components(skeleton_pixels - critical, shape)

    provisional_nodes = list(node_components)
    edge_records = []
    for component in chain_components:
        adjacent_nodes = sorted(
            {
                node_by_pixel[neighbour]
                for point in component
                for neighbour in _neighbours(point, shape)
                if neighbour in node_by_pixel
            }
        )
        if not adjacent_nodes:
            # A pure cycle has no degree != 2 pixel. Introduce one deterministic
            # zero-area anchor; no skeleton pixel is removed from the chain.
            anchor = min(component)
            node_id = len(provisional_nodes)
            provisional_nodes.append(())
            start_node = end_node = node_id
            ordered = _order_cycle(component, anchor, shape)
        elif len(adjacent_nodes) == 1:
            start_node = end_node = adjacent_nodes[0]
            ordered = _order_chain(component, node_by_pixel, start_node, shape)
        elif len(adjacent_nodes) == 2:
            start_node, end_node = adjacent_nodes
            ordered = _order_chain(component, node_by_pixel, start_node, shape)
        else:
            raise ValueError("degree-two chain touches more than two node clusters")
        edge_records.append((start_node, end_node, ordered))

    # Rarely, two critical clusters touch directly without an intervening
    # degree-two pixel. Record that adjacency as a zero-pixel topological edge.
    direct_pairs = set()
    for pixel, node_id in node_by_pixel.items():
        for neighbour in _neighbours(pixel, shape):
            other = node_by_pixel.get(neighbour)
            if other is not None and other != node_id:
                direct_pairs.add(tuple(sorted((node_id, other))))
    for start_node, end_node in sorted(direct_pairs):
        edge_records.append((start_node, end_node, ()))

    centres = []
    for node_id, component in enumerate(provisional_nodes):
        if component:
            values = np.asarray(component, dtype=float)
            centres.append(tuple(values.mean(axis=0)))
        else:
            cycle = next(record[2] for record in edge_records if record[0] == node_id)
            centres.append(tuple(map(float, cycle[0])))

    incident: list[list[int]] = [[] for _ in provisional_nodes]
    edges = []
    for edge_id, (start_node, end_node, pixels) in enumerate(edge_records):
        start = centres[start_node]
        end = centres[end_node]
        path_yx = (start, *pixels, end)
        path_xy = tuple((float(x), float(y)) for y, x in path_yx)
        edges.append(
            SkeletonEdge(
                id=edge_id,
                start_node=start_node,
                end_node=end_node,
                pixels=pixels,
                path=path_xy,
                length=_path_length(path_xy),
            )
        )
        incident[start_node].append(edge_id)
        if end_node != start_node:
            incident[end_node].append(edge_id)

    nodes = tuple(
        SkeletonNode(
            id=node_id,
            pixels=tuple(component),
            centre=(float(centre[1]), float(centre[0])),
            incident_edges=tuple(sorted(incident[node_id])),
        )
        for node_id, (component, centre) in enumerate(zip(provisional_nodes, centres))
    )
    owned_pixels = sum(len(node.pixels) for node in nodes) + sum(
        len(edge.pixels) for edge in edges
    )
    if owned_pixels != len(skeleton_pixels):
        raise AssertionError("topology compression did not conserve skeleton pixels")
    return SkeletonTopology(nodes, tuple(edges), len(skeleton_pixels))


def _order_cycle(
    pixels: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    available = set(pixels)
    current = anchor
    ordered = [current]
    available.remove(current)
    previous = None
    while available:
        candidates = [
            point for point in _neighbours(current, shape) if point in available
        ]
        if not candidates:
            break
        following = (
            min(candidates)
            if previous is None
            else max(
                candidates,
                key=lambda point: float(
                    np.dot(
                        np.asarray(point, dtype=float) - current,
                        np.asarray(current, dtype=float) - previous,
                    )
                ),
            )
        )
        previous, current = current, following
        available.remove(current)
        ordered.append(current)
    if available:
        raise ValueError("skeleton cycle is not a single chain")
    return tuple(ordered)
