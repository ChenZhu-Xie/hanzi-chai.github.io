"""Directed seed-and-ink diffusion over a Unicode PDF glyph.

The repository candidate contributes only a symbolic hypothesis: directed
stroke order, leaf ownership, turn sequence, and stroke-contact grammar.  It
does not contribute a target-space component mask.  A route is selected on the
PDF skeleton first; only then is the surrounding ink filled from that route.
"""

from __future__ import annotations

import argparse
import heapq
import html
import importlib.util
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRANSFER = load_module("unihan-stroke-transfer.py", "unihan_stroke_transfer_diffusion")
MATCHER = TRANSFER.MATCHER
VECTOR = TRANSFER.VECTOR
PDF = TRANSFER.PDF
RESIDUAL_DECODER = load_module(
    "unihan-residual-decoder.py", "unihan_residual_decoder_diffusion"
)

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


def crossing_numbers(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbour count and crossing number for every skeleton pixel."""
    padded = np.pad(skeleton.astype(bool), 1)
    ring = [
        padded[1 + dy : 1 + dy + skeleton.shape[0], 1 + dx : 1 + dx + skeleton.shape[1]]
        for dy, dx in NEIGHBOURS
    ]
    count = sum(ring)
    transitions = sum((~left) & right for left, right in zip(ring, ring[1:] + ring[:1]))
    return count.astype(np.uint8), transitions.astype(np.uint8)


def cluster_points(mask: np.ndarray, radius: int = 2) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    expanded = cv2.dilate(mask.astype(np.uint8), np.ones((radius * 2 + 1,) * 2, np.uint8))
    count, labels = cv2.connectedComponents(expanded, connectivity=8)
    points = np.argwhere(mask)
    result = []
    for label in range(1, count):
        members = points[labels[points[:, 0], points[:, 1]] == label]
        if not len(members):
            continue
        center = members.mean(axis=0)
        chosen = members[np.argmin(np.sum((members - center) ** 2, axis=1))]
        result.append((int(chosen[0]), int(chosen[1])))
    return result


@dataclass
class SkeletonGraph:
    points: np.ndarray
    point_index: dict[tuple[int, int], int]
    matrix: csr_matrix
    critical: list[tuple[int, int]]
    components: np.ndarray
    crossing: np.ndarray
    scan_seeds: tuple[tuple[int, int], ...]


def diagonal_front_seeds(
    skeleton: np.ndarray,
    crossing: np.ndarray,
    radius: float = 4.0,
) -> list[tuple[int, int]]:
    """Return local first-contact regions of an upper-left scan front.

    In screen coordinates the moving 45-degree front is ``x + y = c``.  Its
    local minima are useful *temporary route seeds*: notably, the upper-left
    corner of 口 is a degree-two bend, so it is neither an endpoint nor a
    junction in the ordinary skeleton graph.  This does not claim that every
    seed is a pen-down point or impose a global stroke order.  Later semantic
    stroke filtering decides whether a route through the seed is relevant.
    """
    points = np.argwhere(skeleton & (crossing == 2))
    if not len(points):
        return []
    tree = cKDTree(points.astype(float))
    fronts = points[:, 0] + points[:, 1]
    minima = np.zeros_like(skeleton, dtype=bool)
    for index, point in enumerate(points):
        neighbours = tree.query_ball_point(point.astype(float), r=radius)
        if neighbours and fronts[index] <= int(np.min(fronts[neighbours])):
            minima[tuple(point)] = True
    return cluster_points(minima, radius=2)


def build_skeleton_graph(skeleton: np.ndarray) -> SkeletonGraph:
    points = np.argwhere(skeleton)
    point_index = {tuple(map(int, point)): index for index, point in enumerate(points)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for index, (y, x) in enumerate(points):
        for dy, dx in NEIGHBOURS:
            other = point_index.get((int(y + dy), int(x + dx)))
            if other is None or other <= index:
                continue
            weight = math.sqrt(2) if dy and dx else 1.0
            rows.extend((index, other))
            cols.extend((other, index))
            weights.extend((weight, weight))
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(points), len(points)))
    _count, crossing = crossing_numbers(skeleton)
    endpoints = [tuple(map(int, point)) for point in np.argwhere(skeleton & (crossing == 1))]
    junctions = cluster_points(skeleton & (crossing >= 3), radius=2)
    scan_seeds = diagonal_front_seeds(skeleton, crossing)
    critical = list(dict.fromkeys(endpoints + junctions + scan_seeds))
    _component_count, components = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    return SkeletonGraph(
        points,
        point_index,
        matrix,
        critical,
        components,
        crossing,
        tuple(scan_seeds),
    )


def reconstruct_path(
    points: np.ndarray, predecessors: np.ndarray, source: int, target: int
) -> np.ndarray | None:
    if source == target:
        return points[[source]][:, ::-1].astype(float)
    current = target
    indices = [current]
    while current != source:
        current = int(predecessors[current])
        if current < 0:
            return None
        indices.append(current)
        if len(indices) > len(points):
            return None
    indices.reverse()
    return points[np.asarray(indices)][:, ::-1].astype(float)


def graph_adjacency(matrix: csr_matrix) -> list[list[tuple[int, float]]]:
    return [
        [
            (int(matrix.indices[position]), float(matrix.data[position]))
            for position in range(matrix.indptr[node], matrix.indptr[node + 1])
        ]
        for node in range(matrix.shape[0])
    ]


def shortest_indices(
    adjacency: list[list[tuple[int, float]]],
    source: int,
    target: int,
    banned_edge: tuple[int, int] | None = None,
) -> tuple[list[int] | None, float]:
    distance = [math.inf] * len(adjacency)
    predecessor = [-1] * len(adjacency)
    distance[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        current, node = heapq.heappop(queue)
        if current != distance[node]:
            continue
        if node == target:
            break
        for other, weight in adjacency[node]:
            if banned_edge is not None and {node, other} == set(banned_edge):
                continue
            proposed = current + weight
            if proposed + 1e-9 < distance[other]:
                distance[other] = proposed
                predecessor[other] = node
                heapq.heappush(queue, (proposed, other))
    if not math.isfinite(distance[target]):
        return None, math.inf
    path = [target]
    while path[-1] != source:
        parent = predecessor[path[-1]]
        if parent < 0:
            return None, math.inf
        path.append(parent)
    path.reverse()
    return path, distance[target]


def resample(points: np.ndarray, count: int = 24) -> np.ndarray:
    if len(points) <= 1:
        return np.repeat(points[:1], count, axis=0)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def angle_delta(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


def tangent_angles(points: np.ndarray, count: int = 20) -> np.ndarray:
    sampled = resample(points, count + 1)
    vectors = np.diff(sampled, axis=0)
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def direction_dtw(first: np.ndarray, second: np.ndarray) -> float:
    left = tangent_angles(first)
    right = tangent_angles(second)
    table = np.full((len(left) + 1, len(right) + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            local = angle_delta(float(left[i - 1]), float(right[j - 1])) / math.pi
            table[i, j] = local + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[-1, -1] / (len(left) + len(right)))


def total_turn(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    simplified = cv2.approxPolyDP(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        # A PDF skeleton acquires tiny up/down detours at every crossing.  At
        # the previous 1.2% scale those nubs made one straight 皿 baseline look
        # like four full turns.  Three percent is still far below a genuine
        # component-scale corner while suppressing junction-sized noise.
        epsilon=max(2.0, length * 0.03),
        closed=False,
    ).reshape(-1, 2)
    if len(simplified) < 3:
        return 0.0
    vectors = np.diff(simplified, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    return float(
        sum(angle_delta(float(left), float(right)) for left, right in zip(angles, angles[1:]))
    )


def polyline_distance(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return math.inf
    a = resample(first, min(80, max(16, len(first))))
    b = resample(second, min(80, max(16, len(second))))
    return float(min(cKDTree(a).query(b)[0].min(), cKDTree(b).query(a)[0].min()))


def normalized_point(point: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, maximum = bounds
    return (point - minimum) / np.maximum(maximum - minimum, 1.0)


def semantic_direction_compatible(
    feature: str,
    expected_sector: str,
    points: np.ndarray,
) -> bool:
    """Check the named stroke's directed displacement, not only its octant.

    The generic sector tolerance admits one neighbouring octant for raster
    noise.  For 撇 that would also admit a purely horizontal westbound road,
    even though a real 撇 must make material progress both leftward and
    downward.  This invariant is independent of exact font geometry.
    """
    order = RESIDUAL_DECODER.ORDER
    observed = order.direction_sector(points, fraction=0.08)
    if not order.sector_compatible(expected_sector, observed):
        return False
    points = np.asarray(points, dtype=float)
    if feature != "撇" or len(points) < 2:
        return True
    displacement = points[-1] - points[0]
    length = float(np.linalg.norm(displacement))
    return bool(
        length > 1e-6
        and float(displacement[0]) < -0.18 * length
        and float(displacement[1]) > 0.18 * length
    )


@dataclass
class RouteCandidate:
    start_node: int
    end_node: int
    points: np.ndarray
    pixels: frozenset[tuple[int, int]]
    component: int
    score: float
    evidence: dict


def all_routes(graph: SkeletonGraph) -> list[dict]:
    nodes = [graph.point_index[point] for point in graph.critical if point in graph.point_index]
    scan_nodes = {
        graph.point_index[point]
        for point in graph.scan_seeds
        if point in graph.point_index
    }
    base_nodes = [node for node in nodes if node not in scan_nodes]
    # A diagonal-front seed is a local hypothesis, not a new globally
    # connected landmark.  Link it only to nearby ordinary critical nodes in
    # the same ink island.  This retains both directions of useful routes
    # without quadratically connecting every seed to every other seed.
    seed_neighbours: dict[int, set[int]] = {}
    for seed in scan_nodes:
        y, x = graph.points[seed]
        component = int(graph.components[y, x])
        local = []
        for node in base_nodes:
            other_y, other_x = graph.points[node]
            if int(graph.components[other_y, other_x]) != component:
                continue
            distance = math.hypot(float(other_y - y), float(other_x - x))
            local.append((distance, node))
        seed_neighbours[seed] = {
            node for _distance, node in sorted(local)[:6]
        }
    adjacency = graph_adjacency(graph.matrix)
    degrees = np.asarray([len(items) for items in adjacency])
    routes = []
    for start_position, start in enumerate(nodes):
        distances, predecessors = dijkstra(
            graph.matrix, directed=False, indices=start, return_predecessors=True
        )
        for end_position, end in enumerate(nodes):
            if start == end or not np.isfinite(distances[end]):
                continue
            if start in scan_nodes and end not in seed_neighbours[start]:
                continue
            if end in scan_nodes and start not in seed_neighbours[end]:
                continue
            base_indices = []
            current = end
            while current != start:
                base_indices.append(current)
                current = int(predecessors[current])
                if current < 0 or len(base_indices) > len(graph.points):
                    base_indices = []
                    break
            if not base_indices:
                continue
            base_indices.append(start)
            base_indices.reverse()
            alternatives: list[tuple[list[int], float]] = [(base_indices, float(distances[end]))]
            # A single shortest path silently fixes the answer on loops.  At
            # each fork used by that path, close only its chosen outgoing road
            # and ask for the next reachable route.  This exposes topologically
            # distinct roads with the same pen-down and pen-up without an
            # expensive all-simple-path explosion.
            fork_positions = [
                index
                for index in range(len(base_indices) - 1)
                if index == 0 or degrees[base_indices[index]] >= 3
            ]
            for index in fork_positions:
                alternate, alternate_length = shortest_indices(
                    adjacency,
                    start,
                    end,
                    (base_indices[index], base_indices[index + 1]),
                )
                if alternate is None or alternate_length > distances[end] * 3.2:
                    continue
                pixels = frozenset(alternate)
                if any(
                    len(pixels & frozenset(existing)) / max(1, len(pixels | frozenset(existing)))
                    > 0.9
                    for existing, _length in alternatives
                ):
                    continue
                alternatives.append((alternate, float(alternate_length)))
            y, x = graph.points[start]
            for indices, length in sorted(alternatives, key=lambda item: item[1])[:5]:
                path = graph.points[np.asarray(indices)][:, ::-1].astype(float)
                if len(path) < 3:
                    continue
                routes.append(
                    {
                        "startNode": start_position,
                        "endNode": end_position,
                        "points": path,
                        "pixels": frozenset((int(p[0]), int(p[1])) for p in path),
                        "component": int(graph.components[y, x]),
                        "length": length,
                        "diagonalFrontSeed": start in scan_nodes,
                        "diagonalFrontSeedAtEnd": end in scan_nodes,
                    }
                )
    return routes


def stroke_contact_matrix(strokes: list[dict], threshold: float = 3.0) -> np.ndarray:
    size = len(strokes)
    matrix = np.zeros((size, size), dtype=bool)
    for left in range(size):
        for right in range(left):
            contact = polyline_distance(strokes[left]["points"], strokes[right]["points"]) <= threshold
            matrix[left, right] = matrix[right, left] = contact
    return matrix


def diagonal_front_leaf_keys(
    strokes: list[dict],
    contacts: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    """Leaves where a scan seed is a safe route-recovery hypothesis.

    Proven cases are a three-stroke closed box such as 口 and the four-stroke
    broken grass head whose direction signature is horizontal, vertical,
    vertical, horizontal.  Merely starting with a vertical is not enough (日
    and 皿 have additional internal strokes and already possess stable routes).
    Keeping this structural gate avoids exposing every component to
    speculative scan-seed roads.
    """
    if contacts is None:
        contacts = stroke_contact_matrix(strokes)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        key = (int(stroke.get("componentId", -1)), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(index)
    eligible = set()
    for key, indices in groups.items():
        sectors = [
            RESIDUAL_DECODER.ORDER.FEATURE_INITIAL_SECTORS.get(
                str(strokes[index].get("feature") or ""),
                "unknown",
            )
            for index in indices
        ]
        if sectors == ["E", "S", "S", "E"]:
            eligible.add(key)
            continue
        if len(indices) != 3:
            continue
        if sectors[0] != "S":
            continue
        if all(bool(contacts[left, right]) for left in indices for right in indices if left > right):
            eligible.add(key)
    return eligible


def broken_grass_head_keys(strokes: list[dict]) -> set[tuple[int, int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        key = (int(stroke.get("componentId", -1)), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(index)
    result = set()
    for key, indices in groups.items():
        sectors = [
            RESIDUAL_DECODER.ORDER.FEATURE_INITIAL_SECTORS.get(
                str(strokes[index].get("feature") or ""), "unknown"
            )
            for index in indices
        ]
        if sectors == ["E", "S", "S", "E"]:
            result.add(key)
    return result


def contact_signature(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Return closest contact as directed arclength fractions on both paths.

    Absolute coordinates and scale are discarded.  The stable fact retained is
    whether the contact occurs near each stroke's start, middle, or end.
    """
    def dense(points: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return resample(points, max(32, min(256, int(math.ceil(length)) + 1)))

    first = dense(first)
    second = dense(second)
    delta = first[:, None, :] - second[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    first_index, second_index = np.unravel_index(np.argmin(distances), distances.shape)

    def fraction(points: np.ndarray, index: int) -> float:
        if len(points) < 2:
            return 0.0
        if len(points) <= 1:
            return 0.0
        # ``dense`` is sampled at equal arclength intervals.
        return float(index / (len(points) - 1))

    return fraction(first, first_index), fraction(second, second_index), float(distances[first_index, second_index])


def component_mapping(strokes: list[dict], graph: SkeletonGraph) -> dict[tuple[int, int], int]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for stroke in strokes:
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(stroke)
    target_components = []
    for label in sorted(set(graph.components[graph.components > 0].tolist())):
        pixels = np.argwhere(graph.components == label)
        target_components.append((label, pixels[:, ::-1].mean(axis=0)))
    candidate_groups = []
    for key, members in groups.items():
        points = np.vstack([member["points"] for member in members])
        candidate_groups.append((key, points.mean(axis=0)))
    if len(candidate_groups) == len(target_components):
        structural = recursive_structural_component_mapping(groups, target_components)
        if structural is not None:
            return structural
        # Hierarchy-free legacy candidates retain an ordinal fallback.  New
        # candidates must use their recursive decomposition above.
        candidate_groups.sort(key=lambda item: (item[1][1], item[1][0]))
        target_components.sort(key=lambda item: (item[1][1], item[1][0]))
        return {candidate[0]: target[0] for candidate, target in zip(candidate_groups, target_components)}

    # PDF ink islands and semantic leaf components are not one-to-one: several
    # touching leaves may form one island, while a disconnected dot may split a
    # single leaf.  Never collapse every leaf onto the largest island.  Match
    # only their coarse glyph-global centroids, allowing many leaves to share
    # the same target island; no candidate outline or scale is transferred.
    candidate_centres = np.vstack([centre for _key, centre in candidate_groups])
    target_centres = np.vstack([centre for _label, centre in target_components])
    candidate_min = np.vstack([stroke["points"] for stroke in strokes]).min(axis=0)
    candidate_max = np.vstack([stroke["points"] for stroke in strokes]).max(axis=0)
    target_min = graph.points[:, ::-1].min(axis=0)
    target_max = graph.points[:, ::-1].max(axis=0)
    candidate_centres = (candidate_centres - candidate_min) / np.maximum(
        candidate_max - candidate_min, 1.0
    )
    target_centres = (target_centres - target_min) / np.maximum(
        target_max - target_min, 1.0
    )
    return {
        key: target_components[
            int(np.argmin(np.linalg.norm(target_centres - centre, axis=1)))
        ][0]
        for (key, _raw_centre), centre in zip(candidate_groups, candidate_centres)
    }


def recursive_structural_component_mapping(
    groups: dict[tuple[int, int], list[dict]],
    target_components: list[tuple[int, np.ndarray]],
) -> dict[tuple[int, int], int] | None:
    """Map equal-count PDF islands by the candidate decomposition tree.

    Candidate geometry contributes sibling *order* only.  Its absolute size,
    position, and outline never cross into the PDF.  At each ⿰/⿱ node the
    matching target islands are partitioned recursively along that operator's
    axis before any stroke route is ranked.
    """
    if not groups or len(groups) != len(target_components):
        return None
    hierarchy_by_key = {
        key: list((members[0].get("hierarchy") or []))
        for key, members in groups.items()
    }
    if any(len(hierarchy) < 2 for hierarchy in hierarchy_by_key.values()):
        return None
    roots = {
        (int(hierarchy[-1]["id"]), str(hierarchy[-1].get("label", "")))
        for hierarchy in hierarchy_by_key.values()
    }
    if len(roots) != 1:
        return None
    group_centres = {
        key: np.vstack([member["points"] for member in members]).mean(axis=0)
        for key, members in groups.items()
    }
    target_centres = {int(label): np.asarray(centre, dtype=float) for label, centre in target_components}
    mapping: dict[tuple[int, int], int] = {}

    def assign(
        keys: list[tuple[int, int]],
        labels: list[int],
        parent_id: int,
    ) -> bool:
        if len(keys) != len(labels) or not keys:
            return False
        if len(keys) == 1:
            mapping[keys[0]] = labels[0]
            return True
        parent_items = []
        child_buckets: dict[tuple, list[tuple[int, int]]] = {}
        child_parent_ids: dict[tuple, int | None] = {}
        for key in keys:
            hierarchy = hierarchy_by_key[key]
            positions = [
                index for index, item in enumerate(hierarchy) if int(item["id"]) == parent_id
            ]
            if not positions or positions[-1] == 0:
                return False
            position = positions[-1]
            parent_items.append(hierarchy[position])
            child = hierarchy[position - 1]
            if position - 1 == 0:
                token = ("leaf", key)
                child_parent_ids[token] = None
            else:
                token = (
                    "node",
                    int(child["id"]),
                    str(child.get("familyKey", child["id"])),
                )
                child_parent_ids[token] = int(child["id"])
            child_buckets.setdefault(token, []).append(key)
        operators = {str(item.get("label", "")) for item in parent_items}
        if len(operators) != 1:
            return False
        operator = operators.pop()
        if operator in {"⿰", "⿲"}:
            axis = 0
        elif operator in {"⿱", "⿳"}:
            axis = 1
        else:
            return False
        ordered_children = sorted(
            child_buckets,
            key=lambda token: float(
                np.vstack([group_centres[key] for key in child_buckets[token]])[:, axis].mean()
            ),
        )
        ordered_labels = sorted(labels, key=lambda label: float(target_centres[label][axis]))
        offset = 0
        for token in ordered_children:
            child_keys = child_buckets[token]
            child_labels = ordered_labels[offset : offset + len(child_keys)]
            offset += len(child_keys)
            child_parent = child_parent_ids[token]
            if child_parent is None:
                if len(child_keys) != 1 or len(child_labels) != 1:
                    return False
                mapping[child_keys[0]] = child_labels[0]
            elif not assign(child_keys, child_labels, child_parent):
                return False
        return offset == len(ordered_labels)

    root_id, _root_operator = next(iter(roots))
    labels = [int(label) for label, _centre in target_components]
    if not assign(list(groups), labels, root_id) or len(mapping) != len(groups):
        return None
    return mapping


def component_label_options(
    strokes: list[dict], graph: SkeletonGraph
) -> dict[tuple[int, int], set[int]]:
    """Return coarse structural islands allowed for each semantic leaf.

    When leaf and PDF-island counts differ, exact geometry is unsafe.  We keep
    only column membership plus the first/last leaf order inside a column.
    Middle leaves may choose either adjacent island during route search.
    """
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for stroke in strokes:
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(stroke["points"])
    labels = sorted(set(graph.components[graph.components > 0].tolist()))
    if len(groups) == len(labels):
        mapping = component_mapping(strokes, graph)
        return {key: {label} for key, label in mapping.items()}

    candidate_points = np.vstack([stroke["points"] for stroke in strokes])
    candidate_min, candidate_max = candidate_points.min(axis=0), candidate_points.max(axis=0)
    target_points = graph.points[:, ::-1].astype(float)
    target_min, target_max = target_points.min(axis=0), target_points.max(axis=0)
    candidate_centres = {
        key: normalized_point(np.vstack(members).mean(axis=0), (candidate_min, candidate_max))
        for key, members in groups.items()
    }
    target_centres = {
        label: normalized_point(
            np.argwhere(graph.components == label)[:, ::-1].mean(axis=0),
            (target_min, target_max),
        )
        for label in labels
    }
    options = {}
    for key, centre in candidate_centres.items():
        horizontal = {label: abs(float(point[0] - centre[0])) for label, point in target_centres.items()}
        nearest = min(horizontal.values())
        options[key] = {
            label for label, distance in horizontal.items() if distance <= nearest + 0.18
        }

    columns: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for key, allowed in options.items():
        columns.setdefault(tuple(sorted(allowed)), []).append(key)
    for allowed_tuple, keys in columns.items():
        if len(allowed_tuple) <= 1 or len(keys) <= 1:
            continue
        ordered_keys = sorted(keys, key=lambda key: candidate_centres[key][1])
        ordered_labels = sorted(allowed_tuple, key=lambda label: target_centres[label][1])
        options[ordered_keys[0]] = {ordered_labels[0]}
        options[ordered_keys[-1]] = {ordered_labels[-1]}
    return options


def has_forward_continuation(
    graph: SkeletonGraph,
    points: np.ndarray,
    minimum_cosine: float = 0.5,
    adjacency: list[list[tuple[int, float]]] | None = None,
) -> bool:
    """Whether a route stops while its current road visibly continues.

    This is deliberately a local topological test rather than a length prior.
    A legitimate pen-up at a skeleton endpoint is accepted; at any other node,
    an unconsumed neighbour aligned with the incoming tangent means that the
    route has stopped in the middle of a continuous stroke.
    """
    if len(points) < 2:
        return False
    end_xy = np.asarray(points[-1], dtype=float)
    end_yx = tuple(np.rint(end_xy[::-1]).astype(int))
    end_index = graph.point_index.get(end_yx)
    if end_index is None:
        distances = np.linalg.norm(graph.points[:, ::-1] - end_xy, axis=1)
        end_index = int(np.argmin(distances))
        if distances[end_index] > 2.0:
            return False
        end_yx = tuple(map(int, graph.points[end_index]))
    if int(graph.crossing[end_yx]) == 1:
        return False

    # Use a several-pixel tail rather than the final pixel so one-pixel
    # skeleton stair-steps do not change the inferred pen direction.
    tail = np.asarray(points[-1], dtype=float)
    anchor = np.asarray(points[0], dtype=float)
    walked = 0.0
    for index in range(len(points) - 2, -1, -1):
        walked += float(np.linalg.norm(points[index + 1] - points[index]))
        anchor = np.asarray(points[index], dtype=float)
        if walked >= 8.0:
            break
    incoming = tail - anchor
    norm = float(np.linalg.norm(incoming))
    if norm <= 1e-9:
        return False
    incoming /= norm

    route_pixels = {
        (int(round(point[0])), int(round(point[1]))) for point in points
    }
    adjacency = adjacency or graph_adjacency(graph.matrix)
    for neighbour, _weight in adjacency[end_index]:
        neighbour_xy = graph.points[neighbour][::-1].astype(float)
        pixel = (int(neighbour_xy[0]), int(neighbour_xy[1]))
        if pixel in route_pixels:
            continue
        outgoing = neighbour_xy - tail
        outgoing_norm = float(np.linalg.norm(outgoing))
        if outgoing_norm <= 1e-9:
            continue
        if float(np.dot(incoming, outgoing / outgoing_norm)) >= minimum_cosine:
            return True
    return False


def stroke_rule_signatures(strokes: list[dict]) -> list[dict]:
    """Identify a reusable stroke role inside an exact leaf component."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(index)
    output = []
    for index, stroke in enumerate(strokes):
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        members = groups[key]
        ordinal = members.index(index) + 1
        count = len(members)
        feature = str(stroke.get("feature") or "未知")
        hierarchy = stroke.get("hierarchy") or []
        root_operator = str(hierarchy[-1].get("label", "unknown")) if hierarchy else "unknown"
        context_key = f"depth:{len(hierarchy)}|root:{root_operator}"
        output.append(
            {
                "key": f"{key[0]}|{ordinal}|{count}|{feature}",
                "componentId": key[0],
                "occurrence": key[1],
                "ordinal": ordinal,
                "count": count,
                "feature": feature,
                "contextKey": context_key,
            }
        )
    return output


def point_topology_role(graph: SkeletonGraph, point: np.ndarray) -> str:
    distances = np.linalg.norm(graph.points[:, ::-1].astype(float) - point, axis=1)
    nearest = graph.points[int(np.argmin(distances))]
    crossing = int(graph.crossing[tuple(nearest)])
    if crossing <= 1:
        return "endpoint"
    if crossing >= 3:
        return "junction"
    return "path"


def direction_sector(points: np.ndarray) -> str:
    sampled = resample(points, 24)
    window = min(6, len(sampled) - 1)
    vector = sampled[window] - sampled[0]
    angle = (math.degrees(math.atan2(float(vector[1]), float(vector[0]))) + 360.0) % 360.0
    sectors = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    return sectors[int((angle + 22.5) // 45.0) % 8]


def relative_turn_class(incoming: np.ndarray, outgoing: np.ndarray) -> str:
    left_norm = float(np.linalg.norm(incoming))
    right_norm = float(np.linalg.norm(outgoing))
    if left_norm < 1e-6 or right_norm < 1e-6:
        return "unknown"
    incoming = incoming / left_norm
    outgoing = outgoing / right_norm
    signed = math.degrees(
        math.atan2(
            float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]),
            float(np.dot(incoming, outgoing)),
        )
    )
    if abs(signed) <= 28.0:
        return "straight"
    if abs(signed) >= 152.0:
        return "reverse"
    # SVG/PDF y grows downward, so use unambiguous screen-clock terminology.
    return "clockwise" if signed > 0 else "counterclockwise"


def junction_decision_details(points: np.ndarray, graph: SkeletonGraph) -> list[dict]:
    """Describe chosen and locally available exits at real T/cross junctions."""
    trace = critical_node_trace(points, graph)
    if not trace:
        return []
    sampled = resample(points, 180)
    adjacency = graph_adjacency(graph.matrix)
    output = []
    for item in trace:
        point = np.asarray(item["point"], dtype=float)
        index = int(np.argmin(np.linalg.norm(sampled - point, axis=1)))
        before = max(0, index - 6)
        after = min(len(sampled) - 1, index + 6)
        if before == index or after == index:
            continue
        incoming = sampled[index] - sampled[before]
        chosen = relative_turn_class(incoming, sampled[after] - sampled[index])
        critical_yx = graph.critical[int(item["node"])]
        graph_index = graph.point_index.get(critical_yx)
        available = set()
        if graph_index is not None:
            origin = graph.points[graph_index][::-1].astype(float)
            for neighbour, _weight in adjacency[graph_index]:
                outgoing = graph.points[neighbour][::-1].astype(float) - origin
                available.add(relative_turn_class(incoming, outgoing))
        available.discard("unknown")
        output.append(
            {
                "point": item["point"],
                "fraction": item["fraction"],
                "chosen": chosen,
                "available": sorted(available),
                "rejected": sorted(available - {chosen, "reverse"}),
            }
        )
    return output


def junction_turn_sequence(points: np.ndarray, graph: SkeletonGraph) -> list[str]:
    """Return the directed decisions made at real T/cross junctions."""
    return [item["chosen"] for item in junction_decision_details(points, graph)]


def learned_rule_for_stroke(
    model: dict | None,
    signature: dict,
    excluded_case: str | None,
    source: str | None = None,
) -> dict | None:
    if not model:
        return None
    schema_version = int(model.get("schemaVersion", 1))
    if schema_version >= 2:
        if not source:
            return None
        lookup_key = f"{source.upper()}|{signature['key']}"
    else:
        lookup_key = signature["key"]
    examples = [
        example
        for example in model.get("examplesBySignature", {}).get(lookup_key, [])
        if example.get("case") != excluded_case
        and example.get("contextKey") == signature.get("contextKey")
    ]
    if not examples:
        return None
    cases = sorted({str(example["case"]) for example in examples})
    role_counts: dict[str, int] = {}
    sector_counts: dict[str, int] = {}
    turn_counts: dict[tuple[str, ...], int] = {}
    junction_choice_counts: dict[str, int] = {}
    earlier_contact_counts: dict[bool, int] = {}
    positions = []
    glyph_positions = []
    for example in examples:
        role = str(example["start"]["topologyRole"])
        role_counts[role] = role_counts.get(role, 0) + 1
        sector = str(example["start"]["directionSector"])
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        turns = tuple(example["junctionTurnSequence"])
        turn_counts[turns] = turn_counts.get(turns, 0) + 1
        for junction in example.get("junctions", []):
            choice = str(junction["chosen"])
            junction_choice_counts[choice] = junction_choice_counts.get(choice, 0) + 1
        touches_earlier = bool(example["start"].get("touchesEarlierStroke", False))
        earlier_contact_counts[touches_earlier] = earlier_contact_counts.get(touches_earlier, 0) + 1
        positions.append(example["start"]["componentNormalized"])
        glyph_positions.append(example["start"]["glyphNormalized"])
    dominant_role, role_support = max(role_counts.items(), key=lambda item: item[1])
    dominant_sector, sector_support = max(sector_counts.items(), key=lambda item: item[1])
    dominant_turns, turn_support = max(turn_counts.items(), key=lambda item: item[1])
    dominant_contact, contact_support = max(
        earlier_contact_counts.items(), key=lambda item: item[1]
    )
    dominant_choice, choice_support = (
        max(junction_choice_counts.items(), key=lambda item: item[1])
        if junction_choice_counts
        else (None, 0)
    )
    component_positions = np.asarray(positions, dtype=float)
    glyph_positions_array = np.asarray(glyph_positions, dtype=float)
    return {
        "signature": signature["key"],
        "support": len(cases),
        "exampleCount": len(examples),
        "cases": cases,
        "startRole": dominant_role,
        "startRoleConfidence": role_support / len(examples),
        "startSector": dominant_sector,
        "startSectorConfidence": sector_support / len(examples),
        "componentStartMean": component_positions.mean(axis=0).tolist(),
        "componentStartStd": component_positions.std(axis=0).tolist(),
        "glyphStartMean": glyph_positions_array.mean(axis=0).tolist(),
        "glyphStartStd": glyph_positions_array.std(axis=0).tolist(),
        "junctionTurns": list(dominant_turns),
        "junctionConfidence": turn_support / len(examples),
        "junctionChoice": dominant_choice,
        "junctionChoiceConfidence": (
            choice_support / sum(junction_choice_counts.values())
            if junction_choice_counts
            else 0.0
        ),
        "touchesEarlierStroke": dominant_contact,
        "touchesEarlierConfidence": contact_support / len(examples),
    }


def route_learned_cost(
    route: dict,
    graph: SkeletonGraph,
    rule: dict | None,
    target_glyph_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[float, dict]:
    role = point_topology_role(graph, route["points"][0])
    sector = direction_sector(route["points"])
    turns = junction_turn_sequence(route["points"], graph)
    if rule is None:
        return 0.0, {
            "trainingSupport": 0,
            "routeStartRole": role,
            "routeJunctionTurns": turns,
            "explanation": "no held-out exact-component training rule; no learned score applied",
        }
    # One example is useful explanatory evidence but is too brittle to steer
    # the route. Two or more independent cases unlock a bounded learned cost.
    enabled = int(rule["support"]) >= 2
    role_mismatch = role != rule["startRole"]
    sector_mismatch = sector != rule["startSector"]
    route_glyph_start = normalized_point(route["points"][0], target_glyph_bounds)
    glyph_start_distance = float(
        np.linalg.norm(route_glyph_start - np.asarray(rule["glyphStartMean"], dtype=float))
    )
    # Skeleton role is deliberately explanatory only: a pen start can change
    # from endpoint to ordinary path merely because another stroke crosses it.
    # A glyph-relative start prior is allowed only when independent cases put
    # this exact leaf stroke in a tight, repeatable region.
    stable_start = max(rule["glyphStartStd"]) <= 0.08
    position_cost = (
        min(0.35, max(0.0, glyph_start_distance - 0.10) * 0.8)
        if enabled and stable_start
        else 0.0
    )
    sector_cost = (
        0.12 * float(rule["startSectorConfidence"])
        if enabled and rule["startSectorConfidence"] >= 0.8 and sector_mismatch
        else 0.0
    )
    # Compare each encountered decision, not the number of detected junctions.
    # Different fonts can merge/split a junction cluster; absence of a detected
    # node is therefore never penalised.  A consistent human choice can only
    # reject an explicit contradictory turn.
    expected_choice = rule["junctionChoice"]
    contradictory_turns = (
        sum(turn != expected_choice for turn in turns)
        if expected_choice is not None
        else 0
    )
    turn_cost = (
        min(0.7, 0.35 * contradictory_turns * float(rule["junctionChoiceConfidence"]))
        if enabled and rule["junctionChoiceConfidence"] >= 0.85
        else 0.0
    )
    turns_mismatch = bool(contradictory_turns)
    clauses = [
        f"start role {rule['startRole']} is explanatory only",
        f"start sector {rule['startSector']} ({rule['startSectorConfidence']:.0%})",
        "junction choices " + ("do not contradict" if not turns_mismatch else "contradict held-out truth"),
    ]
    return position_cost + sector_cost + turn_cost, {
        "trainingSupport": int(rule["support"]),
        "trainingCases": rule["cases"],
        "learnedScoringEnabled": enabled,
        "learnedStartRole": rule["startRole"],
        "routeStartRole": role,
        "startRoleMismatch": role_mismatch,
        "startRoleScoringEnabled": False,
        "learnedStartSector": rule["startSector"],
        "routeStartSector": sector,
        "startSectorMismatch": sector_mismatch,
        "stableGlyphStart": stable_start,
        "learnedGlyphStart": [round(float(value), 5) for value in rule["glyphStartMean"]],
        "routeGlyphStart": [round(float(value), 5) for value in route_glyph_start],
        "glyphStartDistance": round(glyph_start_distance, 5),
        "learnedJunctionTurns": rule["junctionTurns"],
        "routeJunctionTurns": turns,
        "junctionTurnsMismatch": turns_mismatch,
        "learnedStartCost": round(position_cost + sector_cost, 5),
        "learnedJunctionCost": round(turn_cost, 5),
        "learnedTouchesEarlierStroke": rule["touchesEarlierStroke"],
        "learnedTouchesEarlierConfidence": round(
            float(rule["touchesEarlierConfidence"]), 5
        ),
        "explanation": "; ".join(clauses),
    }


def rank_routes(
    strokes: list[dict],
    graph: SkeletonGraph,
    raw_routes: list[dict],
    ordinal_weight: float = 0.12,
    learned_model: dict | None = None,
    excluded_training_case: str | None = None,
    source: str | None = None,
    enforce_component_labels: bool = True,
) -> list[list[RouteCandidate]]:
    allowed_components = component_label_options(strokes, graph)
    all_candidate_points = np.vstack([stroke["points"] for stroke in strokes])
    candidate_glyph_bounds = (
        all_candidate_points.min(axis=0),
        all_candidate_points.max(axis=0),
    )
    target_glyph_points = graph.points[:, ::-1].astype(float)
    target_glyph_bounds = (
        target_glyph_points.min(axis=0),
        target_glyph_points.max(axis=0),
    )
    candidate_diagonal = float(
        np.linalg.norm(candidate_glyph_bounds[1] - candidate_glyph_bounds[0])
    )
    target_diagonal = float(
        np.linalg.norm(target_glyph_bounds[1] - target_glyph_bounds[0])
    )

    candidate_contacts = stroke_contact_matrix(strokes)
    diagonal_seed_leaves = diagonal_front_leaf_keys(strokes, candidate_contacts)
    candidate_end_is_contact = []
    for index, stroke in enumerate(strokes):
        constrained = False
        for other_index, other in enumerate(strokes):
            if index == other_index or not candidate_contacts[index, other_index]:
                continue
            current_fraction, _other_fraction, _distance = contact_signature(
                stroke["points"], other["points"]
            )
            if current_fraction >= 0.84:
                constrained = True
                break
        candidate_end_is_contact.append(constrained)

    adjacency = graph_adjacency(graph.matrix)
    signatures = stroke_rule_signatures(strokes)
    ranked = []
    for stroke_index, stroke in enumerate(strokes):
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        expected_turn = total_turn(stroke["points"])
        expected_length_fraction = float(
            np.linalg.norm(np.diff(stroke["points"], axis=0), axis=1).sum()
            / max(1.0, candidate_diagonal)
        )
        # Coarse glyph-global order is meaningful (top before bottom, left
        # before right); a component-relative coordinate is not.  In a fully
        # connected PDF skeleton the latter would incorrectly stretch every
        # leaf component over the whole character.
        expected_start = normalized_point(stroke["points"][0], candidate_glyph_bounds)
        learned_rule = learned_rule_for_stroke(
            learned_model,
            signatures[stroke_index],
            excluded_training_case,
            source=source,
        )
        expected_sector = RESIDUAL_DECODER.ORDER.FEATURE_INITIAL_SECTORS.get(
            str(stroke.get("feature") or ""),
            "unknown",
        )
        options = []
        component_recovery_options = []
        for route in raw_routes:
            # A diagonal seed resolves the specific ambiguity “which complete
            # vertical covers this leaf's first-contact region?”.  Do not
            # expose those extra roads to leaves whose first semantic stroke
            # is not vertical (八, 扌, etc.), and never run a stroke backwards
            # merely to terminate at a seed.
            if route.get("diagonalFrontSeedAtEnd") or (
                route.get("diagonalFrontSeed")
                and key not in diagonal_seed_leaves
            ):
                continue
            component_allowed = route["component"] in allowed_components[key]
            unnormalized = RouteCandidate(
                start_node=route["startNode"],
                end_node=route["endNode"],
                points=route["points"],
                pixels=route["pixels"],
                component=int(route["component"]),
                score=0.0,
                evidence={},
            )
            # Only a semantic vertical needs its sizeable calligraphic head
            # cap removed before local ranking.  Applying this early to curves
            # and diagonals perturbs legitimate 撇/捺 entry geometry; those
            # retain the established post-selection normalization path.
            normalized = (
                trim_initial_medial_spur(unnormalized, expected_sector)
                if expected_sector == "S"
                else unnormalized
            )
            scoring_route = {**route, "points": normalized.points}
            direction = direction_dtw(stroke["points"], normalized.points)
            turn = abs(total_turn(normalized.points) - expected_turn) / math.pi
            route_start = normalized_point(normalized.points[0], target_glyph_bounds)
            ordinal = float(np.linalg.norm(route_start - expected_start))
            normalized_length = float(
                np.linalg.norm(np.diff(normalized.points, axis=0), axis=1).sum()
            )
            route_length_fraction = normalized_length / max(1.0, target_diagonal)
            premature_stop = (
                not candidate_end_is_contact[stroke_index]
                and has_forward_continuation(
                    graph, normalized.points, adjacency=adjacency
                )
                and route_length_fraction < expected_length_fraction * 0.62
            )
            # Direction/turn grammar dominates. Ordinal position is a weak tie
            # breaker. A free pen-up cannot truncate a visibly continuing road.
            premature_stop_cost = 1.6 if premature_stop else 0.0
            learned_cost, learned_evidence = route_learned_cost(
                scoring_route, graph, learned_rule, target_glyph_bounds
            )
            component_recovery_cost = (
                0.6 if enforce_component_labels and not component_allowed else 0.0
            )
            score = (
                5.0 * direction
                + 0.8 * turn
                + ordinal_weight * ordinal
                + premature_stop_cost
                + learned_cost
                + component_recovery_cost
            )
            candidate = RouteCandidate(
                    start_node=route["startNode"],
                    end_node=route["endNode"],
                    points=normalized.points,
                    pixels=route["pixels"],
                    component=int(route["component"]),
                    score=score,
                    evidence={
                        "directionGrammar": round(direction, 5),
                        "turnGrammar": round(turn, 5),
                        "ordinalTieBreak": round(ordinal, 5),
                        "prematureJunctionStop": premature_stop,
                        "prematureStopCost": premature_stop_cost,
                        "candidateLengthFraction": round(expected_length_fraction, 5),
                        "routeLengthFraction": round(route_length_fraction, 5),
                        "learnedRuleSignature": signatures[stroke_index]["key"],
                        "learnedCost": round(learned_cost, 5),
                        "componentLabelAllowed": component_allowed,
                        "componentLabelRecoveryCost": component_recovery_cost,
                        "diagonalFrontSeedRoute": bool(
                            route.get("diagonalFrontSeed")
                        ),
                        **normalized.evidence,
                        **learned_evidence,
                    },
                )
            if enforce_component_labels and not component_allowed:
                component_recovery_options.append(candidate)
            else:
                options.append(candidate)
        options.sort(key=lambda item: item.score)
        component_recovery_options.sort(key=lambda item: item.score)
        if not options:
            options = component_recovery_options
            component_recovery_options = []
        if not options:
            ranked.append([])
            continue
        best_direction = min(item.evidence["directionGrammar"] for item in options)
        # Pen direction is a hard local constraint.  Global contact consistency
        # may choose among compatible roads, but must not turn a horizontal
        # stroke into a diagonal/vertical road merely to gain a contact vote.
        compatible = [
            item
            for item in options
            if item.evidence["directionGrammar"] <= best_direction + 0.22
        ]
        # Do not let a local template-DTW winner erase every route in another
        # pen-down direction before the semantic stroke-order guard runs.  A
        # connected neighbouring component can make the wrong road look more
        # like the candidate globally (notably the two strokes of 八).  Keep a
        # small, score-ordered reserve for every observed start sector so the
        # whole-character decoder can backtrack into the correct universe.
        retained = list(compatible[:48])
        retained_ids = {id(item) for item in retained}
        sector_counts: dict[str, int] = {}
        order = RESIDUAL_DECODER.ORDER
        for item in options:
            sector = order.direction_sector(item.points, fraction=0.08)
            if sector_counts.get(sector, 0) >= 12:
                continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if id(item) not in retained_ids:
                retained.append(item)
                retained_ids.add(id(item))
        recovered = []
        recovered_paths: set[tuple[tuple[int, int], ...]] = set()
        for item in component_recovery_options:
            normalized = trim_initial_medial_spur(item, expected_sector)
            if semantic_direction_compatible(
                str(stroke.get("feature") or ""),
                expected_sector,
                normalized.points,
            ):
                path_key = tuple(
                    (int(round(float(x))), int(round(float(y))))
                    for x, y in normalized.points
                )
                if path_key in recovered_paths:
                    continue
                recovered_paths.add(path_key)
                recovered.append(
                    RouteCandidate(
                        start_node=item.start_node,
                        end_node=item.end_node,
                        points=item.points,
                        pixels=item.pixels,
                        component=item.component,
                        score=item.score,
                        evidence={
                            **item.evidence,
                            "recoveredAcrossConnectedComponentBoundary": True,
                        },
                    )
                )
            if len(recovered) >= 24:
                break
        retained.extend(recovered)
        retained.sort(key=lambda item: item.score)
        ranked.append(retained)
    return ranked


def apply_stroke_order_guard(
    strokes: list[dict],
    ranked: list[list[RouteCandidate]],
    source: str,
    codepoint: int,
    normative_catalog=None,
) -> tuple[list[list[RouteCandidate]], list[dict]]:
    """Add stroke-order knowledge without replacing the classic decoder.

    Candidate geometry already supplies a directed stroke grammar.  This guard
    makes its pen-down direction explicit and source-auditable.  A direction
    backed by hard semantic/repository evidence stays hard even when every
    locally preferred road disagrees: restoring those opposite roads would
    make a 撇 consume a 捺 and prevents whole-character backtracking.
    """
    order = RESIDUAL_DECODER.ORDER
    expectations = order.compile_stroke_expectations(
        strokes,
        source,
        codepoint,
        normative_catalog,
    )
    guarded: list[list[RouteCandidate]] = []
    audit = []
    for expectation, options in zip(expectations, ranked):
        compatible = []
        rejected = []
        observed_sector_counts: dict[str, int] = {}
        for option in options:
            option = trim_initial_medial_spur(option, expectation.expected_sector)
            observed = order.direction_sector(option.points, fraction=0.08)
            observed_sector_counts[observed] = observed_sector_counts.get(observed, 0) + 1
            if semantic_direction_compatible(
                expectation.feature,
                expectation.expected_sector,
                option.points,
            ):
                evidence = {
                    **option.evidence,
                    "strokeOrderExpectedSector": expectation.expected_sector,
                    "strokeOrderObservedSector": observed,
                    "strokeOrderEvidence": [
                        {
                            "level": item.level,
                            "provenance": item.provenance,
                            "rule": item.rule,
                            "hard": item.hard,
                        }
                        for item in expectation.evidence
                    ],
                }
                compatible.append(
                    RouteCandidate(
                        start_node=option.start_node,
                        end_node=option.end_node,
                        points=option.points,
                        pixels=option.pixels,
                        component=option.component,
                        score=option.score,
                        evidence=evidence,
                    )
                )
            else:
                rejected.append(option)
        hard_direction_constraint = any(
            bool(item.hard)
            and item.rule in {
                "semantic-pen-down-direction",
                "candidate-stroke-order-and-direction",
            }
            for item in expectation.evidence
        )
        fallback = bool(options) and not compatible and not hard_direction_constraint
        guarded.append(list(options) if fallback else compatible)
        audit.append(
            {
                "stroke": expectation.stroke_index + 1,
                "componentId": expectation.component_id,
                "feature": expectation.feature,
                "expectedSector": expectation.expected_sector,
                "classicCandidateCount": len(options),
                "acceptedCandidateCount": len(options) if fallback else len(compatible),
                "rejectedOppositeDirection": len(rejected),
                "observedSectorCounts": observed_sector_counts,
                "fallbackToClassicCandidates": fallback,
                "hardDirectionConstraint": hard_direction_constraint,
                "evidence": [
                    {
                        "level": item.level,
                        "provenance": item.provenance,
                        "rule": item.rule,
                        "hard": item.hard,
                    }
                    for item in expectation.evidence
                ],
            }
        )
    return guarded, audit


def normalize_selected_pen_paths(
    strokes: list[dict],
    routes: list[RouteCandidate],
    graph: SkeletonGraph,
    source: str,
    codepoint: int,
    normative_catalog=None,
) -> list[RouteCandidate]:
    """Normalize medial-axis artefacts only after global route selection.

    Route selection must retain the classic candidates' complete coverage
    evidence.  This pass changes only the directed pen centreline once the
    whole-character solution is frozen.
    """
    expectations = RESIDUAL_DECODER.ORDER.compile_stroke_expectations(
        strokes,
        source,
        codepoint,
        normative_catalog,
    )
    normalized = []
    for stroke, route, expectation in zip(strokes, routes, expectations):
        route = trim_selected_vertical_medial_spur(route, expectation.expected_sector)
        route = normalize_hook_terminal_branch(route, graph, str(stroke.get("feature", "")))
        normalized.append(route)
    return normalized


def normalize_hook_terminal_branch(
    option: RouteCandidate,
    graph: SkeletonGraph,
    feature: str,
    minimum_gain: float = 1.25,
) -> RouteCandidate:
    """Prefer the longest unexplored terminal branch of a selected hook.

    Medial-axis extraction can split a calligraphic hook into a short pen-rest
    spur and a longer stroke body.  Starting at the final junction, this pass
    blocks the already consumed trunk and explores only the remaining maze.
    It changes the directed centreline, never the owned ink pixels.
    """
    if "钩" not in feature or len(option.points) < 4:
        return option
    trace = critical_node_trace(option.points, graph)
    if not trace:
        return option
    zone = trace[-1]
    points = np.asarray(option.points, dtype=float)
    zone_point = np.asarray(zone["point"], dtype=float)
    anchor_position = int(np.argmin(np.linalg.norm(points - zone_point, axis=1)))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if total <= 1e-6 or float(cumulative[anchor_position] / total) < 0.65:
        return option

    route_nodes = [
        graph.point_index.get((int(round(float(y))), int(round(float(x)))))
        for x, y in points
    ]
    anchor = route_nodes[anchor_position]
    if anchor is None:
        return option
    blocked = {node for node in route_nodes[:anchor_position] if node is not None}
    adjacency = graph_adjacency(graph.matrix)
    distance = [math.inf] * len(adjacency)
    predecessor = [-1] * len(adjacency)
    distance[anchor] = 0.0
    queue = [(0.0, anchor)]
    while queue:
        current, node = heapq.heappop(queue)
        if current != distance[node]:
            continue
        for other, weight in adjacency[node]:
            if other in blocked:
                continue
            proposed = current + weight
            if proposed >= distance[other]:
                continue
            distance[other] = proposed
            predecessor[other] = node
            heapq.heappush(queue, (proposed, other))

    endpoints = []
    component = option.component
    for point in np.argwhere(graph.crossing == 1):
        y, x = map(int, point)
        node = graph.point_index.get((y, x))
        if (
            node is not None
            and node not in blocked
            and int(graph.components[y, x]) == component
            and math.isfinite(distance[node])
        ):
            endpoints.append(node)
    if not endpoints:
        return option
    endpoint = max(endpoints, key=lambda node: distance[node])
    original_length = float(total - cumulative[anchor_position])
    replacement_length = float(distance[endpoint])
    if replacement_length <= max(2.0, original_length * minimum_gain):
        return option

    indices = [endpoint]
    while indices[-1] != anchor:
        parent = predecessor[indices[-1]]
        if parent < 0 or len(indices) > len(graph.points):
            return option
        indices.append(parent)
    indices.reverse()
    replacement = graph.points[np.asarray(indices)][:, ::-1].astype(float)
    normalized_points = np.vstack([points[:anchor_position], replacement])
    return RouteCandidate(
        start_node=option.start_node,
        end_node=option.end_node,
        points=normalized_points,
        pixels=option.pixels,
        component=option.component,
        score=option.score,
        evidence={
            **option.evidence,
            "normalizedTerminalHookBranch": True,
            "normalizedTerminalHookAnchor": [
                round(float(value), 2) for value in replacement[0]
            ],
            "normalizedTerminalHookOriginalEnd": [
                round(float(value), 2) for value in points[-1]
            ],
            "normalizedTerminalHookReplacementEnd": [
                round(float(value), 2) for value in replacement[-1]
            ],
            "normalizedTerminalHookOriginalLength": round(original_length, 5),
            "normalizedTerminalHookReplacementLength": round(replacement_length, 5),
        },
    )


def trim_initial_medial_spur(
    option: RouteCandidate,
    expected_sector: str,
    maximum_fraction: float | None = None,
) -> RouteCandidate:
    """Lightly normalize candidate caps before stroke-order comparison."""
    if maximum_fraction is None:
        # A vertical stroke in a serif/calligraphic source often begins on a
        # sizeable horizontal head cap.  That cap is ink owned by the stroke,
        # but it is not the directed pen-down road.  Allow the logical tip to
        # advance farther than for other strokes so 横-before-竖 ordering and
        # same-leaf geometry are compared against the stable vertical trunk.
        maximum_fraction = 0.30 if expected_sector == "S" else 0.12
    points = np.asarray(option.points, dtype=float)
    if len(points) < 3:
        return option
    order = RESIDUAL_DECODER.ORDER
    if order.sector_compatible(
        expected_sector,
        order.direction_sector(points, fraction=0.06),
    ):
        return option
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if total <= 1e-6:
        return option
    chosen = None
    for index in range(1, len(points) - 1):
        if float(cumulative[index] / total) > maximum_fraction:
            break
        observed = order.direction_sector(points[index:], fraction=0.06)
        if order.sector_compatible(expected_sector, observed):
            chosen = index
            break
    if chosen is None:
        return option
    trimmed_points = points[chosen:]
    return RouteCandidate(
        start_node=option.start_node,
        end_node=option.end_node,
        points=trimmed_points,
        # The cap remains ink owned by this stroke.  Only the logical directed
        # centreline starts at the stable trunk after the calligraphic spur.
        pixels=option.pixels,
        component=option.component,
        score=option.score,
        evidence={
            **option.evidence,
            "trimmedInitialMedialSpur": True,
            "trimmedInitialMedialSpurPixels": chosen,
            "trimmedInitialMedialSpurFraction": round(float(cumulative[chosen] / total), 5),
        },
    )


def trim_selected_vertical_medial_spur(
    option: RouteCandidate,
    expected_sector: str,
    maximum_fraction: float = 0.12,
) -> RouteCandidate:
    """Remove a vertical stroke's cap after the global route is selected.

    The removed skeleton remains target ink and will regain the stroke's colour
    during width restoration.  Only the artificial centreline detour is
    removed, so an endpoint serif cannot masquerade as the first turn of a
    vertical hook or other directed stroke.
    """
    # A vertical pen-down has an unambiguous local tangent, so a short
    # horizontal/diagonal cap is medial-axis noise.  Curved or rising strokes
    # may legitimately enter through an adjacent sector; do not normalize
    # those until their stroke-specific grammar can prove the same invariant.
    if expected_sector != "S":
        return option
    points = np.asarray(option.points, dtype=float)
    if len(points) < 3:
        return option
    order = RESIDUAL_DECODER.ORDER
    if order.direction_sector(points[:2], fraction=1.0) == expected_sector:
        return option
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if total <= 1e-6:
        return option
    chosen = None
    for index in range(1, len(points) - 1):
        if float(cumulative[index] / total) > maximum_fraction:
            break
        observed = order.direction_sector(
            points[index : min(len(points), index + 2)],
            fraction=1.0,
        )
        if observed == expected_sector:
            chosen = index
            break
    if chosen is None:
        return option
    trimmed_points = points[chosen:]
    return RouteCandidate(
        start_node=option.start_node,
        end_node=option.end_node,
        points=trimmed_points,
        pixels=option.pixels,
        component=option.component,
        score=option.score,
        evidence={
            **option.evidence,
            "trimmedInitialMedialSpur": True,
            "trimmedInitialMedialSpurPixels": chosen,
            "trimmedInitialMedialSpurFraction": round(float(cumulative[chosen] / total), 5),
        },
    )


def _candidate_endpoint_contacts(strokes: list[dict], threshold: float = 3.0) -> list[tuple[bool, bool]]:
    """Whether each candidate pen-down/up intentionally touches another stroke."""
    sampled = [resample(np.asarray(stroke["points"], dtype=float), 80) for stroke in strokes]
    output = []
    for index, stroke in enumerate(strokes):
        points = np.asarray(stroke["points"], dtype=float)
        roles = []
        for endpoint in (points[0], points[-1]):
            touches = any(
                float(np.linalg.norm(other - endpoint, axis=1).min()) <= threshold
                for other_index, other in enumerate(sampled)
                if other_index != index
            )
            roles.append(touches)
        output.append((roles[0], roles[1]))
    return output


def apply_stroke_topology_guard(
    strokes: list[dict],
    ranked: list[list[RouteCandidate]],
    graph: SkeletonGraph,
) -> tuple[list[list[RouteCandidate]], list[dict]]:
    """Keep free pen endpoints distinct from internal whole-glyph junctions.

    A junction is a possible *contact* along a stroke, not automatically a
    pen-down or pen-up.  A free pen-down at a junction is incomplete.  A free
    pen-up at a junction, however, is a valid parallel universe when the
    candidate-length grammar says the stroke is already complete: another leaf
    may merely touch it there.  Only an explicitly premature junction stop is
    removed before whole-character look-ahead.
    """
    endpoint_contacts = _candidate_endpoint_contacts(strokes)
    guarded: list[list[RouteCandidate]] = []
    audit = []
    for index, options in enumerate(ranked):
        start_is_contact, end_is_contact = endpoint_contacts[index]
        accepted = []
        rejected_starts = 0
        rejected_ends = 0
        kept_complete_junction_ends = 0
        for option in options:
            start_role = point_topology_role(graph, option.points[0])
            end_role = point_topology_role(graph, option.points[-1])
            wrong_start = not start_is_contact and start_role == "junction"
            junction_end = not end_is_contact and end_role == "junction"
            wrong_end = junction_end and bool(
                option.evidence.get("prematureJunctionStop", False)
            )
            if junction_end and not wrong_end:
                kept_complete_junction_ends += 1
            if wrong_start:
                rejected_starts += 1
            if wrong_end:
                rejected_ends += 1
            if wrong_start or wrong_end:
                continue
            evidence = {
                **option.evidence,
                "candidatePenDownTouchesSibling": start_is_contact,
                "candidatePenUpTouchesSibling": end_is_contact,
                "routePenDownRole": start_role,
                "routePenUpRole": end_role,
                "completeStrokeTopology": True,
            }
            accepted.append(
                RouteCandidate(
                    start_node=option.start_node,
                    end_node=option.end_node,
                    points=option.points,
                    pixels=option.pixels,
                    component=option.component,
                    score=option.score,
                    evidence=evidence,
                )
            )
        fallback = bool(options) and not accepted
        guarded.append(list(options) if fallback else accepted)
        audit.append(
            {
                "stroke": index + 1,
                "componentId": int(strokes[index]["componentId"]),
                "feature": str(strokes[index].get("feature") or "未知"),
                "candidatePenDownTouchesSibling": start_is_contact,
                "candidatePenUpTouchesSibling": end_is_contact,
                "classicCandidateCount": len(options),
                "acceptedCandidateCount": len(options) if fallback else len(accepted),
                "rejectedJunctionStarts": rejected_starts,
                "rejectedJunctionEnds": rejected_ends,
                "keptCompleteJunctionEnds": kept_complete_junction_ends,
                "fallbackToClassicCandidates": fallback,
            }
        )
    return guarded, audit


def observed_contact(first: RouteCandidate, second: RouteCandidate) -> bool:
    if first.pixels & second.pixels:
        return True
    return polyline_distance(first.points, second.points) <= 2.2


def relative_stroke_signature(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise position inside one leaf, independent of translation/scale."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    combined = np.vstack([first, second])
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    displacement = (first.mean(axis=0) - second.mean(axis=0)) / span
    first_min, first_max = first.min(axis=0), first.max(axis=0)
    second_min, second_max = second.min(axis=0), second.max(axis=0)
    overlap = np.maximum(
        0.0,
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
    ) / span
    return np.r_[displacement, overlap]


def violates_strong_relative_axis_order(
    expected: np.ndarray,
    observed: np.ndarray,
    minimum_expected_separation: float = 0.18,
    minimum_observed_separation: float = 0.035,
) -> bool:
    """Whether a clear candidate left/right or above/below order collapsed.

    The candidate's scale and exact placement are untrusted, but a strong
    internal order such as 八's 撇 being left of its 捺 is structural.  Reversal
    *or collapse onto the same road* is therefore a hard contradiction.
    """
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    for axis in (0, 1):
        reference = float(expected[axis])
        if abs(reference) < minimum_expected_separation:
            continue
        aligned = math.copysign(1.0, reference) * float(observed[axis])
        if aligned < minimum_observed_separation:
            return True
    return False


def route_reuse_ratio(first: RouteCandidate, second: RouteCandidate) -> float:
    """Fraction of the shorter centreline already consumed by another pen."""
    return len(first.pixels & second.pixels) / max(
        1, min(len(first.pixels), len(second.pixels))
    )


def leaf_centroid_order_cost(
    strokes: list[dict],
    routes: list[RouteCandidate],
    horizontal_band: float = 0.10,
) -> tuple[float, dict]:
    """Score the first stroke inside a completed leaf-route hypothesis.

    Candidate semantics already provide the first stroke's coarse direction.
    Among routes of that same direction class, compare centroids in the tight
    bounding box of the *PDF hypothesis*: top-to-bottom first, then left-to-
    right for centroids within ten percent of leaf height.  Candidate absolute
    coordinates never enter this calculation.
    """
    if not strokes or len(strokes) != len(routes):
        return 0.0, {"enabled": False, "reason": "incomplete-leaf"}
    order = RESIDUAL_DECODER.ORDER
    sectors = [
        order.FEATURE_INITIAL_SECTORS.get(
            str(stroke.get("feature") or ""), "unknown"
        )
        for stroke in strokes
    ]
    compatible = [index for index, sector in enumerate(sectors) if sector == sectors[0]]
    if len(compatible) <= 1:
        return 0.0, {
            "enabled": False,
            "reason": "no-same-direction-rival",
            "firstSector": sectors[0],
        }
    leaf_points = np.vstack([route.points for route in routes])
    leaf_min = leaf_points.min(axis=0)
    leaf_span = np.maximum(leaf_points.max(axis=0) - leaf_min, 1.0)
    centroids = np.asarray(
        [(route.points.mean(axis=0) - leaf_min) / leaf_span for route in routes],
        dtype=float,
    )
    top = min(float(centroids[index, 1]) for index in compatible)
    same_row = [
        index
        for index in compatible
        if float(centroids[index, 1]) <= top + horizontal_band
    ]
    preferred = min(
        same_row,
        key=lambda index: (
            float(centroids[index, 0]),
            float(centroids[index, 1]),
        ),
    )
    mismatch = preferred != 0
    # This is deliberately soft.  Stroke topology and normative order remain
    # hard constraints; centroid order only separates otherwise plausible
    # whole-leaf universes.
    cost = 1.0 if mismatch else 0.0
    return cost, {
        "enabled": True,
        "horizontalBand": horizontal_band,
        "firstSector": sectors[0],
        "compatibleStrokes": [index + 1 for index in compatible],
        "preferredStroke": preferred + 1,
        "centroids": [
            [round(float(point[0]), 5), round(float(point[1]), 5)]
            for point in centroids
        ],
        "mismatch": mismatch,
    }


def prune_open_leaf_beam(
    states: list[dict],
    width: int,
    first_stroke_index: int | None = None,
) -> list[dict]:
    """Prune without erasing every universe of an unfinished leaf.

    Until all strokes of the current leaf have been selected, its centroid,
    complete contact matrix, and coverage cannot be evaluated.  Preserve the
    best continuation for every surviving first-stroke route hypothesis, then
    fill the remaining beam slots globally by score.
    """
    ordered = sorted(states, key=lambda item: item["score"])
    if first_stroke_index is None:
        return ordered[:width]
    representatives: dict[int, dict] = {}
    for state in ordered:
        ranks = state.get("routeRanks", [])
        if first_stroke_index >= len(ranks):
            continue
        representatives.setdefault(int(ranks[first_stroke_index]), state)
    selected = sorted(representatives.values(), key=lambda item: item["score"])[
        :width
    ]
    selected_ids = {id(state) for state in selected}
    for state in ordered:
        if len(selected) >= width:
            break
        if id(state) in selected_ids:
            continue
        selected.append(state)
    return sorted(selected, key=lambda item: item["score"])


def grass_diagonal_recovery_needed(
    strokes: list[dict],
    routes: list[RouteCandidate],
) -> list[tuple[int, int]]:
    grass_keys = broken_grass_head_keys(strokes)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        key = (int(stroke.get("componentId", -1)), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(index)
    failed = []
    for key in grass_keys:
        indices = groups[key]
        cost, _evidence = leaf_centroid_order_cost(
            [strokes[index] for index in indices],
            [routes[index] for index in indices],
        )
        if cost > 0:
            failed.append(key)
    return sorted(failed)


def choose_routes(
    strokes: list[dict],
    ranked: list[list[RouteCandidate]],
    total_skeleton_pixels: int,
    coverage_weight: float = 12.0,
    leaf_centroid_weight: float = 0.25,
) -> tuple[list[RouteCandidate], dict]:
    expected_contact = stroke_contact_matrix(strokes)
    parent = list(range(len(strokes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    group_keys = [
        (int(stroke.get("componentId", -1)), int(stroke.get("occurrence", 0)))
        for stroke in strokes
    ]
    group_members: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(group_keys):
        group_members.setdefault(key, []).append(index)
    group_last_index = {key: indices[-1] for key, indices in group_members.items()}
    diagonal_seed_leaves = diagonal_front_leaf_keys(strokes, expected_contact)
    group_centres = {
        key: np.vstack([strokes[index]["points"] for index in indices]).mean(axis=0)
        for key, indices in group_members.items()
    }
    sibling_buckets: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for key, indices in group_members.items():
        hierarchy = strokes[indices[0]].get("hierarchy") or []
        if len(hierarchy) < 2:
            continue
        parent_info = hierarchy[1]
        operator = str(parent_info.get("label", ""))
        if operator not in {"⿰", "⿱", "⿲", "⿳"}:
            continue
        sibling_buckets.setdefault((int(parent_info["id"]), operator), []).append(key)
    sibling_predecessor: dict[int, tuple[tuple[int, int], int]] = {}
    for (_parent, operator), keys in sibling_buckets.items():
        axis = 0 if operator in {"⿰", "⿲"} else 1
        keys.sort(key=lambda key: group_centres[key][axis])
        for previous_key, current_key in zip(keys, keys[1:]):
            sibling_predecessor[group_members[current_key][0]] = previous_key, axis
    route_bounds = np.vstack(
        [point for options in ranked for option in options for point in option.points]
    )
    route_spans = np.maximum(route_bounds.max(axis=0) - route_bounds.min(axis=0), 1.0)
    for left in range(len(strokes)):
        for right in range(left):
            if group_keys[left] == group_keys[right] and expected_contact[left, right]:
                union(left, right)
    expected_signatures = {}
    for current in range(len(strokes)):
        for previous in range(current):
            if expected_contact[current, previous]:
                expected_signatures[current, previous] = contact_signature(
                    strokes[current]["points"], strokes[previous]["points"]
                )[:2]
    relation_cache: dict[
        tuple[int, int, int, int],
        tuple[float, float, float, list[str], bool, bool],
    ] = {}
    for current_index, current_options in enumerate(ranked):
        for previous_index in range(current_index):
            expected = bool(expected_contact[current_index, previous_index])
            for current_rank, current_option in enumerate(current_options):
                for previous_rank, previous_option in enumerate(ranked[previous_index]):
                    relation_cost = 0.0
                    contact_role_cost = 0.0
                    relative_position_cost = 0.0
                    notes = []
                    observed = observed_contact(current_option, previous_option)
                    same_leaf = (
                        group_keys[current_index][0] >= 0
                        and group_keys[current_index] == group_keys[previous_index]
                    )
                    hard_contact_violation = same_leaf and expected != observed
                    hard_relative_order_violation = False
                    if expected and not observed:
                        relation_cost += 0.25
                        notes.append(f"missing-contact-{previous_index + 1}")
                        if same_leaf:
                            notes.append(f"required-contact-{previous_index + 1}")
                        if (
                            group_keys[current_index] == group_keys[previous_index]
                            and current_option.evidence.get("learnedScoringEnabled")
                            and current_option.evidence.get("learnedTouchesEarlierStroke")
                            and float(
                                current_option.evidence.get(
                                    "learnedTouchesEarlierConfidence", 0.0
                                )
                            )
                            >= 0.85
                        ):
                            relation_cost += 1.25
                            notes.append(
                                f"verified-same-leaf-contact-{previous_index + 1}"
                            )
                    elif observed and not expected:
                        relation_cost += 0.15
                        notes.append(f"extra-contact-{previous_index + 1}")
                        if same_leaf:
                            notes.append(f"forbidden-contact-{previous_index + 1}")
                    elif expected and observed:
                        expected_current, expected_previous = expected_signatures[
                            current_index, previous_index
                        ]
                        observed_current, observed_previous, _ = contact_signature(
                            current_option.points, previous_option.points
                        )
                        mismatch = abs(observed_current - expected_current) + abs(
                            observed_previous - expected_previous
                        )
                        contact_role_cost += (2.0 if same_leaf else 0.35) * mismatch
                        if mismatch > 0.16:
                            notes.append(
                                f"contact-role-{previous_index + 1}:"
                                f"{observed_current:.2f}/{observed_previous:.2f}"
                                f"!={expected_current:.2f}/{expected_previous:.2f}"
                            )
                    if same_leaf:
                        expected_relative = relative_stroke_signature(
                            strokes[current_index]["points"],
                            strokes[previous_index]["points"],
                        )
                        observed_relative = relative_stroke_signature(
                            current_option.points,
                            previous_option.points,
                        )
                        relative_mismatch = float(
                            np.mean(np.abs(expected_relative - observed_relative))
                        )
                        relative_position_cost = 1.2 * relative_mismatch
                        # A clear left/right or above/below order is a hard
                        # invariant only for disjoint sibling strokes (for
                        # example 八's 撇/捺).  Intersecting strokes such as
                        # 扌's 横/竖钩 can exchange a great deal of relative
                        # centroid mass across a source glyph; treating that
                        # secondary cue as hard incorrectly deletes the true
                        # topology.  Their order still contributes the soft
                        # relative-position cost above.
                        hard_relative_order_violation = (
                            not expected
                            and violates_strong_relative_axis_order(
                                expected_relative,
                                observed_relative,
                            )
                        )
                        if hard_relative_order_violation:
                            notes.append(
                                f"hard-relative-axis-{previous_index + 1}"
                            )
                        if relative_mismatch > 0.28:
                            notes.append(
                                f"same-leaf-relative-position-{previous_index + 1}:"
                                f"{relative_mismatch:.2f}"
                            )
                    if (
                        find(current_index) == find(previous_index)
                        and current_option.component != previous_option.component
                    ):
                        relation_cost += 2.0
                        notes.append(f"same-leaf-split-island-{previous_index + 1}")
                    relation_cache[
                        current_index, current_rank, previous_index, previous_rank
                    ] = (
                        relation_cost,
                        contact_role_cost,
                        relative_position_cost,
                        notes,
                        hard_contact_violation,
                        hard_relative_order_violation,
                    )

    beam = [
        {
            "score": 0.0,
            "routes": [],
            "routeRanks": [],
            "occupied": frozenset(),
            "steps": [],
            "usedHardContactFallback": False,
            "usedHardGeometryFallback": False,
            "usedFutureFeasibilityFallback": False,
        }
    ]

    def remaining_routes_feasible(
        next_index: int,
        selected_routes: list[RouteCandidate],
    ) -> bool:
        """Cheap look-ahead: every later pen must retain one legal road.

        This is deliberately a necessary, not sufficient, test.  It catches a
        current stroke stealing the only road of a later stroke without
        recursively exploding all future combinations.  Pairwise contact and
        relative-position constraints are *not* evaluated here: before the
        intervening strokes have been chosen, those constraints are not a
        sound reason to prune a partial universe.  The outer beam evaluates
        them exactly when both routes are present.
        """
        for future_index in range(next_index, len(ranked)):
            has_legal_option = False
            for future_option in ranked[future_index]:
                illegal = any(
                    route_reuse_ratio(future_option, previous_route) >= 0.72
                    for previous_route in selected_routes
                )
                if illegal:
                    continue
                has_legal_option = True
                break
            if not has_legal_option:
                return False
        return True

    for index, options in enumerate(ranked):
        next_beam = []
        hard_fallback_beam = []
        first_stroke_in_leaf = group_members[group_keys[index]][0] == index
        for state in beam:
            # The candidate hierarchy has already told us which semantic leaf
            # comes next, and the order guard has already restricted routes to
            # the expected stroke class.  Only within that compatible set do
            # we use the upper-left diagonal front: the first route of a leaf
            # should cover its earliest still-unpainted contact region.  The
            # contact is a seed region, not necessarily the exact pen-down.
            diagonal_front = None
            if first_stroke_in_leaf and group_keys[index] in diagonal_seed_leaves:
                compatible_fronts = []
                for candidate_option in options:
                    if any(
                        route_reuse_ratio(candidate_option, previous_route) >= 0.72
                        for previous_route in state["routes"]
                    ):
                        continue
                    residual_pixels = candidate_option.pixels - state["occupied"]
                    if not residual_pixels:
                        continue
                    compatible_fronts.append(
                        min(float(x + y) for x, y in residual_pixels)
                    )
                if compatible_fronts:
                    diagonal_front = min(compatible_fronts)
            for option_rank, option in enumerate(options):
                score = state["score"] + option.score
                shared = len(state["occupied"] & option.pixels)
                relation_cost = 0.0
                contact_role_cost = 0.0
                relative_position_cost = 0.0
                relation_notes = []
                hard_contact_violation = False
                hard_relative_order_violation = False
                for previous_index, previous_rank in enumerate(state["routeRanks"]):
                    (
                        relation,
                        role,
                        relative,
                        notes,
                        hard_contact,
                        hard_relative,
                    ) = relation_cache[index, option_rank, previous_index, previous_rank]
                    relation_cost += relation
                    contact_role_cost += role
                    relative_position_cost += relative
                    relation_notes.extend(notes)
                    hard_contact_violation = hard_contact_violation or hard_contact
                    hard_relative_order_violation = (
                        hard_relative_order_violation or hard_relative
                    )
                structural_order_cost = 0.0
                predecessor = sibling_predecessor.get(index)
                if predecessor is not None:
                    predecessor_key, axis = predecessor
                    previous_centres = [
                        float(np.mean(route.points[:, axis]))
                        for previous_index, route in enumerate(state["routes"])
                        if group_keys[previous_index] == predecessor_key
                    ]
                    if previous_centres:
                        previous_centre = float(np.mean(previous_centres))
                        current_centre = float(np.mean(option.points[:, axis]))
                        violation = (
                            previous_centre - current_centre
                        ) / float(route_spans[axis])
                        if violation > 0.03:
                            structural_order_cost = 2.0 + 4.0 * violation
                            relation_notes.append(
                                f"sibling-order-{predecessor_key[0]}:{violation:.2f}"
                            )
                diagonal_front_cost = 0.0
                if diagonal_front is not None:
                    residual_pixels = option.pixels - state["occupied"]
                    if residual_pixels:
                        option_front = min(float(x + y) for x, y in residual_pixels)
                        diagonal_front_cost = min(
                            1.0,
                            6.0
                            * max(0.0, option_front - diagonal_front)
                            / max(1.0, float(np.linalg.norm(route_spans))),
                        )
                        if diagonal_front_cost > 0.08:
                            relation_notes.append(
                                f"first-leaf-diagonal-front:{option_front - diagonal_front:.1f}"
                            )
                leaf_centroid_cost = 0.0
                leaf_centroid_evidence = {
                    "enabled": False,
                    "reason": "leaf-not-complete",
                }
                if group_last_index[group_keys[index]] == index:
                    member_indices = group_members[group_keys[index]]
                    member_routes = [
                        option if member_index == index else state["routes"][member_index]
                        for member_index in member_indices
                    ]
                    member_strokes = [strokes[member_index] for member_index in member_indices]
                    raw_leaf_centroid_cost, leaf_centroid_evidence = leaf_centroid_order_cost(
                        member_strokes,
                        member_routes,
                    )
                    leaf_centroid_cost = leaf_centroid_weight * raw_leaf_centroid_cost
                    if leaf_centroid_cost:
                        relation_notes.append(
                            "leaf-centroid-first-stroke:"
                            f"{leaf_centroid_evidence['preferredStroke']}"
                        )
                reuse_cost = max(0, shared - 2) / max(8, len(option.pixels)) * 7.0
                hard_reuse_violation = any(
                    route_reuse_ratio(option, previous_route) >= 0.72
                    for previous_route in state["routes"]
                )
                if hard_reuse_violation:
                    relation_notes.append("duplicate-route-reuse")
                new_pixels = len(option.pixels - state["occupied"])
                new_coverage = new_pixels / max(1, len(option.pixels))
                coverage_reward = coverage_weight * new_pixels / max(1, total_skeleton_pixels)
                score += (
                    relation_cost
                    + contact_role_cost
                    + relative_position_cost
                    + structural_order_cost
                    + diagonal_front_cost
                    + leaf_centroid_cost
                    + reuse_cost
                    - coverage_reward
                )
                if hard_contact_violation:
                    relation_notes.append("same-leaf-contact-fallback")
                selected_routes = state["routes"] + [option]
                selected_ranks = state["routeRanks"] + [option_rank]
                future_feasible = remaining_routes_feasible(
                    index + 1,
                    selected_routes,
                )
                if not future_feasible:
                    relation_notes.append("remaining-route-infeasible")
                hard_geometry_violation = (
                    hard_relative_order_violation or hard_reuse_violation
                )
                hard_violation = (
                    hard_contact_violation
                    or hard_geometry_violation
                    or not future_feasible
                )
                destination = hard_fallback_beam if hard_violation else next_beam
                fallback_penalty = (
                    (1000.0 if hard_contact_violation else 0.0)
                    + (1000.0 if hard_geometry_violation else 0.0)
                    + (1000.0 if not future_feasible else 0.0)
                )
                destination.append(
                    {
                        "score": score + fallback_penalty,
                        "routes": selected_routes,
                        "routeRanks": selected_ranks,
                        "occupied": state["occupied"] | option.pixels,
                        "usedHardContactFallback": state["usedHardContactFallback"]
                        or hard_contact_violation,
                        "usedHardGeometryFallback": state[
                            "usedHardGeometryFallback"
                        ]
                        or hard_geometry_violation,
                        "usedFutureFeasibilityFallback": state[
                            "usedFutureFeasibilityFallback"
                        ]
                        or not future_feasible,
                        "steps": state["steps"]
                        + [
                            {
                                "stroke": index + 1,
                                "localRank": option_rank + 1,
                                "baseScore": round(option.score, 5),
                                "relationCost": round(relation_cost, 5),
                                "contactRoleCost": round(contact_role_cost, 5),
                                "relativePositionCost": round(relative_position_cost, 5),
                                "structuralOrderCost": round(structural_order_cost, 5),
                                "diagonalFrontCost": round(diagonal_front_cost, 5),
                                "leafCentroidOrderCost": round(leaf_centroid_cost, 5),
                                "leafCentroidOrder": leaf_centroid_evidence,
                                "reuseCost": round(reuse_cost, 5),
                                "newCoverage": round(new_coverage, 5),
                                "newSkeletonPixels": new_pixels,
                                "coverageReward": round(coverage_reward, 5),
                                "remainingRoutesFeasible": future_feasible,
                                "notes": relation_notes,
                            }
                        ],
                    }
                )
        if not next_beam:
            next_beam = hard_fallback_beam
        if not next_beam:
            raise RuntimeError(
                f"stroke {index + 1} has no globally feasible route hypothesis"
            )
        current_key = group_keys[index]
        open_leaf_first = (
            group_members[current_key][0]
            if index < group_last_index[current_key]
            else None
        )
        beam = prune_open_leaf_beam(
            next_beam,
            width=350,
            first_stroke_index=open_leaf_first,
        )
    best = beam[0]
    runner_up_margin = beam[1]["score"] - best["score"] if len(beam) > 1 else None
    unexplained_pixels = max(0, int(total_skeleton_pixels - len(best["occupied"])))
    unexplained_ratio = unexplained_pixels / max(1, total_skeleton_pixels)
    review_reasons = []
    if unexplained_ratio >= 0.15:
        review_reasons.append("large-unexplained-skeleton")
    if runner_up_margin is not None and runner_up_margin < 0.005:
        review_reasons.append("near-tied-global-solutions")
    if any(step["localRank"] >= 24 for step in best["steps"]):
        review_reasons.append("candidate-stroke-grammar-mismatch")
    if best["usedHardContactFallback"]:
        review_reasons.append("same-leaf-contact-fallback")
    if best["usedHardGeometryFallback"]:
        review_reasons.append("hard-geometry-fallback")
    if best["usedFutureFeasibilityFallback"]:
        review_reasons.append("future-feasibility-fallback")
    return best["routes"], {
        "score": round(best["score"], 6),
        "beamWidth": 350,
        "coverageWeight": coverage_weight,
        "leafCentroidWeight": leaf_centroid_weight,
        "unexplainedSkeletonPixels": unexplained_pixels,
        "unexplainedSkeletonRatio": round(unexplained_ratio, 6),
        "status": "needs-review" if review_reasons else "safe-candidate",
        "reviewReasons": review_reasons,
        "steps": best["steps"],
        "runnerUpMargin": round(runner_up_margin, 6) if runner_up_margin is not None else None,
    }


def geodesic_owners(
    target: np.ndarray,
    graph: SkeletonGraph,
    routes: list[RouteCandidate],
    maximum_restoration_steps: float = 8.0,
    maximum_radial_width: float = 14.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ink through a skeleton gate before restoring stroke width.

    A direct 2-D flood can leave the end of one stroke and run longitudinally
    into a touching sibling component.  Only a narrow skeleton halo repairs
    medial-axis discretisation around the selected route.  A remote unclaimed
    branch remains unexplained instead of inheriting the nearest stroke's
    colour; a later route must explicitly claim it.  Ink width is then restored
    radially from the owned skeleton roads.
    """
    skeleton_distance = np.full(len(graph.points), np.inf)
    skeleton_owner = np.full(len(graph.points), -1, dtype=np.int16)
    queue = []
    for label, route in enumerate(routes):
        for x, y in route.pixels:
            index = graph.point_index.get((int(y), int(x)))
            if index is None:
                continue
            if skeleton_distance[index] > 0 or label < skeleton_owner[index]:
                skeleton_distance[index] = 0.0
                skeleton_owner[index] = label
                heapq.heappush(queue, (0.0, label, index))
    adjacency = graph_adjacency(graph.matrix)
    route_nodes_by_label = [
        {
            index
            for x, y in route.pixels
            if (index := graph.point_index.get((int(y), int(x)))) is not None
        }
        for route in routes
    ]
    junction_points = np.argwhere(graph.crossing >= 3).astype(float)
    near_junction = (
        cKDTree(junction_points).query(graph.points.astype(float))[0] <= 2.0
        if len(junction_points)
        else np.zeros(len(graph.points), dtype=bool)
    )
    while queue:
        current, label, index = heapq.heappop(queue)
        if current != skeleton_distance[index] or label != skeleton_owner[index]:
            continue
        for neighbour, step in adjacency[index]:
            if (
                bool(near_junction[index])
                and neighbour not in route_nodes_by_label[label]
            ):
                # This is an unchosen universe at a real junction.  Preserve
                # the road as unexplained for another/later stroke instead of
                # letting the current colour turn into it.
                continue
            proposed = current + step
            if proposed > maximum_restoration_steps:
                continue
            if proposed + 1e-9 < skeleton_distance[neighbour] or (
                abs(proposed - skeleton_distance[neighbour]) <= 1e-9
                and label < skeleton_owner[neighbour]
            ):
                skeleton_distance[neighbour] = proposed
                skeleton_owner[neighbour] = label
                heapq.heappush(queue, (proposed, label, neighbour))

    owner = np.full(target.shape, -1, dtype=np.int16)
    distance = np.full(target.shape, np.inf)
    ink_points = np.argwhere(target)
    owned_indices = np.flatnonzero(skeleton_owner >= 0)
    if len(ink_points) and len(owned_indices):
        nearest_distance, nearest_position = cKDTree(
            graph.points[owned_indices].astype(float)
        ).query(ink_points.astype(float))
        nearest_owned_all = owned_indices[np.asarray(nearest_position, dtype=int)]
        local_half_width = cv2.distanceTransform(
            target.astype(np.uint8), cv2.DIST_L2, 5
        )
        nearest_yx = graph.points[nearest_owned_all]
        allowed_width = np.minimum(
            maximum_radial_width,
            local_half_width[nearest_yx[:, 0], nearest_yx[:, 1]] + 2.5,
        )
        valid = nearest_distance <= allowed_width
        valid_ink = ink_points[valid]
        nearest_owned = nearest_owned_all[valid]
        owner[valid_ink[:, 0], valid_ink[:, 1]] = skeleton_owner[nearest_owned]
        distance[valid_ink[:, 0], valid_ink[:, 1]] = nearest_distance[valid]
    return owner, distance


def diffusion_arrivals(
    target: np.ndarray, owner: np.ndarray, routes: list[RouteCandidate]
) -> tuple[np.ndarray, list[dict], float]:
    arrival = np.full(target.shape, np.inf)
    events = []
    offset = 0.0
    for label, route in enumerate(routes):
        path = route.points
        lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        queue = []
        for point, longitudinal in zip(path, lengths):
            x, y = np.rint(point).astype(int)
            if not (0 <= y < target.shape[0] and 0 <= x < target.shape[1]):
                continue
            time = offset + float(longitudinal)
            if time < arrival[y, x]:
                arrival[y, x] = time
                heapq.heappush(queue, (time, y, x))
        while queue:
            current, y, x = heapq.heappop(queue)
            if current != arrival[y, x]:
                continue
            for dy, dx in NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if not (
                    0 <= ny < target.shape[0]
                    and 0 <= nx < target.shape[1]
                    and target[ny, nx]
                    and owner[ny, nx] == label
                ):
                    continue
                # Radial wetting is faster than longitudinal pen travel, but is
                # still stopped by the PDF boundary and the route watershed.
                proposed = current + (0.6 * math.sqrt(2) if dy and dx else 0.6)
                if proposed < arrival[ny, nx]:
                    arrival[ny, nx] = proposed
                    heapq.heappush(queue, (proposed, ny, nx))
        finite = arrival[(owner == label) & np.isfinite(arrival)]
        pen_up_time = offset + float(lengths[-1])
        end_time = max(float(finite.max()), pen_up_time) if len(finite) else pen_up_time
        critical_events = []
        for point_index, point in enumerate(path):
            if point_index in {0, len(path) - 1}:
                critical_events.append(
                    {
                        "kind": "pen-down" if point_index == 0 else "pen-up",
                        "point": [round(float(point[0]), 2), round(float(point[1]), 2)],
                        "time": round(offset + float(lengths[point_index]), 2),
                    }
                )
        events.append(
            {
                "stroke": label + 1,
                "start": critical_events[0],
                "end": critical_events[-1],
                "startTime": round(offset, 2),
                "endTime": round(end_time, 2),
                "stopReason": "候选方向语法已消费完；当前收笔后没有合法主出口",
            }
        )
        offset = end_time + 7.0
    return arrival, events, float(offset)


def vector_angle_error(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-6 or right_norm < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def logical_junction_zones(graph: SkeletonGraph, radius: float = 4.0) -> list[dict]:
    """Fold nearby skeleton junction pixels into one physical contact zone."""
    junctions = [
        (index, point)
        for index, point in enumerate(graph.critical)
        if int(graph.crossing[point]) >= 3
    ]
    if not junctions:
        return []
    raw_xy = np.asarray([(x, y) for _index, (y, x) in junctions], dtype=float)
    parent = list(range(len(junctions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    zone_limit = radius * 2.5
    for left in range(len(junctions)):
        for right in range(left):
            if float(np.linalg.norm(raw_xy[left] - raw_xy[right])) < zone_limit:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(junctions)):
        groups.setdefault(find(index), []).append(index)
    zones = []
    for members in groups.values():
        node_ids = sorted(int(junctions[index][0]) for index in members)
        point = raw_xy[members].mean(axis=0)
        extent = max(float(np.linalg.norm(raw_xy[index] - point)) for index in members)
        zones.append(
            {
                "node": node_ids[0],
                "nodes": node_ids,
                "point": point,
                "suppressionRadius": extent + radius * 0.5,
            }
        )
    return zones


def turn_landmarks(
    points: np.ndarray,
    graph: SkeletonGraph | None = None,
    junction_radius: float = 4.0,
) -> list[float]:
    """Return stable fractions of visually meaningful bends in a directed arc."""
    sampled = resample(points, 72)
    if len(sampled) < 9:
        return []
    # Suppress one-pixel skeleton stair-steps without erasing real corners.
    padded = np.pad(sampled, ((3, 3), (0, 0)), mode="edge")
    sampled = np.vstack([padded[index : index + 7].mean(axis=0) for index in range(len(sampled))])
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    total = float(cumulative[-1])
    if total < 1e-6:
        return []
    candidates = []
    window = 3
    for index in range(window, len(sampled) - window):
        incoming = sampled[index] - sampled[index - window]
        outgoing = sampled[index + window] - sampled[index]
        angle = vector_angle_error(incoming, outgoing)
        if angle >= 24.0:
            candidates.append((angle, index))
    selected = []
    compound_zones = []
    if graph is not None:
        compound_zones = [
            zone
            for zone in logical_junction_zones(graph, junction_radius)
            if len(zone["nodes"]) > 1
        ]
    for angle, index in sorted(candidates, reverse=True):
        if any(
            float(np.linalg.norm(sampled[index] - zone["point"]))
            <= float(zone["suppressionRadius"])
            for zone in compound_zones
        ):
            continue
        if any(abs(index - other) < 7 for _other_angle, other in selected):
            continue
        selected.append((angle, index))
    return sorted(round(float(cumulative[index] / total), 4) for _angle, index in selected)


def critical_node_trace(points: np.ndarray, graph: SkeletonGraph, radius: float = 4.0) -> list[dict]:
    """Describe critical skeleton nodes visited by a directed trajectory."""
    zones = logical_junction_zones(graph, radius)
    if not zones or len(points) < 2:
        return []

    sampled = resample(points, 180)
    critical_xy = np.asarray([zone["point"] for zone in zones], dtype=float)
    tree = cKDTree(critical_xy)
    distances, indices = tree.query(sampled)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    total = max(1e-6, float(cumulative[-1]))
    trace = []
    previous = None
    for sample_index, (distance, critical_index) in enumerate(zip(distances, indices)):
        if float(distance) > radius or int(critical_index) == previous:
            continue
        zone = zones[int(critical_index)]
        previous = int(critical_index)
        point = critical_xy[critical_index]
        trace.append(
            {
                "node": int(zone["node"]),
                "nodes": list(zone["nodes"]),
                "point": [round(float(point[0]), 2), round(float(point[1]), 2)],
                "fraction": round(float(cumulative[sample_index] / total), 4),
            }
        )
    return trace


def landmark_alignment_error(left: list[float], right: list[float]) -> float | None:
    if not left and not right:
        return 0.0
    if not left or not right:
        return None
    count = min(len(left), len(right))
    left_sample = np.interp(np.linspace(0, len(left) - 1, count), np.arange(len(left)), left)
    right_sample = np.interp(np.linspace(0, len(right) - 1, count), np.arange(len(right)), right)
    return float(np.mean(np.abs(left_sample - right_sample)))


def directed_sequence_dtw(
    predicted: np.ndarray,
    expected: np.ndarray,
    samples: int = 96,
) -> dict:
    """Align two pen trajectories by logical order, not wall-clock time.

    DTW may locally wait or advance on either curve, so sampling rate and
    drawing speed are irrelevant.  Its steps remain monotone, which preserves
    pen-down -> turns/contacts -> pen-up ordering and cannot reverse a stroke.
    """
    left = resample(predicted, samples)
    right = resample(expected, samples)
    pair_cost = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    total = np.full((samples + 1, samples + 1), np.inf, dtype=float)
    total[0, 0] = 0.0
    predecessor = np.zeros((samples, samples), dtype=np.uint8)
    for left_index in range(samples):
        for right_index in range(samples):
            choices = (
                total[left_index, right_index],
                total[left_index, right_index + 1],
                total[left_index + 1, right_index],
            )
            step = int(np.argmin(choices))
            total[left_index + 1, right_index + 1] = pair_cost[left_index, right_index] + choices[step]
            predecessor[left_index, right_index] = step
    left_index = right_index = samples - 1
    path = []
    while left_index >= 0 and right_index >= 0:
        path.append((left_index, right_index))
        step = int(predecessor[left_index, right_index])
        if step == 0:
            left_index -= 1
            right_index -= 1
        elif step == 1:
            left_index -= 1
        else:
            right_index -= 1
    path.reverse()
    distances = np.asarray([pair_cost[left_index, right_index] for left_index, right_index in path])
    mapping = []
    for left_index in range(samples):
        matches = [right_index for current_left, right_index in path if current_left == left_index]
        mapping.append(float(np.mean(matches)) / (samples - 1) if matches else left_index / (samples - 1))
    display_indices = np.linspace(0, samples - 1, 33).round().astype(int)
    return {
        "mean": float(distances.mean()),
        "p95": float(np.percentile(distances, 95)),
        "maximum": float(distances.max(initial=0.0)),
        "logicProgressMap": [round(mapping[index], 4) for index in display_indices],
    }


def evaluate_truth(
    annotation_path: Path,
    codepoint: int,
    source: str,
    glyph_id: int,
    candidate: list[dict],
    routes: list[RouteCandidate],
    target: np.ndarray,
    owner: np.ndarray,
    graph: SkeletonGraph,
    canvas: int,
) -> tuple[dict, list[list[list[float]]]]:
    truth, normalized, _corrections = TRANSFER.load_human_annotations(
        annotation_path,
        codepoint=codepoint,
        source=source,
        glyph_id=glyph_id,
        candidate=candidate,
        canvas=canvas,
    )
    start_errors = []
    end_errors = []
    chamfers = []
    sequence_errors = []
    sequence_p95 = []
    sequence_progress_maps = []
    start_angles = []
    end_angles = []
    length_ratios = []
    turn_reports = []
    topology_reports = []
    for predicted, expected in zip(routes, truth):
        start_errors.append(float(np.linalg.norm(predicted.points[0] - expected["points"][0])))
        end_errors.append(float(np.linalg.norm(predicted.points[-1] - expected["points"][-1])))
        predicted_dense = resample(predicted.points, 160)
        expected_dense = resample(expected["points"], 160)
        left = cKDTree(predicted_dense).query(expected_dense)[0].mean()
        right = cKDTree(expected_dense).query(predicted_dense)[0].mean()
        chamfers.append(float((left + right) / 2))
        sequence_alignment = directed_sequence_dtw(predicted.points, expected["points"])
        sequence_errors.append(sequence_alignment["mean"])
        sequence_p95.append(sequence_alignment["p95"])
        sequence_progress_maps.append(sequence_alignment["logicProgressMap"])
        tangent_window = 10
        start_angles.append(
            vector_angle_error(
                predicted_dense[tangent_window] - predicted_dense[0],
                expected_dense[tangent_window] - expected_dense[0],
            )
        )
        end_angles.append(
            vector_angle_error(
                predicted_dense[-1] - predicted_dense[-1 - tangent_window],
                expected_dense[-1] - expected_dense[-1 - tangent_window],
            )
        )
        predicted_length = float(np.linalg.norm(np.diff(predicted_dense, axis=0), axis=1).sum())
        expected_length = float(np.linalg.norm(np.diff(expected_dense, axis=0), axis=1).sum())
        length_ratios.append(predicted_length / max(1e-6, expected_length))
        predicted_turns = turn_landmarks(predicted.points, graph=graph)
        expected_turns = turn_landmarks(expected["points"], graph=graph)
        turn_reports.append(
            {
                "predictedFractions": predicted_turns,
                "truthFractions": expected_turns,
                "countDelta": len(predicted_turns) - len(expected_turns),
                "meanFractionError": (
                    None
                    if (error := landmark_alignment_error(predicted_turns, expected_turns)) is None
                    else round(error, 4)
                ),
            }
        )
        predicted_nodes = critical_node_trace(predicted.points, graph)
        expected_nodes = critical_node_trace(expected["points"], graph)
        predicted_ids = [item["node"] for item in predicted_nodes]
        expected_ids = [item["node"] for item in expected_nodes]
        common = len(set(predicted_ids) & set(expected_ids))
        precision = common / len(set(predicted_ids)) if predicted_ids else (1.0 if not expected_ids else 0.0)
        recall = common / len(set(expected_ids)) if expected_ids else (1.0 if not predicted_ids else 0.0)
        topology_reports.append(
            {
                "predicted": predicted_nodes,
                "truth": expected_nodes,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "exactDirectedSequence": predicted_ids == expected_ids,
            }
        )
    truth_lines = [item["points"] for item in truth]
    truth_masks, _ambiguous, _metrics = TRANSFER.partition_human_truth(
        target, truth_lines, truth, normalized["annotations"], canvas
    )
    component_ids = sorted({int(item["componentId"]) for item in truth})
    stroke_scores = []
    stroke_precision = []
    stroke_recall = []
    for index, truth_mask in enumerate(truth_masks):
        predicted_mask = owner == index
        intersection = int((truth_mask & predicted_mask).sum())
        union = int((truth_mask | predicted_mask).sum())
        stroke_scores.append(intersection / union if union else 1.0)
        stroke_precision.append(intersection / int(predicted_mask.sum()) if predicted_mask.any() else 0.0)
        stroke_recall.append(intersection / int(truth_mask.sum()) if truth_mask.any() else 0.0)
    component_scores = {}
    component_ranges = {}
    for component_id in component_ids:
        truth_mask = np.zeros_like(target)
        predicted_mask = np.zeros_like(target)
        for index, item in enumerate(truth):
            if int(item["componentId"]) == component_id:
                truth_mask |= truth_masks[index]
                predicted_mask |= owner == index
        intersection = int((truth_mask & predicted_mask).sum())
        union = int((truth_mask | predicted_mask).sum())
        component_scores[str(component_id)] = round(intersection / union, 6) if union else 1.0
        component_ranges[str(component_id)] = {
            "precision": round(intersection / int(predicted_mask.sum()), 6) if predicted_mask.any() else 0.0,
            "recall": round(intersection / int(truth_mask.sum()), 6) if truth_mask.any() else 0.0,
            "leakagePercent": round(
                (int(predicted_mask.sum()) - intersection) / max(1, int(predicted_mask.sum())) * 100,
                4,
            ),
            "missedPercent": round(
                (int(truth_mask.sum()) - intersection) / max(1, int(truth_mask.sum())) * 100,
                4,
            ),
        }
    per_stroke = []
    for index, item in enumerate(truth):
        if index >= len(routes):
            per_stroke.append(
                {
                    "stroke": index + 1,
                    "feature": item.get("feature"),
                    "componentId": int(item["componentId"]),
                    "predictedStart": None,
                    "truthStart": [round(float(value), 2) for value in item["points"][0]],
                    "predictedEnd": None,
                    "truthEnd": [round(float(value), 2) for value in item["points"][-1]],
                    "directedSequenceDtwMeanPercent": None,
                    "logicProgressMap": [],
                    "inkPrecision": round(stroke_precision[index], 6),
                    "inkRecall": round(stroke_recall[index], 6),
                    "iou": round(stroke_scores[index], 6),
                    "turns": None,
                    "topology": None,
                    "issues": ["no-feasible-route"],
                }
            )
            continue
        issues = []
        if start_errors[index] / canvas > 0.035:
            issues.append("wrong-start")
        if end_errors[index] / canvas > 0.05:
            issues.append("wrong-end")
        if sequence_errors[index] / canvas > 0.045:
            issues.append("wrong-directed-path")
        if stroke_recall[index] < 0.78:
            issues.append("premature-or-incomplete-coverage")
        if stroke_precision[index] < 0.78:
            issues.append("stroke-boundary-leakage")
        if start_angles[index] > 42.0:
            issues.append("wrong-start-direction")
        turn = turn_reports[index]
        if turn["countDelta"] != 0 or (
            turn["meanFractionError"] is not None and turn["meanFractionError"] > 0.13
        ):
            issues.append("wrong-turn-landmarks")
        if not topology_reports[index]["exactDirectedSequence"]:
            issues.append("wrong-topology-node-sequence")
        per_stroke.append(
            {
                "stroke": index + 1,
                "feature": item.get("feature"),
                "componentId": int(item["componentId"]),
                "startErrorPercent": round(start_errors[index] / canvas * 100, 4),
                "endErrorPercent": round(end_errors[index] / canvas * 100, 4),
                "predictedStart": [round(float(value), 2) for value in routes[index].points[0]],
                "truthStart": [round(float(value), 2) for value in item["points"][0]],
                "predictedEnd": [round(float(value), 2) for value in routes[index].points[-1]],
                "truthEnd": [round(float(value), 2) for value in item["points"][-1]],
                "directedSequenceDtwMeanPercent": round(sequence_errors[index] / canvas * 100, 4),
                "directedSequenceDtwP95Percent": round(sequence_p95[index] / canvas * 100, 4),
                "logicProgressMap": sequence_progress_maps[index],
                "startTangentErrorDegrees": round(start_angles[index], 2),
                "endTangentErrorDegrees": round(end_angles[index], 2),
                "lengthRatio": round(length_ratios[index], 4),
                "inkPrecision": round(stroke_precision[index], 6),
                "inkRecall": round(stroke_recall[index], 6),
                "iou": round(stroke_scores[index], 6),
                "turns": turn_reports[index],
                "topology": topology_reports[index],
                "issues": issues,
            }
        )
    issue_counts: dict[str, int] = {}
    for report in per_stroke:
        for issue in report["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return (
        {
            "truthReadAfterPrediction": True,
            "partialPrediction": len(routes) != len(truth),
            "predictedStrokeCount": len(routes),
            "expectedStrokeCount": len(truth),
            "meanStartErrorPercent": round(float(np.mean(start_errors)) / canvas * 100, 4) if start_errors else 0.0,
            "perStrokeStartErrorPercent": [round(value / canvas * 100, 4) for value in start_errors],
            "meanEndErrorPercent": round(float(np.mean(end_errors)) / canvas * 100, 4) if end_errors else 0.0,
            "perStrokeEndErrorPercent": [round(value / canvas * 100, 4) for value in end_errors],
            "meanCenterlineChamferPercent": round(float(np.mean(chamfers)) / canvas * 100, 4) if chamfers else 0.0,
            "perStrokeCenterlineChamferPercent": [round(value / canvas * 100, 4) for value in chamfers],
            "meanDirectedSequenceDtwPercent": round(float(np.mean(sequence_errors)) / canvas * 100, 4) if sequence_errors else 0.0,
            "trajectoryTimeSemantics": "monotone logical order; local drawing speed is ignored",
            "meanStartTangentErrorDegrees": round(float(np.mean(start_angles)), 2) if start_angles else 0.0,
            "perStrokeIoU": [round(value, 6) for value in stroke_scores],
            "perStrokeInkPrecision": [round(value, 6) for value in stroke_precision],
            "perStrokeInkRecall": [round(value, 6) for value in stroke_recall],
            "strokeMacroIoU": round(float(np.mean(stroke_scores)), 6),
            "componentIoU": component_scores,
            "componentRangeAudit": component_ranges,
            "componentMacroIoU": round(float(np.mean(list(component_scores.values()))), 6),
            "issueCounts": issue_counts,
            "perStrokeTrajectoryAudit": per_stroke,
        },
        [[(points / canvas * 100).round(3).tolist() for points in truth_lines]][0],
    )


def color_for_stroke(stroke: dict) -> str:
    return str(stroke.get("color") or "#0ea5e9")


def outline_glyph_copy(glyph: dict, suffix: str = "-outline") -> dict:
    """Clone exact PDF paths under fresh IDs and make their contours visible."""
    root = ET.fromstring(f"<root>{glyph['definitions']}</root>")
    id_map = {
        element.get("id"): f"{element.get('id')}{suffix}"
        for element in root.iter()
        if element.get("id")
    }
    for element in root.iter():
        old_id = element.get("id")
        if old_id in id_map:
            element.set("id", id_map[old_id])
        for key, value in list(element.attrib.items()):
            local_key = key.rsplit("}", 1)[-1]
            if local_key == "href" and value.startswith("#") and value[1:] in id_map:
                element.set(key, f"#{id_map[value[1:]]}")
            else:
                element.set(
                    key,
                    re.sub(
                        r"url\(#([^)]+)\)",
                        lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                        value,
                    ),
                )
        if element.tag.rsplit("}", 1)[-1] == "path":
            for attribute in ("fill", "stroke", "stroke-width", "stroke-linejoin"):
                element.attrib.pop(attribute, None)
            declarations = []
            for declaration in element.get("style", "").split(";"):
                key = declaration.partition(":")[0].strip()
                if key and key not in {
                    "fill",
                    "stroke",
                    "stroke-width",
                    "stroke-linejoin",
                    "vector-effect",
                }:
                    declarations.append(declaration.strip())
            declarations.extend(
                (
                    "fill:none",
                    "stroke:#0f172a",
                    "stroke-width:.38",
                    "stroke-linejoin:round",
                    "vector-effect:non-scaling-stroke",
                )
            )
            element.set("style", ";".join(declarations))
    attributes = dict(glyph["useAttributes"])
    reference = str(attributes["href"])
    if reference.startswith("#") and reference[1:] in id_map:
        attributes["href"] = f"#{id_map[reference[1:]]}"
    return {
        "definitions": "".join(ET.tostring(child, encoding="unicode") for child in root),
        "useAttributes": attributes,
    }


def glyph_use_markup(glyph: dict) -> str:
    attributes = glyph["useAttributes"]
    return (
        f'<use href="{html.escape(str(attributes["href"]), quote=True)}" '
        f'x="{html.escape(str(attributes.get("x", 0)), quote=True)}" '
        f'y="{html.escape(str(attributes.get("y", 0)), quote=True)}"/>'
    )


def svg_polyline(points: np.ndarray, color: str, width: float = 2.2) -> str:
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def _mask_pixels(mask: np.ndarray) -> list[list[int]]:
    return [[int(x), int(y)] for y, x in np.argwhere(np.asarray(mask, dtype=bool))]


def _half_edge_payload(edge_ids: list[int], directed, graph, contact: np.ndarray) -> list[dict]:
    if directed is None or graph is None:
        return [{"edgeId": int(edge_id)} for edge_id in edge_ids]
    output = []
    for edge_id in edge_ids:
        edge = directed.edges[int(edge_id)]
        points_yx = graph.points[np.asarray(edge.point_indices, dtype=int)]
        outside_contact = any(not contact[int(y), int(x)] for y, x in points_yx)
        output.append(
            {
                "edgeId": int(edge_id),
                "roadId": int(edge.road_id),
                "startSector": edge.start_sector,
                "endSector": edge.end_sector,
                "outsideContact": bool(outside_contact),
                "points": [[int(x), int(y)] for y, x in points_yx],
            }
        )
    return output


def _forbidden_half_edges(route_edge_ids: list[int], directed) -> list[int]:
    if directed is None:
        return []
    forbidden = []
    for incoming_id, outgoing_id in zip(route_edge_ids, route_edge_ids[1:]):
        incoming = directed.edges[int(incoming_id)]
        outgoing = directed.edges[int(outgoing_id)]
        if incoming.end_gate != outgoing.start_gate or incoming.road_id == outgoing.road_id:
            continue
        gate = directed.gates[incoming.end_gate]
        forbidden.extend(
            edge_id
            for edge_id in gate.outgoing
            if directed.edges[edge_id].road_id != incoming.road_id
            and edge_id != outgoing_id
        )
    return list(dict.fromkeys(forbidden))


def residual_review_payload(result, target: np.ndarray, graph=None, directed=None) -> dict:
    """Return sequential, JSON-safe evidence for the review HTML."""
    target = np.asarray(target, dtype=bool)
    unexplained = target.copy()
    reusable = np.zeros(target.shape, dtype=bool)
    owner = np.full(target.shape, -1, dtype=np.int16)
    layers = []
    for index, region in enumerate(result.regions):
        step = dict(result.steps[index]) if index < len(result.steps) else {"stroke": index + 1}
        before = unexplained.copy()
        unexplained &= ~np.asarray(region.mask, dtype=bool)
        reusable |= np.asarray(region.contact_mask, dtype=bool)
        owner[np.asarray(region.mask, dtype=bool)] = index
        selected_ids = [int(item) for item in step.get("routeEdgeIds", ())]
        forbidden_ids = [int(item) for item in step.get("forbiddenHalfEdges", ())]
        if not forbidden_ids:
            forbidden_ids = _forbidden_half_edges(selected_ids, directed)
        layers.append(
            {
                "stroke": index + 1,
                "residualBefore": _mask_pixels(before),
                "residualAfter": _mask_pixels(unexplained),
                "reusableContact": _mask_pixels(reusable),
                "selectedHalfEdges": _half_edge_payload(
                    selected_ids, directed, graph, np.asarray(region.contact_mask, dtype=bool)
                ),
                "forbiddenHalfEdges": _half_edge_payload(
                    forbidden_ids, directed, graph, np.asarray(region.contact_mask, dtype=bool)
                ),
                "evidence": step,
            }
        )
    return {
        "residualLayers": layers,
        "visibleOwner": [
            [int(x), int(y), int(owner[y, x])]
            for y, x in np.argwhere(owner >= 0)
        ],
    }


def residual_review_markup() -> str:
    return '''<!-- animated from D.residualLayers --><div class="residual-controls">
<label><input id="show-residual-before" type="checkbox">残余墨迹（本笔前）</label>
<label><input id="show-residual-after" type="checkbox">残余墨迹（本笔后）</label>
<label><input id="show-reusable-contact" type="checkbox" checked>可复用接触区</label>
<label><input id="show-selected-half-edges" type="checkbox" checked><span style="color:#22c55e">已选半边</span></label>
<label><input id="show-forbidden-half-edges" type="checkbox"><span style="color:#ef4444">禁止出口</span></label>
<label><input id="show-visible-owner" type="checkbox" checked>当前可见归属</label>
</div><h3>落笔候选：接受理由 / 拒绝理由</h3>
<pre id="residual-evidence"></pre><h3>后续笔画可行性</h3><pre id="later-feasibility"></pre>'''


def build_html(
    *,
    record: dict,
    glyph: dict,
    glyph_id: int,
    strokes: list[dict],
    target: np.ndarray,
    skeleton: np.ndarray,
    graph: SkeletonGraph,
    ranked: list[list[RouteCandidate]],
    routes: list[RouteCandidate],
    owner: np.ndarray,
    arrival: np.ndarray,
    events: list[dict],
    maximum_time: float,
    decision: dict,
    evaluation: dict | None,
    truth_lines: list[list[list[float]]],
    residual_result=None,
) -> str:
    canvas = target.shape[0]
    events = [dict(item) for item in events]
    event_offset = float(maximum_time)
    while len(events) < len(strokes):
        stroke_number = len(events) + 1
        events.append(
            {
                "stroke": stroke_number,
                "start": None,
                "end": None,
                "startTime": round(event_offset, 2),
                "endTime": round(event_offset + 4.0, 2),
                "stopReason": "残余解码器未找到满足硬约束的路线；保留 needs-review",
            }
        )
        event_offset += 7.0
    maximum_time = max(1.0, event_offset)
    pixels = []
    for y, x in np.argwhere(target):
        label = int(owner[y, x])
        if label < 0 or not np.isfinite(arrival[y, x]):
            continue
        pixels.append([int(x), int(y), label, round(float(arrival[y, x]), 2)])
    route_payload = [
        [[round(float(x), 2), round(float(y), 2)] for x, y in route.points]
        for route in routes
    ]
    skeleton_payload = [[int(x), int(y)] for y, x in np.argwhere(skeleton)]
    option_payload = []
    for index, options in enumerate(ranked):
        option_payload.append(
            [
                {
                    "rank": rank + 1,
                    "start": [round(float(option.points[0, 0]), 2), round(float(option.points[0, 1]), 2)],
                    "end": [round(float(option.points[-1, 0]), 2), round(float(option.points[-1, 1]), 2)],
                    "score": round(option.score, 5),
                    **option.evidence,
                }
                for rank, option in enumerate(options[:8])
            ]
        )
    candidate_svgs = "".join(
        svg_polyline(stroke["points"], color_for_stroke(stroke), 2.5) for stroke in strokes
    )
    original_use = glyph_use_markup(glyph)
    outline_glyph = outline_glyph_copy(glyph)
    outline_use = glyph_use_markup(outline_glyph)
    colors = [color_for_stroke(stroke) for stroke in strokes]
    metadata = [
        {
            "stroke": index + 1,
            "feature": stroke["feature"],
            "componentId": int(stroke["componentId"]),
            "occurrence": int(stroke.get("occurrence", 0)),
            "color": colors[index],
        }
        for index, stroke in enumerate(strokes)
    ]
    residual_payload = (
        residual_review_payload(
            residual_result,
            target,
            graph=graph,
            directed=RESIDUAL_DECODER.DIRECTED.build_directed_skeleton(graph),
        )
        if residual_result is not None
        else {"residualLayers": [], "visibleOwner": []}
    )
    payload = json.dumps(
        {
            "size": canvas,
            "pixels": pixels,
            "skeleton": skeleton_payload,
            "routes": route_payload,
            "options": option_payload,
            "strokes": metadata,
            "events": events,
            "maximumTime": maximum_time,
            "decision": decision,
            "evaluation": evaluation,
            "truth": truth_lines,
            **residual_payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    evaluation_html = (
        "尚未载入人工真值；本页只展示冻结预测。"
        if evaluation is None
        else (
            f"预测在第 {evaluation['predictedStrokeCount'] + 1} 笔前停止："
            f"只评分 {evaluation['predictedStrokeCount']}/{evaluation['expectedStrokeCount']} 条冻结路线；"
            "余下笔画保留 no-feasible-route。"
            if evaluation.get("partialPrediction")
            else (
                f"真值在预测冻结后才读取：起点平均误差 {evaluation['meanStartErrorPercent']:.2f}% · "
                f"中心线 Chamfer {evaluation['meanCenterlineChamferPercent']:.2f}% · "
                f"逐笔 Macro IoU {evaluation['strokeMacroIoU'] * 100:.1f}% · "
                f"部件 Macro IoU {evaluation['componentMacroIoU'] * 100:.1f}%"
            )
        )
    )
    decision_html = (
        "自动候选可进入后续验收"
        if decision.get("status") == "safe-candidate"
        else "必须人工复核：" + ", ".join(decision.get("reviewReasons", []))
    )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U+{record['unicode']:04X} 有向墨迹扩散</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172033;font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}header{{background:#0f172a;color:white;padding:14px 20px}}h1{{font-size:20px;margin:0 0 5px}}header p{{margin:3px 0;color:#cbd5e1}}main{{padding:14px;display:grid;grid-template-columns:minmax(260px,.7fr) minmax(430px,1.25fr) minmax(300px,.85fr);gap:12px}}section{{background:white;border:1px solid #cbd5e1;border-radius:10px;overflow:hidden}}h2{{font-size:14px;margin:0;padding:9px 11px;background:#f1f5f9}}.body{{padding:10px}}svg{{display:block;width:100%;height:auto}}.pdf-definitions{{position:absolute;width:0;height:0;overflow:hidden}}.stage{{position:relative;aspect-ratio:1;background:white}}.stage svg,.stage canvas{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}#pdf-native-fill{{z-index:1}}#ink-diffusion-canvas{{z-index:2;image-rendering:auto}}#pdf-native-outline{{z-index:3}}.pdf-fill-use{{fill:#334155;opacity:.16}}.pdf-outline-use{{fill:none;stroke:#0f172a;stroke-width:.38;stroke-linejoin:round;vector-effect:non-scaling-stroke;opacity:.58}}.controls{{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;margin-bottom:8px}}button{{padding:5px 10px;border:1px solid #94a3b8;border-radius:6px;background:white;cursor:pointer}}input[type=range]{{width:100%}}#outline-opacity{{width:80px}}.stroke-list{{display:grid;gap:6px}}.stroke{{padding:7px;border:1px solid #cbd5e1;border-left:7px solid var(--c);border-radius:6px;cursor:pointer}}.stroke.active{{outline:3px solid #38bdf8}}pre{{white-space:pre-wrap;max-height:310px;overflow:auto;font-size:11px;background:#f8fafc;padding:8px;border-radius:6px}}.legend,.residual-controls{{display:flex;gap:8px;flex-wrap:wrap;font-size:12px}}.residual-controls{{padding:7px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px}}.badge{{padding:3px 7px;border-radius:999px;background:#e2e8f0}}.warning{{background:#fef3c7;border:1px solid #f59e0b;padding:8px;border-radius:6px}}.good{{background:#dcfce7;border:1px solid #4ade80;padding:8px;border-radius:6px}}label{{display:inline-flex;gap:5px;align-items:center}}@media(max-width:1150px){{main{{grid-template-columns:1fr}}}}
</style>
<header><h1>U+{record['unicode']:04X} {html.escape(chr(record['unicode']))} · {html.escape(record['source'])} 源 · glyph {glyph_id}</h1><p>主前沿只沿 PDF 骨架有向前进；横向墨迹波只负责填满真实轮廓。candidate 坐标未用于生成 PDF 色块。</p><p>{html.escape(decision_html)}</p><p>{html.escape(evaluation_html)}</p></header>
<main>
<section><h2>Candidate：只读符号化假说</h2><div class="body"><svg viewBox="0 0 100 100">{candidate_svgs}</svg><div class="warning">这里的坐标、大小和占据空间不可信；只读取笔顺、方向、转折、接触关系和叶部件 ID。</div><div class="stroke-list" id="stroke-list"></div></div></section>
<section><h2>PDF 墨迹域上的有向扩散</h2><div class="body"><div class="controls"><button id="play">播放</button><input id="time" type="range" min="0" max="1000" value="0"><output id="clock"></output></div><div class="legend"><label><input id="show-pdf-fill" type="checkbox" checked>PDF 原生实体</label><label><input id="show-pdf-outline" type="checkbox" checked>PDF 原生轮廓</label><label>轮廓透明度 <input id="outline-opacity" type="range" min="10" max="100" value="58"></label><label><input id="show-skeleton" type="checkbox" checked>骨架路网</label><label><input id="show-routes" type="checkbox" checked>已选主路</label><label><input id="show-truth" type="checkbox" checked>人工真值笔尖轨迹（仅验收）</label><span class="badge">Shift＋滚轮＝全页正倒放</span><span class="badge">红点＝模型笔尖</span><span class="badge">青点＝人工笔尖（单调逻辑对齐，非等时）</span><span class="badge">白圈＝候选起点</span></div><svg class="pdf-definitions" aria-hidden="true"><defs>{glyph['definitions']}</defs></svg><div class="stage"><svg id="pdf-native-fill" viewBox="0 0 100 100" aria-label="原生 PDF 实体背景"><g class="pdf-fill-use pdf-source-fit">{original_use}</g></svg><canvas id="ink-diffusion-canvas" width="{canvas}" height="{canvas}"></canvas><svg id="pdf-native-outline" viewBox="0 0 100 100" aria-label="原生 PDF 顶层轮廓"><defs>{outline_glyph['definitions']}</defs><g class="pdf-outline-use pdf-source-fit">{outline_use}</g></svg></div></div></section>
<section><h2>当前定格与路口裁决</h2><div class="body"><div id="state" class="good"></div><h3>起点候选（前八）</h3><pre id="options"></pre>{residual_review_markup()}<h3>全局离散解</h3><pre>{html.escape(json.dumps(decision, ensure_ascii=False, indent=2))}</pre></div></section>
</main>
<script>
const D={payload}; const canvas=document.querySelector('#ink-diffusion-canvas'),ctx=canvas.getContext('2d'); const slider=document.querySelector('#time'); const clock=document.querySelector('#clock'); const requestedTime=new URLSearchParams(location.search).get('time'); if(requestedTime!==null)slider.value=Math.max(0,Math.min(1000,Number(requestedTime))); let playing=false,activeStroke=0,last=performance.now();
const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)]; const colors=D.strokes.map(s=>hex(s.color));
function fitPdfSource(el){{const b=el.getBBox();if(!(b.width>0&&b.height>0))return;const s=Math.min(84/b.width,84/b.height),tx=50-s*(b.x+b.width/2),ty=50-s*(b.y+b.height/2);el.setAttribute('transform',`matrix(${{s}} 0 0 ${{s}} ${{tx}} ${{ty}})`);}}
document.querySelectorAll('.pdf-source-fit').forEach(fitPdfSource);
function stageTime(){{return Number(slider.value)/1000*D.maximumTime}}
function logicTruthFraction(audit,fraction){{const map=audit?.logicProgressMap;if(!map?.length)return fraction;const position=Math.max(0,Math.min(map.length-1,fraction*(map.length-1))),left=Math.floor(position),right=Math.min(map.length-1,left+1),mix=position-left;return map[left]*(1-mix)+map[right]*mix}}
function arcLengthPrefix(points,fraction){{if(!points.length)return[];const clamped=Math.max(0,Math.min(1,fraction));if(clamped<=0)return[points[0]];if(clamped>=1)return points.slice();const lengths=[0];for(let i=1;i<points.length;i++)lengths.push(lengths.at(-1)+Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]));const total=lengths.at(-1);if(total<=1e-9)return[points[0],points.at(-1)];const target=clamped*total,prefix=[points[0]];for(let i=1;i<points.length;i++){{if(lengths[i]<target-1e-9){{prefix.push(points[i]);continue}}const segment=lengths[i]-lengths[i-1],mix=segment<=1e-9?1:(target-lengths[i-1])/segment;prefix.push([points[i-1][0]+(points[i][0]-points[i-1][0])*mix,points[i-1][1]+(points[i][1]-points[i-1][1])*mix]);break}}return prefix}}
function paintPixels(img,pixels,r,g,b,a){{for(const [x,y] of pixels){{const i=(y*D.size+x)*4;img.data[i]=r;img.data[i+1]=g;img.data[i+2]=b;img.data[i+3]=a}}}}
function drawHalfEdges(edges,color,width){{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash([]);for(const edge of edges){{if(!edge.points?.length)continue;ctx.beginPath();edge.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke()}}}}
function render(){{const t=stageTime(),img=ctx.createImageData(D.size,D.size),currentEvent=D.events.find(e=>t>=e.startTime&&t<=e.endTime)||D.events.find(e=>t<e.startTime)||D.events.at(-1),layerIndex=Math.max(0,(currentEvent?.stroke||1)-1),layer=D.residualLayers?.[layerIndex];document.querySelector('#pdf-native-fill').style.display=document.querySelector('#show-pdf-fill').checked?'block':'none';document.querySelector('#pdf-native-outline').style.display=document.querySelector('#show-pdf-outline').checked?'block':'none';document.querySelector('.pdf-outline-use').style.opacity=Number(document.querySelector('#outline-opacity').value)/100;if(document.querySelector('#show-skeleton').checked)for(const [x,y] of D.skeleton){{const i=(y*D.size+x)*4;img.data[i]=100;img.data[i+1]=116;img.data[i+2]=139;img.data[i+3]=105}}
if(document.querySelector('#show-visible-owner')?.checked!==false)for(const [x,y,label,at] of D.pixels){{if(at>t)continue;const i=(y*D.size+x)*4,[r,g,b]=colors[label];img.data[i]=r;img.data[i+1]=g;img.data[i+2]=b;img.data[i+3]=220}}if(layer&&document.querySelector('#show-residual-before').checked)paintPixels(img,layer.residualBefore,59,130,246,90);if(layer&&document.querySelector('#show-residual-after').checked)paintPixels(img,layer.residualAfter,168,85,247,105);if(layer&&document.querySelector('#show-reusable-contact').checked)paintPixels(img,layer.reusableContact,250,204,21,180);ctx.putImageData(img,0,0);if(layer&&document.querySelector('#show-selected-half-edges').checked)drawHalfEdges(layer.selectedHalfEdges,'#22c55e',2.2);if(layer&&document.querySelector('#show-forbidden-half-edges').checked)drawHalfEdges(layer.forbiddenHalfEdges,'#ef4444',2.2);
if(document.querySelector('#show-routes').checked){{ctx.lineWidth=1.2;for(let i=0;i<D.routes.length;i++){{const route=D.routes[i],event=D.events[i];if(t<event.startTime)continue;const fraction=Math.min(1,(t-event.startTime)/Math.max(1,event.end.time-event.startTime));const count=Math.max(1,Math.floor(route.length*fraction));ctx.strokeStyle=D.strokes[i].color;ctx.beginPath();for(let j=0;j<count;j++){{const [x,y]=route[j];j?ctx.lineTo(x,y):ctx.moveTo(x,y)}}ctx.stroke();ctx.fillStyle='white';ctx.strokeStyle='#0f172a';ctx.beginPath();ctx.arc(route[0][0],route[0][1],3,0,Math.PI*2);ctx.fill();ctx.stroke();if(fraction<1){{const p=route[Math.min(route.length-1,count-1)];ctx.fillStyle='#ef4444';ctx.beginPath();ctx.arc(p[0],p[1],3.7,0,Math.PI*2);ctx.fill()}}}}
}}
if(document.querySelector('#show-truth').checked&&D.truth.length){{for(let i=0;i<D.truth.length;i++){{const event=D.events[i];if(t<event.startTime)continue;const active=t<=event.endTime,modelFraction=active?Math.max(0,Math.min(1,(t-event.startTime)/Math.max(1,event.end.time-event.startTime))):1,audit=D.evaluation?.perStrokeTrajectoryAudit?.[i],truthFraction=active?logicTruthFraction(audit,modelFraction):1,truth=D.truth[i].map(([x,y])=>[x/100*D.size,y/100*D.size]),prefix=arcLengthPrefix(truth,truthFraction),truthTip=prefix.at(-1);ctx.strokeStyle='#06b6d4';ctx.lineWidth=2.3;ctx.setLineDash([]);ctx.beginPath();prefix.forEach(([x,y],index)=>index?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();if(active){{const model=D.routes[i];if(model?.length){{const modelCount=Math.max(1,Math.floor(model.length*modelFraction)),modelTip=model[Math.min(model.length-1,modelCount-1)];ctx.strokeStyle='rgba(15,23,42,.6)';ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(modelTip[0],modelTip[1]);ctx.lineTo(truthTip[0],truthTip[1]);ctx.stroke();ctx.setLineDash([])}}ctx.fillStyle='#06b6d4';ctx.beginPath();ctx.arc(truthTip[0],truthTip[1],3.7,0,Math.PI*2);ctx.fill()}}}}}}
const event=currentEvent;activeStroke=Math.max(0,event.stroke-1);document.querySelectorAll('.stroke').forEach((e,i)=>e.classList.toggle('active',i===activeStroke));const s=D.strokes[activeStroke],audit=D.evaluation?.perStrokeTrajectoryAudit?.[activeStroke],hasRouteAudit=audit?.directedSequenceDtwMeanPercent!=null,auditText=audit?(hasRouteAudit?`<br>有向序列 DTW ${{audit.directedSequenceDtwMeanPercent.toFixed(2)}}% · 起点 ${{audit.startErrorPercent.toFixed(2)}}% · 终点 ${{audit.endErrorPercent.toFixed(2)}}%<br>覆盖 P/R ${{(audit.inkPrecision*100).toFixed(1)}}% / ${{(audit.inkRecall*100).toFixed(1)}}% · ${{audit.issues.length?audit.issues.join('、'):'未触发轨迹告警'}}`:`<br>${{audit.issues.join('、')}}`):'';document.querySelector('#state').innerHTML=`<b>第 ${{s.stroke}} 笔 · ${{s.feature}} · 部件 ${{s.componentId}}</b><br>主前沿：${{t<event.startTime?'等待落笔':t>=event.endTime?'已收笔':'沿唯一合法主路扩散'}}<br>停止条件：${{event.stopReason}}${{auditText}}`;document.querySelector('#options').textContent=JSON.stringify({{routeOptions:D.options[activeStroke],trajectoryAudit:audit}},null,2);if(document.querySelector('#residual-evidence'))document.querySelector('#residual-evidence').textContent=JSON.stringify(layer?.evidence||{{status:'no residual route for this stroke'}},null,2);if(document.querySelector('#later-feasibility'))document.querySelector('#later-feasibility').textContent=JSON.stringify({{remainingLaterStrokeFeasible:layer?.evidence?.remainingLaterStrokeFeasible??false,unexplainedInk:layer?.evidence?.unexplainedInk??null}},null,2);clock.textContent=`${{t.toFixed(1)}} / ${{D.maximumTime.toFixed(1)}}`;}}
for(const [i,s] of D.strokes.entries()){{const e=document.createElement('div');e.className='stroke';e.style.setProperty('--c',s.color);e.innerHTML=`第 ${{i+1}} 笔 · ${{s.feature}}<br>叶部件 ${{s.componentId}} #${{s.occurrence}}`;e.onclick=()=>{{slider.value=Math.round(D.events[i].startTime/D.maximumTime*1000);render()}};document.querySelector('#stroke-list').append(e)}}
function stopPlayback(){{playing=false;document.querySelector('#play').textContent='播放'}}
function scrubTimeline(event){{if(!event.shiftKey||event.ctrlKey||event.altKey||event.metaKey)return;const raw=event.deltaY!==0?event.deltaY:event.deltaX;if(raw===0)return;event.preventDefault();stopPlayback();const unit=event.deltaMode===1?16:event.deltaMode===2?window.innerHeight:1;const magnitude=Math.min(60,Math.max(4,Math.abs(raw)*unit*.25));slider.value=Math.max(0,Math.min(1000,Number(slider.value)+Math.sign(raw)*magnitude));render()}}
slider.oninput=()=>{{stopPlayback();render()}};document.addEventListener('wheel',scrubTimeline,{{passive:false}});document.querySelectorAll('input[type=checkbox]').forEach(e=>e.onchange=render);document.querySelector('#outline-opacity').oninput=render;document.querySelector('#play').onclick=()=>{{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放';last=performance.now();requestAnimationFrame(tick)}};function tick(now){{if(!playing)return;const next=Math.min(1000,Number(slider.value)+(now-last)/D.maximumTime*160);last=now;slider.value=next;render();if(next>=1000)stopPlayback();else requestAnimationFrame(tick)}}render();
</script></html>'''


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--unicode", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--glyph-id", type=int, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--ordinal-weight", type=float, default=0.12)
    parser.add_argument("--coverage-weight", type=float, default=12.0)
    parser.add_argument("--leaf-centroid-weight", type=float, default=0.25)
    parser.add_argument(
        "--decoder",
        choices=("hybrid", "legacy", "residual"),
        default="hybrid",
    )
    parser.add_argument("--learned-rules", type=Path)
    parser.add_argument("--stroke-order-catalog", type=Path)
    parser.add_argument("--beam-width", type=int, default=350)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    return parser


def residual_decision_payload(result, target: np.ndarray) -> dict:
    leakage = sum(
        int(region.forbidden_leak_mask.sum()) for region in result.regions
    )
    unexplained = int(result.ledger.unexplained.sum())
    return {
        "decoder": "residual",
        "score": round(float(result.best_score), 6),
        "status": result.status,
        "reviewReasons": list(result.review_reasons),
        "runnerUpMargin": (
            None
            if result.runner_up_margin is None
            else round(float(result.runner_up_margin), 6)
        ),
        "steps": list(result.steps),
        "forbiddenBranchLeakagePixels": leakage,
        "residualUnexplainedInkPixels": unexplained,
        "residualUnexplainedInkRatio": round(
            unexplained / max(1, int(np.asarray(target, dtype=bool).sum())), 6
        ),
    }


def main():
    args = build_argument_parser().parse_args()

    codepoint = int(args.unicode.removeprefix("U+").removeprefix("u+"), 16)
    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    row = next(item for item in rows if item["unicode"] == codepoint)
    candidate = TRANSFER.candidate_strokes(row, args.glyph_id)
    records, _sizes = PDF.parse_pdf_cells(args.bbox_cache)
    record = next(
        item for item in records if item["unicode"] == codepoint and item["source"] == args.source
    )
    page_svg = MATCHER.load_pdf_page_svg(args.pdf, record["page"])
    glyph = VECTOR.extract_pdf_glyph(
        page_svg, record["bbox"], f"u{codepoint:04x}-{args.source.lower()}"
    )
    source_mask = MATCHER.render_pdf_vector_cell(
        page_svg, MATCHER.chart_glyph_bbox(record["bbox"]), size=args.canvas
    ) < 224
    target = TRANSFER.normalize_target(source_mask, args.canvas)
    skeleton = TRANSFER.skeletonize(target)
    graph = build_skeleton_graph(skeleton)
    raw_routes = all_routes(graph)
    learned_model = (
        json.loads(args.learned_rules.read_text("utf-8"))
        if args.learned_rules
        else None
    )
    excluded_training_case = f"U+{codepoint:04X}-{args.source}"
    ranked = rank_routes(
        candidate,
        graph,
        raw_routes,
        ordinal_weight=args.ordinal_weight,
        learned_model=learned_model,
        excluded_training_case=excluded_training_case,
        source=args.source,
        enforce_component_labels=args.decoder != "residual",
    )
    normative_catalog = RESIDUAL_DECODER.ORDER.load_normative_catalog(
        args.stroke_order_catalog
    )
    order_guard_audit = None
    topology_guard_audit = None
    if args.decoder == "hybrid":
        ranked, order_guard_audit = apply_stroke_order_guard(
            candidate,
            ranked,
            source=args.source,
            codepoint=codepoint,
            normative_catalog=normative_catalog,
        )
        ranked, topology_guard_audit = apply_stroke_topology_guard(
            candidate,
            ranked,
            graph,
        )
    if any(not options for options in ranked):
        missing = [
            {
                "stroke": index + 1,
                "feature": candidate[index].get("feature"),
                "componentId": candidate[index].get("componentId"),
                "orderGuard": order_guard_audit[index] if order_guard_audit else None,
                "topologyGuard": (
                    topology_guard_audit[index] if topology_guard_audit else None
                ),
            }
            for index, options in enumerate(ranked)
            if not options
        ]
        raise RuntimeError(
            "candidate strokes have no feasible PDF route: "
            + json.dumps(missing, ensure_ascii=False)
        )
    residual_result = None
    if args.decoder == "residual":
        residual_result = RESIDUAL_DECODER.decode_residual_routes(
            target,
            graph,
            candidate,
            ranked,
            source=args.source,
            codepoint=codepoint,
            learned_model=learned_model,
            normative_catalog=normative_catalog,
            beam_width=args.beam_width,
        )
        routes = list(residual_result.routes)
        owner = residual_result.ledger.visible_owner
        decision = residual_decision_payload(residual_result, target)
    else:
        grass_keys = broken_grass_head_keys(candidate)
        baseline_ranked = []
        for stroke, options in zip(candidate, ranked):
            key = (
                int(stroke.get("componentId", -1)),
                int(stroke.get("occurrence", 0)),
            )
            if args.decoder == "hybrid" and key in grass_keys:
                without_scan_seed = [
                    option
                    for option in options
                    if not option.evidence.get("diagonalFrontSeedRoute")
                ]
                baseline_ranked.append(without_scan_seed or options)
            else:
                baseline_ranked.append(options)
        routes, decision = choose_routes(
            candidate,
            baseline_ranked,
            int(skeleton.sum()),
            coverage_weight=args.coverage_weight,
            leaf_centroid_weight=args.leaf_centroid_weight,
        )
        grass_recovery_keys = (
            grass_diagonal_recovery_needed(candidate, routes)
            if args.decoder == "hybrid"
            else []
        )
        if grass_recovery_keys:
            baseline_score = decision["score"]
            routes, decision = choose_routes(
                candidate,
                ranked,
                int(skeleton.sum()),
                coverage_weight=args.coverage_weight,
                leaf_centroid_weight=args.leaf_centroid_weight,
            )
            decision["grassDiagonalFrontRecovery"] = {
                "triggered": True,
                "leafKeys": [list(key) for key in grass_recovery_keys],
                "baselineScore": baseline_score,
                "reason": "completed-grass-head-centroid-order-mismatch",
            }
        else:
            decision["grassDiagonalFrontRecovery"] = {
                "triggered": False,
                "leafKeys": [],
            }
        if args.decoder == "hybrid":
            routes = normalize_selected_pen_paths(
                candidate,
                routes,
                graph,
                source=args.source,
                codepoint=codepoint,
                normative_catalog=normative_catalog,
            )
        owner, _owner_distance = geodesic_owners(target, graph, routes)
        decision["decoder"] = args.decoder
        if order_guard_audit is not None:
            decision["strokeOrderGuard"] = order_guard_audit
        if topology_guard_audit is not None:
            decision["strokeTopologyGuard"] = topology_guard_audit
    arrival, events, maximum_time = diffusion_arrivals(target, owner, routes)

    # The prediction is now frozen. Only evaluation below may read target truth.
    evaluation = None
    truth_lines: list[list[list[float]]] = []
    if args.annotations:
        evaluation, truth_lines = evaluate_truth(
            args.annotations,
            codepoint,
            args.source,
            args.glyph_id,
            candidate,
            routes,
            target,
            owner,
            graph,
            args.canvas,
        )
    decision.update(
        {
            "model": (
                "sequential-directed-residual-ink-v1"
                if args.decoder == "residual"
                else (
                    "classic-directed-skeleton-with-stroke-order-guard-v1"
                    if args.decoder == "hybrid"
                    else "directed-skeleton-front-plus-bounded-radial-wetting-v2"
                )
            ),
            "candidateGeometryUsedForTargetMask": False,
            "candidateOrdinalTieBreakWeight": args.ordinal_weight,
            "beamWidth": args.beam_width if args.decoder == "residual" else None,
            "criticalNodeCount": len(graph.critical),
            "routeHypothesisCount": len(raw_routes),
            "learnedRules": {
                "enabled": learned_model is not None,
                "model": str(args.learned_rules) if args.learned_rules else None,
                "excludedTargetCase": excluded_training_case if learned_model else None,
                "minimumSupportForScoring": 2,
            },
            "unexplainedSkeletonPixels": int(
                skeleton.sum() - len(set().union(*(route.pixels for route in routes)))
            ),
        }
    )
    review_payload = (
        residual_review_payload(
            residual_result,
            target,
            graph=graph,
            directed=RESIDUAL_DECODER.DIRECTED.build_directed_skeleton(graph),
        )
        if residual_result is not None
        else None
    )
    document = build_html(
        record=record,
        glyph=glyph,
        glyph_id=args.glyph_id,
        strokes=candidate,
        target=target,
        skeleton=skeleton,
        graph=graph,
        ranked=ranked,
        routes=routes,
        owner=owner,
        arrival=arrival,
        events=events,
        maximum_time=maximum_time,
        decision=decision,
        evaluation=evaluation,
        truth_lines=truth_lines,
        residual_result=residual_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, "utf-8")
    audit = {
        "unicode": f"U+{codepoint:04X}",
        "character": chr(codepoint),
        "source": args.source,
        "glyphId": args.glyph_id,
        "output": str(args.output),
        "decision": decision,
        "evaluation": evaluation,
        "residualReview": review_payload,
        "strokes": [
            {
                "stroke": index + 1,
                "feature": stroke["feature"],
                "componentId": int(stroke["componentId"]),
                "occurrence": int(stroke.get("occurrence", 0)),
                **(
                    {
                        "selectedHalfEdges": review_payload["residualLayers"][index]["selectedHalfEdges"],
                        "forbiddenHalfEdges": review_payload["residualLayers"][index]["forbiddenHalfEdges"],
                    }
                    if residual_result is not None and index < len(residual_result.regions)
                    else {}
                ),
            }
            for index, stroke in enumerate(candidate)
        ],
    }
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), "utf-8"
        )
    print(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
