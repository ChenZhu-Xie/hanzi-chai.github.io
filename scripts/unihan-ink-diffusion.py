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
    critical = list(dict.fromkeys(endpoints + junctions))
    _component_count, components = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    return SkeletonGraph(points, point_index, matrix, critical, components, crossing)


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
        epsilon=max(1.5, length * 0.012),
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
        # This is an ordinal structural mapping, not a geometric warp.
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
) -> dict | None:
    if not model:
        return None
    examples = [
        example
        for example in model.get("examplesBySignature", {}).get(signature["key"], [])
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
            learned_model, signatures[stroke_index], excluded_training_case
        )
        options = []
        for route in raw_routes:
            if route["component"] not in allowed_components[key]:
                continue
            direction = direction_dtw(stroke["points"], route["points"])
            turn = abs(total_turn(route["points"]) - expected_turn) / math.pi
            route_start = normalized_point(route["points"][0], target_glyph_bounds)
            ordinal = float(np.linalg.norm(route_start - expected_start))
            route_length_fraction = float(route["length"] / max(1.0, target_diagonal))
            premature_stop = (
                not candidate_end_is_contact[stroke_index]
                and has_forward_continuation(
                    graph, route["points"], adjacency=adjacency
                )
                and route_length_fraction < expected_length_fraction * 0.62
            )
            # Direction/turn grammar dominates. Ordinal position is a weak tie
            # breaker. A free pen-up cannot truncate a visibly continuing road.
            premature_stop_cost = 1.6 if premature_stop else 0.0
            learned_cost, learned_evidence = route_learned_cost(
                route, graph, learned_rule, target_glyph_bounds
            )
            score = (
                5.0 * direction
                + 0.8 * turn
                + ordinal_weight * ordinal
                + premature_stop_cost
                + learned_cost
            )
            options.append(
                RouteCandidate(
                    start_node=route["startNode"],
                    end_node=route["endNode"],
                    points=route["points"],
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
                        **learned_evidence,
                    },
                )
            )
        options.sort(key=lambda item: item.score)
        best_direction = min(item.evidence["directionGrammar"] for item in options)
        # Pen direction is a hard local constraint.  Global contact consistency
        # may choose among compatible roads, but must not turn a horizontal
        # stroke into a diagonal/vertical road merely to gain a contact vote.
        compatible = [
            item
            for item in options
            if item.evidence["directionGrammar"] <= best_direction + 0.22
        ]
        ranked.append(compatible[:48])
    return ranked


def observed_contact(first: RouteCandidate, second: RouteCandidate) -> bool:
    if first.pixels & second.pixels:
        return True
    return polyline_distance(first.points, second.points) <= 2.2


def choose_routes(
    strokes: list[dict],
    ranked: list[list[RouteCandidate]],
    total_skeleton_pixels: int,
    coverage_weight: float = 12.0,
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
    relation_cache: dict[tuple[int, int, int, int], tuple[float, float, list[str]]] = {}
    for current_index, current_options in enumerate(ranked):
        for previous_index in range(current_index):
            expected = bool(expected_contact[current_index, previous_index])
            for current_rank, current_option in enumerate(current_options):
                for previous_rank, previous_option in enumerate(ranked[previous_index]):
                    relation_cost = 0.0
                    contact_role_cost = 0.0
                    notes = []
                    observed = observed_contact(current_option, previous_option)
                    if expected and not observed:
                        relation_cost += 0.25
                        notes.append(f"missing-contact-{previous_index + 1}")
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
                        contact_role_cost += 0.35 * mismatch
                        if mismatch > 0.16:
                            notes.append(
                                f"contact-role-{previous_index + 1}:"
                                f"{observed_current:.2f}/{observed_previous:.2f}"
                                f"!={expected_current:.2f}/{expected_previous:.2f}"
                            )
                    if (
                        find(current_index) == find(previous_index)
                        and current_option.component != previous_option.component
                    ):
                        relation_cost += 2.0
                        notes.append(f"same-leaf-split-island-{previous_index + 1}")
                    relation_cache[
                        current_index, current_rank, previous_index, previous_rank
                    ] = relation_cost, contact_role_cost, notes

    beam = [
        {
            "score": 0.0,
            "routes": [],
            "routeRanks": [],
            "occupied": frozenset(),
            "steps": [],
        }
    ]
    for index, options in enumerate(ranked):
        next_beam = []
        for state in beam:
            for option_rank, option in enumerate(options):
                score = state["score"] + option.score
                shared = len(state["occupied"] & option.pixels)
                relation_cost = 0.0
                contact_role_cost = 0.0
                relation_notes = []
                for previous_index, previous_rank in enumerate(state["routeRanks"]):
                    relation, role, notes = relation_cache[
                        index, option_rank, previous_index, previous_rank
                    ]
                    relation_cost += relation
                    contact_role_cost += role
                    relation_notes.extend(notes)
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
                reuse_cost = max(0, shared - 2) / max(8, len(option.pixels)) * 7.0
                new_pixels = len(option.pixels - state["occupied"])
                new_coverage = new_pixels / max(1, len(option.pixels))
                coverage_reward = coverage_weight * new_pixels / max(1, total_skeleton_pixels)
                score += (
                    relation_cost
                    + contact_role_cost
                    + structural_order_cost
                    + reuse_cost
                    - coverage_reward
                )
                next_beam.append(
                    {
                        "score": score,
                        "routes": state["routes"] + [option],
                        "routeRanks": state["routeRanks"] + [option_rank],
                        "occupied": state["occupied"] | option.pixels,
                        "steps": state["steps"]
                        + [
                            {
                                "stroke": index + 1,
                                "localRank": option_rank + 1,
                                "baseScore": round(option.score, 5),
                                "relationCost": round(relation_cost, 5),
                                "contactRoleCost": round(contact_role_cost, 5),
                                "structuralOrderCost": round(structural_order_cost, 5),
                                "reuseCost": round(reuse_cost, 5),
                                "newCoverage": round(new_coverage, 5),
                                "newSkeletonPixels": new_pixels,
                                "coverageReward": round(coverage_reward, 5),
                                "notes": relation_notes,
                            }
                        ],
                    }
                )
        next_beam.sort(key=lambda item: item["score"])
        beam = next_beam[:350]
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
    return best["routes"], {
        "score": round(best["score"], 6),
        "beamWidth": 350,
        "coverageWeight": coverage_weight,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ink through a skeleton gate before restoring stroke width.

    A direct 2-D flood can leave the end of one stroke and run longitudinally
    into a touching sibling component.  First assign every skeleton road to a
    directed route by graph distance, then give each ink pixel to its nearest
    already-owned road.  Thus radial wetting may recover width but cannot use
    the filled outline as an unapproved turn at a T/cross junction.
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
    while queue:
        current, label, index = heapq.heappop(queue)
        if current != skeleton_distance[index] or label != skeleton_owner[index]:
            continue
        for neighbour, step in adjacency[index]:
            proposed = current + step
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
    if len(ink_points) and len(graph.points):
        nearest_distance, nearest_index = cKDTree(
            graph.points.astype(float)
        ).query(ink_points.astype(float))
        owner[ink_points[:, 0], ink_points[:, 1]] = skeleton_owner[nearest_index]
        distance[ink_points[:, 0], ink_points[:, 1]] = nearest_distance
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


def turn_landmarks(points: np.ndarray) -> list[float]:
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
    for angle, index in sorted(candidates, reverse=True):
        if any(abs(index - other) < 7 for _other_angle, other in selected):
            continue
        selected.append((angle, index))
    return sorted(round(float(cumulative[index] / total), 4) for _angle, index in selected)


def critical_node_trace(points: np.ndarray, graph: SkeletonGraph, radius: float = 4.0) -> list[dict]:
    """Describe critical skeleton nodes visited by a directed trajectory."""
    junctions = [
        (index, point)
        for index, point in enumerate(graph.critical)
        if int(graph.crossing[point]) >= 3
    ]
    if not junctions or len(points) < 2:
        return []
    sampled = resample(points, 180)
    critical_ids = [index for index, _point in junctions]
    critical_xy = np.asarray([(x, y) for _index, (y, x) in junctions], dtype=float)
    tree = cKDTree(critical_xy)
    distances, indices = tree.query(sampled)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    total = max(1e-6, float(cumulative[-1]))
    trace = []
    previous = None
    for sample_index, (distance, critical_index) in enumerate(zip(distances, indices)):
        if float(distance) > radius or int(critical_index) == previous:
            continue
        node_id = critical_ids[int(critical_index)]
        previous = int(critical_index)
        point = critical_xy[critical_index]
        trace.append(
            {
                "node": int(node_id),
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
        predicted_turns = turn_landmarks(predicted.points)
        expected_turns = turn_landmarks(expected["points"])
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
            "meanStartErrorPercent": round(float(np.mean(start_errors)) / canvas * 100, 4),
            "perStrokeStartErrorPercent": [round(value / canvas * 100, 4) for value in start_errors],
            "meanEndErrorPercent": round(float(np.mean(end_errors)) / canvas * 100, 4),
            "perStrokeEndErrorPercent": [round(value / canvas * 100, 4) for value in end_errors],
            "meanCenterlineChamferPercent": round(float(np.mean(chamfers)) / canvas * 100, 4),
            "perStrokeCenterlineChamferPercent": [round(value / canvas * 100, 4) for value in chamfers],
            "meanDirectedSequenceDtwPercent": round(float(np.mean(sequence_errors)) / canvas * 100, 4),
            "trajectoryTimeSemantics": "monotone logical order; local drawing speed is ignored",
            "meanStartTangentErrorDegrees": round(float(np.mean(start_angles)), 2),
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
) -> str:
    canvas = target.shape[0]
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
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    evaluation_html = (
        "尚未载入人工真值；本页只展示冻结预测。"
        if evaluation is None
        else (
            f"真值在预测冻结后才读取：起点平均误差 {evaluation['meanStartErrorPercent']:.2f}% · "
            f"中心线 Chamfer {evaluation['meanCenterlineChamferPercent']:.2f}% · "
            f"逐笔 Macro IoU {evaluation['strokeMacroIoU'] * 100:.1f}% · "
            f"部件 Macro IoU {evaluation['componentMacroIoU'] * 100:.1f}%"
        )
    )
    decision_html = (
        "自动候选可进入后续验收"
        if decision.get("status") == "safe-candidate"
        else "必须人工复核：" + ", ".join(decision.get("reviewReasons", []))
    )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U+{record['unicode']:04X} 有向墨迹扩散</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172033;font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}header{{background:#0f172a;color:white;padding:14px 20px}}h1{{font-size:20px;margin:0 0 5px}}header p{{margin:3px 0;color:#cbd5e1}}main{{padding:14px;display:grid;grid-template-columns:minmax(260px,.7fr) minmax(430px,1.25fr) minmax(300px,.85fr);gap:12px}}section{{background:white;border:1px solid #cbd5e1;border-radius:10px;overflow:hidden}}h2{{font-size:14px;margin:0;padding:9px 11px;background:#f1f5f9}}.body{{padding:10px}}svg{{display:block;width:100%;height:auto}}.pdf-definitions{{position:absolute;width:0;height:0;overflow:hidden}}.stage{{position:relative;aspect-ratio:1;background:white}}.stage svg,.stage canvas{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}#pdf-native-fill{{z-index:1}}#ink-diffusion-canvas{{z-index:2;image-rendering:auto}}#pdf-native-outline{{z-index:3}}.pdf-fill-use{{fill:#334155;opacity:.16}}.pdf-outline-use{{fill:none;stroke:#0f172a;stroke-width:.38;stroke-linejoin:round;vector-effect:non-scaling-stroke;opacity:.58}}.controls{{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;margin-bottom:8px}}button{{padding:5px 10px;border:1px solid #94a3b8;border-radius:6px;background:white;cursor:pointer}}input[type=range]{{width:100%}}#outline-opacity{{width:80px}}.stroke-list{{display:grid;gap:6px}}.stroke{{padding:7px;border:1px solid #cbd5e1;border-left:7px solid var(--c);border-radius:6px;cursor:pointer}}.stroke.active{{outline:3px solid #38bdf8}}pre{{white-space:pre-wrap;max-height:310px;overflow:auto;font-size:11px;background:#f8fafc;padding:8px;border-radius:6px}}.legend{{display:flex;gap:8px;flex-wrap:wrap;font-size:12px}}.badge{{padding:3px 7px;border-radius:999px;background:#e2e8f0}}.warning{{background:#fef3c7;border:1px solid #f59e0b;padding:8px;border-radius:6px}}.good{{background:#dcfce7;border:1px solid #4ade80;padding:8px;border-radius:6px}}label{{display:inline-flex;gap:5px;align-items:center}}@media(max-width:1150px){{main{{grid-template-columns:1fr}}}}
</style>
<header><h1>U+{record['unicode']:04X} {html.escape(chr(record['unicode']))} · {html.escape(record['source'])} 源 · glyph {glyph_id}</h1><p>主前沿只沿 PDF 骨架有向前进；横向墨迹波只负责填满真实轮廓。candidate 坐标未用于生成 PDF 色块。</p><p>{html.escape(decision_html)}</p><p>{html.escape(evaluation_html)}</p></header>
<main>
<section><h2>Candidate：只读符号化假说</h2><div class="body"><svg viewBox="0 0 100 100">{candidate_svgs}</svg><div class="warning">这里的坐标、大小和占据空间不可信；只读取笔顺、方向、转折、接触关系和叶部件 ID。</div><div class="stroke-list" id="stroke-list"></div></div></section>
<section><h2>PDF 墨迹域上的有向扩散</h2><div class="body"><div class="controls"><button id="play">播放</button><input id="time" type="range" min="0" max="1000" value="0"><output id="clock"></output></div><div class="legend"><label><input id="show-pdf-fill" type="checkbox" checked>PDF 原生实体</label><label><input id="show-pdf-outline" type="checkbox" checked>PDF 原生轮廓</label><label>轮廓透明度 <input id="outline-opacity" type="range" min="10" max="100" value="58"></label><label><input id="show-skeleton" type="checkbox" checked>骨架路网</label><label><input id="show-routes" type="checkbox" checked>已选主路</label><label><input id="show-truth" type="checkbox" checked>人工真值笔尖轨迹（仅验收）</label><span class="badge">红点＝模型笔尖</span><span class="badge">青点＝人工笔尖（单调逻辑对齐，非等时）</span><span class="badge">白圈＝候选起点</span></div><svg class="pdf-definitions" aria-hidden="true"><defs>{glyph['definitions']}</defs></svg><div class="stage"><svg id="pdf-native-fill" viewBox="0 0 100 100" aria-label="原生 PDF 实体背景"><g class="pdf-fill-use pdf-source-fit">{original_use}</g></svg><canvas id="ink-diffusion-canvas" width="{canvas}" height="{canvas}"></canvas><svg id="pdf-native-outline" viewBox="0 0 100 100" aria-label="原生 PDF 顶层轮廓"><defs>{outline_glyph['definitions']}</defs><g class="pdf-outline-use pdf-source-fit">{outline_use}</g></svg></div></div></section>
<section><h2>当前定格与路口裁决</h2><div class="body"><div id="state" class="good"></div><h3>起点候选（前八）</h3><pre id="options"></pre><h3>全局离散解</h3><pre>{html.escape(json.dumps(decision, ensure_ascii=False, indent=2))}</pre></div></section>
</main>
<script>
const D={payload}; const canvas=document.querySelector('#ink-diffusion-canvas'),ctx=canvas.getContext('2d'); const slider=document.querySelector('#time'); const clock=document.querySelector('#clock'); const requestedTime=new URLSearchParams(location.search).get('time'); if(requestedTime!==null)slider.value=Math.max(0,Math.min(1000,Number(requestedTime))); let playing=false,activeStroke=0,last=performance.now();
const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)]; const colors=D.strokes.map(s=>hex(s.color));
function fitPdfSource(el){{const b=el.getBBox();if(!(b.width>0&&b.height>0))return;const s=Math.min(84/b.width,84/b.height),tx=50-s*(b.x+b.width/2),ty=50-s*(b.y+b.height/2);el.setAttribute('transform',`matrix(${{s}} 0 0 ${{s}} ${{tx}} ${{ty}})`);}}
document.querySelectorAll('.pdf-source-fit').forEach(fitPdfSource);
function stageTime(){{return Number(slider.value)/1000*D.maximumTime}}
function logicTruthFraction(audit,fraction){{const map=audit?.logicProgressMap;if(!map?.length)return fraction;const position=Math.max(0,Math.min(map.length-1,fraction*(map.length-1))),left=Math.floor(position),right=Math.min(map.length-1,left+1),mix=position-left;return map[left]*(1-mix)+map[right]*mix}}
function arcLengthPrefix(points,fraction){{if(!points.length)return[];const clamped=Math.max(0,Math.min(1,fraction));if(clamped<=0)return[points[0]];if(clamped>=1)return points.slice();const lengths=[0];for(let i=1;i<points.length;i++)lengths.push(lengths.at(-1)+Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]));const total=lengths.at(-1);if(total<=1e-9)return[points[0],points.at(-1)];const target=clamped*total,prefix=[points[0]];for(let i=1;i<points.length;i++){{if(lengths[i]<target-1e-9){{prefix.push(points[i]);continue}}const segment=lengths[i]-lengths[i-1],mix=segment<=1e-9?1:(target-lengths[i-1])/segment;prefix.push([points[i-1][0]+(points[i][0]-points[i-1][0])*mix,points[i-1][1]+(points[i][1]-points[i-1][1])*mix]);break}}return prefix}}
function render(){{const t=stageTime(),img=ctx.createImageData(D.size,D.size);document.querySelector('#pdf-native-fill').style.display=document.querySelector('#show-pdf-fill').checked?'block':'none';document.querySelector('#pdf-native-outline').style.display=document.querySelector('#show-pdf-outline').checked?'block':'none';document.querySelector('.pdf-outline-use').style.opacity=Number(document.querySelector('#outline-opacity').value)/100;if(document.querySelector('#show-skeleton').checked)for(const [x,y] of D.skeleton){{const i=(y*D.size+x)*4;img.data[i]=100;img.data[i+1]=116;img.data[i+2]=139;img.data[i+3]=105}}
for(const [x,y,label,at] of D.pixels){{if(at>t)continue;const i=(y*D.size+x)*4,[r,g,b]=colors[label];img.data[i]=r;img.data[i+1]=g;img.data[i+2]=b;img.data[i+3]=220}}ctx.putImageData(img,0,0);
if(document.querySelector('#show-routes').checked){{ctx.lineWidth=1.2;for(let i=0;i<D.routes.length;i++){{const route=D.routes[i],event=D.events[i];if(t<event.startTime)continue;const fraction=Math.min(1,(t-event.startTime)/Math.max(1,event.end.time-event.startTime));const count=Math.max(1,Math.floor(route.length*fraction));ctx.strokeStyle=D.strokes[i].color;ctx.beginPath();for(let j=0;j<count;j++){{const [x,y]=route[j];j?ctx.lineTo(x,y):ctx.moveTo(x,y)}}ctx.stroke();ctx.fillStyle='white';ctx.strokeStyle='#0f172a';ctx.beginPath();ctx.arc(route[0][0],route[0][1],3,0,Math.PI*2);ctx.fill();ctx.stroke();if(fraction<1){{const p=route[Math.min(route.length-1,count-1)];ctx.fillStyle='#ef4444';ctx.beginPath();ctx.arc(p[0],p[1],3.7,0,Math.PI*2);ctx.fill()}}}}
}}
if(document.querySelector('#show-truth').checked&&D.truth.length){{for(let i=0;i<D.truth.length;i++){{const event=D.events[i];if(t<event.startTime)continue;const active=t<=event.endTime,modelFraction=active?Math.max(0,Math.min(1,(t-event.startTime)/Math.max(1,event.end.time-event.startTime))):1,audit=D.evaluation?.perStrokeTrajectoryAudit?.[i],truthFraction=active?logicTruthFraction(audit,modelFraction):1,truth=D.truth[i].map(([x,y])=>[x/100*D.size,y/100*D.size]),prefix=arcLengthPrefix(truth,truthFraction),truthTip=prefix.at(-1);ctx.strokeStyle='#06b6d4';ctx.lineWidth=2.3;ctx.setLineDash([]);ctx.beginPath();prefix.forEach(([x,y],index)=>index?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();if(active){{const model=D.routes[i],modelCount=Math.max(1,Math.floor(model.length*modelFraction)),modelTip=model[Math.min(model.length-1,modelCount-1)];ctx.strokeStyle='rgba(15,23,42,.6)';ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(modelTip[0],modelTip[1]);ctx.lineTo(truthTip[0],truthTip[1]);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#06b6d4';ctx.beginPath();ctx.arc(truthTip[0],truthTip[1],3.7,0,Math.PI*2);ctx.fill()}}}}}}
const event=D.events.find(e=>t>=e.startTime&&t<=e.endTime)||D.events.find(e=>t<e.startTime)||D.events.at(-1);activeStroke=Math.max(0,event.stroke-1);document.querySelectorAll('.stroke').forEach((e,i)=>e.classList.toggle('active',i===activeStroke));const s=D.strokes[activeStroke],audit=D.evaluation?.perStrokeTrajectoryAudit?.[activeStroke],auditText=audit?`<br>有向序列 DTW ${{audit.directedSequenceDtwMeanPercent.toFixed(2)}}% · 起点 ${{audit.startErrorPercent.toFixed(2)}}% · 终点 ${{audit.endErrorPercent.toFixed(2)}}%<br>覆盖 P/R ${{(audit.inkPrecision*100).toFixed(1)}}% / ${{(audit.inkRecall*100).toFixed(1)}}% · ${{audit.issues.length?audit.issues.join('、'):'未触发轨迹告警'}}`:'';document.querySelector('#state').innerHTML=`<b>第 ${{s.stroke}} 笔 · ${{s.feature}} · 部件 ${{s.componentId}}</b><br>主前沿：${{t<event.startTime?'等待落笔':t>=event.endTime?'已收笔':'沿唯一合法主路扩散'}}<br>停止条件：${{event.stopReason}}${{auditText}}`;document.querySelector('#options').textContent=JSON.stringify({{routeOptions:D.options[activeStroke],trajectoryAudit:audit}},null,2);clock.textContent=`${{t.toFixed(1)}} / ${{D.maximumTime.toFixed(1)}}`;}}
for(const [i,s] of D.strokes.entries()){{const e=document.createElement('div');e.className='stroke';e.style.setProperty('--c',s.color);e.innerHTML=`第 ${{i+1}} 笔 · ${{s.feature}}<br>叶部件 ${{s.componentId}} #${{s.occurrence}}`;e.onclick=()=>{{slider.value=Math.round(D.events[i].startTime/D.maximumTime*1000);render()}};document.querySelector('#stroke-list').append(e)}}
slider.oninput=render;document.querySelectorAll('input[type=checkbox]').forEach(e=>e.onchange=render);document.querySelector('#outline-opacity').oninput=render;document.querySelector('#play').onclick=()=>{{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放';last=performance.now();requestAnimationFrame(tick)}};function tick(now){{if(!playing)return;const next=Math.min(1000,Number(slider.value)+(now-last)/D.maximumTime*160);last=now;slider.value=next;render();if(next>=1000){{playing=false;document.querySelector('#play').textContent='播放'}}else requestAnimationFrame(tick)}}render();
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
    parser.add_argument("--decoder", choices=("legacy", "residual"), default="legacy")
    parser.add_argument("--learned-rules", type=Path)
    parser.add_argument("--stroke-order-catalog", type=Path)
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
    )
    if any(not options for options in ranked):
        raise RuntimeError("at least one candidate stroke has no feasible PDF route")
    if args.decoder == "residual":
        normative_catalog = RESIDUAL_DECODER.ORDER.load_normative_catalog(
            args.stroke_order_catalog
        )
        residual_result = RESIDUAL_DECODER.decode_residual_routes(
            target,
            graph,
            candidate,
            ranked,
            source=args.source,
            codepoint=codepoint,
            learned_model=learned_model,
            normative_catalog=normative_catalog,
        )
        routes = list(residual_result.routes)
        owner = residual_result.ledger.visible_owner
        decision = residual_decision_payload(residual_result, target)
    else:
        routes, decision = choose_routes(
            candidate,
            ranked,
            int(skeleton.sum()),
            coverage_weight=args.coverage_weight,
        )
        owner, _owner_distance = geodesic_owners(target, graph, routes)
        decision["decoder"] = "legacy"
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
                else "directed-skeleton-front-plus-bounded-radial-wetting-v2"
            ),
            "candidateGeometryUsedForTargetMask": False,
            "candidateOrdinalTieBreakWeight": args.ordinal_weight,
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
