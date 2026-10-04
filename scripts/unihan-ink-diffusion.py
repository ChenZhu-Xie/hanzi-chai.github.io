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
    spec.loader.exec_module(module)
    return module


TRANSFER = load_module("unihan-stroke-transfer.py", "unihan_stroke_transfer_diffusion")
MATCHER = TRANSFER.MATCHER
VECTOR = TRANSFER.VECTOR
PDF = TRANSFER.PDF

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
    angles = np.unwrap(tangent_angles(points, 28))
    return float(np.sum(np.abs(np.diff(angles))))


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
    dominant = max(target_components, key=lambda item: np.sum(graph.components == item[0]))[0]
    return {key: dominant for key in groups}


def rank_routes(
    strokes: list[dict], graph: SkeletonGraph, raw_routes: list[dict]
) -> list[list[RouteCandidate]]:
    mapping = component_mapping(strokes, graph)
    candidate_bounds: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    target_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for stroke in strokes:
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        members = [
            item["points"]
            for item in strokes
            if (int(item["componentId"]), int(item.get("occurrence", 0))) == key
        ]
        points = np.vstack(members)
        candidate_bounds[key] = (points.min(axis=0), points.max(axis=0))
    for label in set(mapping.values()):
        points = np.argwhere(graph.components == label)[:, ::-1].astype(float)
        target_bounds[label] = (points.min(axis=0), points.max(axis=0))

    ranked = []
    for stroke in strokes:
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        component = mapping[key]
        expected_turn = total_turn(stroke["points"])
        expected_start = normalized_point(stroke["points"][0], candidate_bounds[key])
        options = []
        for route in raw_routes:
            if route["component"] != component:
                continue
            direction = direction_dtw(stroke["points"], route["points"])
            turn = abs(total_turn(route["points"]) - expected_turn) / math.pi
            route_start = normalized_point(route["points"][0], target_bounds[component])
            ordinal = float(np.linalg.norm(route_start - expected_start))
            # Direction/turn grammar dominates. Ordinal position is a weak tie breaker.
            score = 5.0 * direction + 0.8 * turn + 0.12 * ordinal
            options.append(
                RouteCandidate(
                    start_node=route["startNode"],
                    end_node=route["endNode"],
                    points=route["points"],
                    pixels=route["pixels"],
                    component=component,
                    score=score,
                    evidence={
                        "directionGrammar": round(direction, 5),
                        "turnGrammar": round(turn, 5),
                        "ordinalTieBreak": round(ordinal, 5),
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


def choose_routes(strokes: list[dict], ranked: list[list[RouteCandidate]]) -> tuple[list[RouteCandidate], dict]:
    expected_contact = stroke_contact_matrix(strokes)
    expected_signatures = {}
    for current in range(len(strokes)):
        for previous in range(current):
            if expected_contact[current, previous]:
                expected_signatures[current, previous] = contact_signature(
                    strokes[current]["points"], strokes[previous]["points"]
                )[:2]
    beam = [{"score": 0.0, "routes": [], "occupied": frozenset(), "steps": []}]
    for index, options in enumerate(ranked):
        next_beam = []
        for state in beam:
            for option_rank, option in enumerate(options):
                score = state["score"] + option.score
                shared = len(state["occupied"] & option.pixels)
                relation_cost = 0.0
                contact_role_cost = 0.0
                relation_notes = []
                for previous_index, previous in enumerate(state["routes"]):
                    expected = bool(expected_contact[index, previous_index])
                    observed = observed_contact(option, previous)
                    if expected and not observed:
                        relation_cost += 2.8
                        relation_notes.append(f"missing-contact-{previous_index + 1}")
                    elif observed and not expected:
                        relation_cost += 1.1
                        relation_notes.append(f"extra-contact-{previous_index + 1}")
                    elif expected and observed:
                        expected_current, expected_previous = expected_signatures[index, previous_index]
                        observed_current, observed_previous, _ = contact_signature(
                            option.points, previous.points
                        )
                        mismatch = abs(observed_current - expected_current) + abs(
                            observed_previous - expected_previous
                        )
                        contact_role_cost += 5.0 * mismatch
                        if mismatch > 0.16:
                            relation_notes.append(
                                f"contact-role-{previous_index + 1}:"
                                f"{observed_current:.2f}/{observed_previous:.2f}"
                                f"!={expected_current:.2f}/{expected_previous:.2f}"
                            )
                expected_shared = 5 * sum(expected_contact[index, :index])
                reuse_cost = max(0, shared - expected_shared) / max(8, len(option.pixels)) * 7.0
                new_coverage = len(option.pixels - state["occupied"]) / max(1, len(option.pixels))
                score += relation_cost + contact_role_cost + reuse_cost - 0.45 * new_coverage
                next_beam.append(
                    {
                        "score": score,
                        "routes": state["routes"] + [option],
                        "occupied": state["occupied"] | option.pixels,
                        "steps": state["steps"]
                        + [
                            {
                                "stroke": index + 1,
                                "localRank": option_rank + 1,
                                "baseScore": round(option.score, 5),
                                "relationCost": round(relation_cost, 5),
                                "contactRoleCost": round(contact_role_cost, 5),
                                "reuseCost": round(reuse_cost, 5),
                                "newCoverage": round(new_coverage, 5),
                                "notes": relation_notes,
                            }
                        ],
                    }
                )
        next_beam.sort(key=lambda item: item["score"])
        beam = next_beam[:350]
    best = beam[0]
    return best["routes"], {
        "score": round(best["score"], 6),
        "beamWidth": 350,
        "steps": best["steps"],
        "runnerUpMargin": round(beam[1]["score"] - best["score"], 6) if len(beam) > 1 else None,
    }


def geodesic_owners(target: np.ndarray, routes: list[RouteCandidate]) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape
    distance = np.full(target.shape, np.inf)
    owner = np.full(target.shape, -1, dtype=np.int16)
    queue = []
    for label, route in enumerate(routes):
        for x, y in route.pixels:
            if 0 <= y < height and 0 <= x < width and target[y, x]:
                if distance[y, x] > 0 or label < owner[y, x]:
                    distance[y, x] = 0.0
                    owner[y, x] = label
                    heapq.heappush(queue, (0.0, label, y, x))
    while queue:
        current, label, y, x = heapq.heappop(queue)
        if current != distance[y, x] or label != owner[y, x]:
            continue
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and target[ny, nx]):
                continue
            step = math.sqrt(2) if dy and dx else 1.0
            proposed = current + step
            if proposed + 1e-9 < distance[ny, nx] or (
                abs(proposed - distance[ny, nx]) <= 1e-9 and label < owner[ny, nx]
            ):
                distance[ny, nx] = proposed
                owner[ny, nx] = label
                heapq.heappush(queue, (proposed, label, ny, nx))
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


def evaluate_truth(
    annotation_path: Path,
    codepoint: int,
    source: str,
    glyph_id: int,
    candidate: list[dict],
    routes: list[RouteCandidate],
    target: np.ndarray,
    owner: np.ndarray,
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
    chamfers = []
    for predicted, expected in zip(routes, truth):
        start_errors.append(float(np.linalg.norm(predicted.points[0] - expected["points"][0])))
        predicted_dense = resample(predicted.points, 160)
        expected_dense = resample(expected["points"], 160)
        left = cKDTree(predicted_dense).query(expected_dense)[0].mean()
        right = cKDTree(expected_dense).query(predicted_dense)[0].mean()
        chamfers.append(float((left + right) / 2))
    truth_lines = [item["points"] for item in truth]
    truth_masks, _ambiguous, _metrics = TRANSFER.partition_human_truth(
        target, truth_lines, truth, normalized["annotations"], canvas
    )
    component_ids = sorted({int(item["componentId"]) for item in truth})
    component_scores = {}
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
    return (
        {
            "truthReadAfterPrediction": True,
            "meanStartErrorPercent": round(float(np.mean(start_errors)) / canvas * 100, 4),
            "perStrokeStartErrorPercent": [round(value / canvas * 100, 4) for value in start_errors],
            "meanCenterlineChamferPercent": round(float(np.mean(chamfers)) / canvas * 100, 4),
            "perStrokeCenterlineChamferPercent": [round(value / canvas * 100, 4) for value in chamfers],
            "componentIoU": component_scores,
            "componentMacroIoU": round(float(np.mean(list(component_scores.values()))), 6),
        },
        [[(points / canvas * 100).round(3).tolist() for points in truth_lines]][0],
    )


def color_for_stroke(stroke: dict) -> str:
    return str(stroke.get("color") or "#0ea5e9")


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
    original_use = (
        f'<use href="{glyph["useAttributes"]["href"]}" '
        f'x="{glyph["useAttributes"].get("x", 0)}" y="{glyph["useAttributes"].get("y", 0)}"/>'
    )
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
            f"部件 Macro IoU {evaluation['componentMacroIoU'] * 100:.1f}%"
        )
    )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U+{record['unicode']:04X} 有向墨迹扩散</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172033;font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}}header{{background:#0f172a;color:white;padding:14px 20px}}h1{{font-size:20px;margin:0 0 5px}}header p{{margin:3px 0;color:#cbd5e1}}main{{padding:14px;display:grid;grid-template-columns:minmax(260px,.7fr) minmax(430px,1.25fr) minmax(300px,.85fr);gap:12px}}section{{background:white;border:1px solid #cbd5e1;border-radius:10px;overflow:hidden}}h2{{font-size:14px;margin:0;padding:9px 11px;background:#f1f5f9}}.body{{padding:10px}}svg{{display:block;width:100%;height:auto}}.stage{{position:relative;aspect-ratio:1;background:white}}.stage svg,.stage canvas{{position:absolute;inset:0;width:100%;height:100%}}.source{{fill:#111827;opacity:.22}}canvas{{image-rendering:auto}}.controls{{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;margin-bottom:8px}}button{{padding:5px 10px;border:1px solid #94a3b8;border-radius:6px;background:white;cursor:pointer}}input[type=range]{{width:100%}}.stroke-list{{display:grid;gap:6px}}.stroke{{padding:7px;border:1px solid #cbd5e1;border-left:7px solid var(--c);border-radius:6px;cursor:pointer}}.stroke.active{{outline:3px solid #38bdf8}}pre{{white-space:pre-wrap;max-height:310px;overflow:auto;font-size:11px;background:#f8fafc;padding:8px;border-radius:6px}}.legend{{display:flex;gap:8px;flex-wrap:wrap;font-size:12px}}.badge{{padding:3px 7px;border-radius:999px;background:#e2e8f0}}.warning{{background:#fef3c7;border:1px solid #f59e0b;padding:8px;border-radius:6px}}.good{{background:#dcfce7;border:1px solid #4ade80;padding:8px;border-radius:6px}}label{{display:inline-flex;gap:5px;align-items:center}}@media(max-width:1150px){{main{{grid-template-columns:1fr}}}}
</style>
<header><h1>U+{record['unicode']:04X} {html.escape(chr(record['unicode']))} · {html.escape(record['source'])} 源 · glyph {glyph_id}</h1><p>主前沿只沿 PDF 骨架有向前进；横向墨迹波只负责填满真实轮廓。candidate 坐标未用于生成 PDF 色块。</p><p>{html.escape(evaluation_html)}</p></header>
<main>
<section><h2>Candidate：只读符号化假说</h2><div class="body"><svg viewBox="0 0 100 100">{candidate_svgs}</svg><div class="warning">这里的坐标、大小和占据空间不可信；只读取笔顺、方向、转折、接触关系和叶部件 ID。</div><div class="stroke-list" id="stroke-list"></div></div></section>
<section><h2>PDF 墨迹域上的有向扩散</h2><div class="body"><div class="controls"><button id="play">播放</button><input id="time" type="range" min="0" max="1000" value="0"><output id="clock"></output></div><div class="legend"><label><input id="show-skeleton" type="checkbox" checked>骨架路网</label><label><input id="show-routes" type="checkbox" checked>已选主路</label><label><input id="show-truth" type="checkbox">人工真值中心线（仅验收）</label><span class="badge">红圈＝当前主前沿</span><span class="badge">白圈＝候选起点</span></div><div class="stage"><svg viewBox="0 0 100 100"><defs>{glyph['definitions']}</defs><g class="source">{original_use}</g></svg><canvas id="canvas" width="{canvas}" height="{canvas}"></canvas></div></div></section>
<section><h2>当前定格与路口裁决</h2><div class="body"><div id="state" class="good"></div><h3>起点候选（前八）</h3><pre id="options"></pre><h3>全局离散解</h3><pre>{html.escape(json.dumps(decision, ensure_ascii=False, indent=2))}</pre></div></section>
</main>
<script>
const D={payload}; const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'); const slider=document.querySelector('#time'); const clock=document.querySelector('#clock'); const requestedTime=new URLSearchParams(location.search).get('time'); if(requestedTime!==null)slider.value=Math.max(0,Math.min(1000,Number(requestedTime))); let playing=false,activeStroke=0,last=performance.now();
const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)]; const colors=D.strokes.map(s=>hex(s.color));
function stageTime(){{return Number(slider.value)/1000*D.maximumTime}}
function render(){{const t=stageTime(),img=ctx.createImageData(D.size,D.size); if(document.querySelector('#show-skeleton').checked)for(const [x,y] of D.skeleton){{const i=(y*D.size+x)*4;img.data[i]=100;img.data[i+1]=116;img.data[i+2]=139;img.data[i+3]=105}}
for(const [x,y,label,at] of D.pixels){{if(at>t)continue;const i=(y*D.size+x)*4,[r,g,b]=colors[label];img.data[i]=r;img.data[i+1]=g;img.data[i+2]=b;img.data[i+3]=220}}ctx.putImageData(img,0,0);
if(document.querySelector('#show-routes').checked){{ctx.lineWidth=1.2;for(let i=0;i<D.routes.length;i++){{const route=D.routes[i],event=D.events[i];if(t<event.startTime)continue;const fraction=Math.min(1,(t-event.startTime)/Math.max(1,event.end.time-event.startTime));const count=Math.max(1,Math.floor(route.length*fraction));ctx.strokeStyle=D.strokes[i].color;ctx.beginPath();for(let j=0;j<count;j++){{const [x,y]=route[j];j?ctx.lineTo(x,y):ctx.moveTo(x,y)}}ctx.stroke();ctx.fillStyle='white';ctx.strokeStyle='#0f172a';ctx.beginPath();ctx.arc(route[0][0],route[0][1],3,0,Math.PI*2);ctx.fill();ctx.stroke();if(fraction<1){{const p=route[Math.min(route.length-1,count-1)];ctx.fillStyle='#ef4444';ctx.beginPath();ctx.arc(p[0],p[1],3.7,0,Math.PI*2);ctx.fill()}}}}
}}
if(document.querySelector('#show-truth').checked&&D.truth.length){{ctx.strokeStyle='#06b6d4';ctx.setLineDash([5,4]);ctx.lineWidth=1.5;for(const route of D.truth){{ctx.beginPath();route.forEach(([x,y],i)=>i?ctx.lineTo(x/100*D.size,y/100*D.size):ctx.moveTo(x/100*D.size,y/100*D.size));ctx.stroke()}}ctx.setLineDash([])}}
const event=D.events.find(e=>t>=e.startTime&&t<=e.endTime)||D.events.find(e=>t<e.startTime)||D.events.at(-1);activeStroke=Math.max(0,event.stroke-1);document.querySelectorAll('.stroke').forEach((e,i)=>e.classList.toggle('active',i===activeStroke));const s=D.strokes[activeStroke];document.querySelector('#state').innerHTML=`<b>第 ${{s.stroke}} 笔 · ${{s.feature}} · 部件 ${{s.componentId}}</b><br>主前沿：${{t<event.startTime?'等待落笔':t>=event.endTime?'已收笔':'沿唯一合法主路扩散'}}<br>停止条件：${{event.stopReason}}`;document.querySelector('#options').textContent=JSON.stringify(D.options[activeStroke],null,2);clock.textContent=`${{t.toFixed(1)}} / ${{D.maximumTime.toFixed(1)}}`;}}
for(const [i,s] of D.strokes.entries()){{const e=document.createElement('div');e.className='stroke';e.style.setProperty('--c',s.color);e.innerHTML=`第 ${{i+1}} 笔 · ${{s.feature}}<br>叶部件 ${{s.componentId}} #${{s.occurrence}}`;e.onclick=()=>{{slider.value=Math.round(D.events[i].startTime/D.maximumTime*1000);render()}};document.querySelector('#stroke-list').append(e)}}
slider.oninput=render;document.querySelectorAll('input[type=checkbox]').forEach(e=>e.onchange=render);document.querySelector('#play').onclick=()=>{{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放';last=performance.now();requestAnimationFrame(tick)}};function tick(now){{if(!playing)return;const next=Math.min(1000,Number(slider.value)+(now-last)/D.maximumTime*160);last=now;slider.value=next;render();if(next>=1000){{playing=false;document.querySelector('#play').textContent='播放'}}else requestAnimationFrame(tick)}}render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--unicode", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--glyph-id", type=int, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    ranked = rank_routes(candidate, graph, raw_routes)
    if any(not options for options in ranked):
        raise RuntimeError("at least one candidate stroke has no feasible PDF route")
    routes, decision = choose_routes(candidate, ranked)
    owner, _owner_distance = geodesic_owners(target, routes)
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
            args.canvas,
        )
    decision.update(
        {
            "model": "directed-skeleton-front-plus-bounded-radial-wetting-v1",
            "candidateGeometryUsedForTargetMask": False,
            "candidateOrdinalTieBreakWeight": 0.12,
            "criticalNodeCount": len(graph.critical),
            "routeHypothesisCount": len(raw_routes),
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": decision,
                "evaluation": evaluation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
