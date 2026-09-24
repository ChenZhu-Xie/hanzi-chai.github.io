"""Transfer repository stroke semantics onto a Unicode PDF glyph outline.

This is deliberately template-guided: the repository candidate supplies stroke
order and component ownership.  The PDF outline supplies the target silhouette.
The outline alone cannot uniquely recover overlapping stroke order.
"""

from __future__ import annotations

import argparse
import copy
import html
import importlib.util
import json
import math
import re
import xml.etree.ElementTree as ET
import heapq
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_module("unihan-render-match.py", "unihan_render_match_transfer")
VECTOR = load_module("unihan-render-vector-review.py", "unihan_vector_review_transfer")
PDF = MATCHER.PDF

PATH_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
COMMAND_LENGTHS = {"M": 2, "m": 2, "L": 2, "l": 2, "H": 1, "h": 1, "V": 1, "v": 1, "C": 6, "c": 6, "A": 7, "a": 7}


def _cubic(start: np.ndarray, values: list[float], steps: int) -> list[np.ndarray]:
    relative = np.asarray(values, dtype=float).reshape(3, 2)
    control1, control2, end = start + relative
    return [
        (1 - t) ** 3 * start
        + 3 * (1 - t) ** 2 * t * control1
        + 3 * (1 - t) * t**2 * control2
        + t**3 * end
        for t in np.linspace(0, 1, steps + 1)[1:]
    ]


def _arc(
    start: np.ndarray,
    values: list[float],
    *,
    relative: bool,
    steps: int,
) -> list[np.ndarray]:
    rx, ry, rotation, large_arc, sweep, x, y = values
    end = start + np.array([x, y]) if relative else np.array([x, y], dtype=float)
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or np.allclose(start, end):
        return [end]
    phi = math.radians(rotation % 360)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    delta = (start - end) / 2
    prime = np.array(
        [cos_phi * delta[0] + sin_phi * delta[1], -sin_phi * delta[0] + cos_phi * delta[1]]
    )
    scale = prime[0] ** 2 / rx**2 + prime[1] ** 2 / ry**2
    if scale > 1:
        factor = math.sqrt(scale)
        rx *= factor
        ry *= factor
    sign = -1 if bool(large_arc) == bool(sweep) else 1
    numerator = max(0.0, rx**2 * ry**2 - rx**2 * prime[1] ** 2 - ry**2 * prime[0] ** 2)
    denominator = rx**2 * prime[1] ** 2 + ry**2 * prime[0] ** 2
    coefficient = sign * math.sqrt(numerator / denominator) if denominator else 0
    center_prime = coefficient * np.array([rx * prime[1] / ry, -ry * prime[0] / rx])
    center = np.array(
        [
            cos_phi * center_prime[0] - sin_phi * center_prime[1],
            sin_phi * center_prime[0] + cos_phi * center_prime[1],
        ]
    ) + (start + end) / 2

    def vector_angle(left: np.ndarray, right: np.ndarray) -> float:
        cross = left[0] * right[1] - left[1] * right[0]
        dot = float(np.dot(left, right))
        return math.atan2(cross, dot)

    begin = np.array([(prime[0] - center_prime[0]) / rx, (prime[1] - center_prime[1]) / ry])
    finish = np.array([(-prime[0] - center_prime[0]) / rx, (-prime[1] - center_prime[1]) / ry])
    theta = vector_angle(np.array([1.0, 0.0]), begin)
    delta_theta = vector_angle(begin, finish)
    if not sweep and delta_theta > 0:
        delta_theta -= 2 * math.pi
    elif sweep and delta_theta < 0:
        delta_theta += 2 * math.pi
    count = max(2, round(steps * abs(delta_theta) / (math.pi / 2)))
    output = []
    for value in np.linspace(theta, theta + delta_theta, count + 1)[1:]:
        point = np.array([rx * math.cos(value), ry * math.sin(value)])
        output.append(
            center
            + np.array(
                [cos_phi * point[0] - sin_phi * point[1], sin_phi * point[0] + cos_phi * point[1]]
            )
        )
    return output


def sample_svg_centerline(
    path_data: str, curve_steps: int = 24, *, with_fractions: bool = False
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Sample the small SVG path subset emitted by glyphToSvgMarkup."""
    tokens = PATH_TOKEN.findall(path_data)
    index = 0
    command = None
    current = np.zeros(2, dtype=float)
    subpath = current.copy()
    points: list[np.ndarray] = []
    command_boundaries: list[int] = []
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command in {"Z", "z"}:
                current = subpath.copy()
                points.append(current.copy())
                continue
        if command not in COMMAND_LENGTHS:
            raise ValueError(f"unsupported SVG path command {command!r}")
        length = COMMAND_LENGTHS[command]
        values = [float(value) for value in tokens[index : index + length]]
        if len(values) != length:
            raise ValueError(f"incomplete SVG path command {command}")
        index += length
        relative = command.islower()
        upper = command.upper()
        if upper == "M":
            current = current + values if relative else np.array(values, dtype=float)
            subpath = current.copy()
            points.append(current.copy())
            command = "l" if relative else "L"
        elif upper == "L":
            current = current + values if relative else np.array(values, dtype=float)
            points.append(current.copy())
            command_boundaries.append(len(points) - 1)
        elif upper == "H":
            current = current + [values[0], 0] if relative else np.array([values[0], current[1]])
            points.append(current.copy())
            command_boundaries.append(len(points) - 1)
        elif upper == "V":
            current = current + [0, values[0]] if relative else np.array([current[0], values[0]])
            points.append(current.copy())
            command_boundaries.append(len(points) - 1)
        elif upper == "C":
            if not relative:
                values = [
                    values[0] - current[0],
                    values[1] - current[1],
                    values[2] - current[0],
                    values[3] - current[1],
                    values[4] - current[0],
                    values[5] - current[1],
                ]
            sampled = _cubic(current, values, curve_steps)
            points.extend(sampled)
            current = sampled[-1]
            command_boundaries.append(len(points) - 1)
        elif upper == "A":
            sampled = _arc(current, values, relative=relative, steps=curve_steps)
            points.extend(sampled)
            current = sampled[-1]
            command_boundaries.append(len(points) - 1)
    output = np.asarray(points, dtype=float)
    if not with_fractions:
        return output
    if len(output) < 2:
        return output, []
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(output, axis=0), axis=1))]
    total = float(cumulative[-1])
    fractions = [
        float(cumulative[index] / total)
        for index in command_boundaries[:-1]
        if total > 0 and 0 < index < len(output) - 1
    ]
    return output, fractions


def candidate_strokes(row: dict, glyph_id: int) -> list[dict]:
    by_index = {}
    for leaf in row["candidateLeafSvgs"][str(glyph_id)]:
        root = ET.fromstring(leaf["svg"])
        paths = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "path"]
        if len(paths) != len(leaf["strokeIndices"]):
            raise ValueError(f"leaf {leaf['leafId']} path/stroke count mismatch")
        for stroke_index, path in zip(leaf["strokeIndices"], paths):
            points, bend_fractions = sample_svg_centerline(
                path.attrib["d"], with_fractions=True
            )
            by_index[stroke_index] = {
                "strokeIndex": stroke_index,
                "feature": row["candidates"][str(glyph_id)][stroke_index]["feature"],
                "componentId": leaf["leafId"],
                "familyKey": leaf["familyKey"],
                "color": leaf["color"],
                "points": points,
                "bendFractions": bend_fractions,
            }
    expected = len(row["candidates"][str(glyph_id)])
    if sorted(by_index) != list(range(expected)):
        raise ValueError(f"candidate {glyph_id} has incomplete stroke ownership")
    return [by_index[index] for index in range(expected)]


def normalize_target(binary: np.ndarray, canvas: int, padding: float = 0.08) -> np.ndarray:
    points = np.argwhere(binary)
    if points.size == 0:
        raise ValueError("PDF glyph mask is empty")
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    target = round(canvas * (1 - padding * 2))
    scale = min(target / (right - left), target / (bottom - top))
    width = max(1, round((right - left) * scale))
    height = max(1, round((bottom - top) * scale))
    resized = cv2.resize(
        binary[top:bottom, left:right].astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((canvas, canvas), dtype=bool)
    x = (canvas - width) // 2
    y = (canvas - height) // 2
    output[y : y + height, x : x + width] = resized > 0
    return output


def fit_centerlines(strokes: list[dict], canvas: int, padding: float = 0.08) -> list[np.ndarray]:
    all_points = np.concatenate([stroke["points"] for stroke in strokes])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    extent = maximum - minimum
    target = canvas * (1 - padding * 2)
    positive_extents = extent[extent > 1e-9]
    if not len(positive_extents):
        raise ValueError("candidate centerlines have zero extent")
    scale = min(target / value for value in positive_extents)
    center = (minimum + maximum) / 2
    return [(stroke["points"] - center) * scale + canvas / 2 for stroke in strokes]


def skeletonize(binary: np.ndarray) -> np.ndarray:
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if thinning is not None:
        return thinning(binary.astype(np.uint8) * 255) > 0
    return MATCHER.skeletonize(binary)


def resample_polyline(points: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Densify a median so snapping cannot jump across a long segment unseen."""
    if len(points) < 2:
        return points.copy()
    output = [points[0].astype(float)]
    for start, end in zip(points, points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, math.ceil(distance / step))
        output.extend(start + (end - start) * (index / count) for index in range(1, count + 1))
    return np.asarray(output, dtype=float)


def snap_polyline_coherently(
    line: np.ndarray,
    skeleton_points: np.ndarray,
    tree: cKDTree,
    *,
    max_distance: float,
    neighbors: int = 10,
) -> tuple[np.ndarray, float]:
    """Map one template median to a continuous target-skeleton route.

    A dynamic program balances point proximity with preservation of local
    segment length and direction.  Unlike independent nearest-point snapping,
    it does not freely switch branches at every intersection.
    """
    sampled = resample_polyline(line)
    if not len(sampled):
        return sampled, 0.0
    count = min(neighbors, len(skeleton_points))
    distances, indices = tree.query(sampled, k=count, distance_upper_bound=max_distance)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    candidates: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for point, point_distances, point_indices in zip(sampled, distances, indices):
        valid = np.isfinite(point_distances) & (point_indices < len(skeleton_points))
        if valid.any():
            candidates.append(skeleton_points[point_indices[valid]][:, ::-1].astype(float))
            observed.append(point_distances[valid].astype(float))
        else:
            candidates.append(point[None, :].astype(float))
            observed.append(np.array([max_distance], dtype=float))

    costs = observed[0] ** 2
    backtrack: list[np.ndarray] = []
    for position in range(1, len(sampled)):
        previous = candidates[position - 1]
        current = candidates[position]
        template_delta = sampled[position] - sampled[position - 1]
        template_length = max(1e-6, float(np.linalg.norm(template_delta)))
        transitions = current[:, None, :] - previous[None, :, :]
        transition_lengths = np.linalg.norm(transitions, axis=2)
        length_cost = (transition_lengths - template_length) ** 2 * 1.25
        direction_dot = np.sum(transitions * template_delta, axis=2) / (
            np.maximum(transition_lengths, 1e-6) * template_length
        )
        direction_cost = np.maximum(0.0, 0.35 - direction_dot) ** 2 * 18.0
        previous_offsets = previous - sampled[position - 1]
        current_offsets = current - sampled[position]
        warp_cost = np.sum(
            (current_offsets[:, None, :] - previous_offsets[None, :, :]) ** 2,
            axis=2,
        ) * 2.0
        total = costs[None, :] + length_cost + direction_cost + warp_cost
        parents = np.argmin(total, axis=1)
        costs = observed[position] ** 2 + total[np.arange(len(current)), parents]
        backtrack.append(parents)

    selected = [int(np.argmin(costs))]
    for parents in reversed(backtrack):
        selected.append(int(parents[selected[-1]]))
    selected.reverse()
    snapped = np.asarray([options[index] for options, index in zip(candidates, selected)])
    keep = np.r_[True, np.any(np.diff(snapped, axis=0) != 0, axis=1)]
    return snapped[keep], float(max((values.min() for values in observed), default=0.0))


def snap_centerlines(
    centerlines: list[np.ndarray], target: np.ndarray, max_distance: float
) -> tuple[list[np.ndarray], list[float]]:
    skeleton = skeletonize(target)
    skeleton_points = np.argwhere(skeleton)
    if not len(skeleton_points):
        return centerlines, [max_distance for _line in centerlines]
    tree = cKDTree(skeleton_points[:, ::-1])
    output = []
    maxima = []
    for line in centerlines:
        snapped, maximum = snap_polyline_coherently(
            line, skeleton_points, tree, max_distance=max_distance
        )
        output.append(snapped)
        maxima.append(maximum)
    return output, maxima


def candidate_alignment_metrics(
    strokes: list[dict],
    target: np.ndarray,
    canvas: int,
    focus_component_ids: set[int] | None = None,
) -> dict:
    """Score a candidate before target partitioning hides candidate mistakes."""
    fitted = fit_centerlines(strokes, canvas)
    skeleton = skeletonize(target)
    skeleton_yx = np.argwhere(skeleton)
    if not len(skeleton_yx):
        raise ValueError("target skeleton is empty")
    selected = [
        line
        for stroke, line in zip(strokes, fitted)
        if not focus_component_ids or stroke["componentId"] in focus_component_ids
    ]
    if not selected:
        selected = fitted
    focus_points = np.concatenate([resample_polyline(line) for line in selected])
    if focus_component_ids:
        padding = canvas * 0.055
        minimum = focus_points.min(axis=0) - padding
        maximum = focus_points.max(axis=0) + padding
        in_window = (
            (skeleton_yx[:, 1] >= minimum[0])
            & (skeleton_yx[:, 1] <= maximum[0])
            & (skeleton_yx[:, 0] >= minimum[1])
            & (skeleton_yx[:, 0] <= maximum[1])
        )
        if in_window.any():
            skeleton_yx = skeleton_yx[in_window]
    skeleton_xy = skeleton_yx[:, ::-1].astype(float)
    tree = cKDTree(skeleton_xy)
    snapped_lines = []
    forward_distances = []
    length_distortions = []
    max_distance = canvas * 0.08
    for line in selected:
        dense = resample_polyline(line)
        distances, _indices = tree.query(dense)
        forward_distances.extend(distances.tolist())
        snapped, _maximum = snap_polyline_coherently(
            line, skeleton_yx, tree, max_distance=max_distance
        )
        snapped_lines.append(snapped)
        source_length = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
        target_length = float(np.linalg.norm(np.diff(snapped, axis=0), axis=1).sum())
        length_distortions.append(abs(math.log((target_length + 1) / (source_length + 1))))
    transferred = np.concatenate([line for line in snapped_lines if len(line)])
    reverse_distances, _indices = cKDTree(transferred).query(skeleton_xy)
    scale = 100 / canvas
    forward_mean = float(np.mean(forward_distances)) * scale
    forward_p95 = float(np.percentile(forward_distances, 95)) * scale
    reverse_mean = float(np.mean(reverse_distances)) * scale
    reverse_p95 = float(np.percentile(reverse_distances, 95)) * scale
    length_mean = float(np.mean(length_distortions))
    score = (
        forward_mean * 0.28
        + forward_p95 * 0.12
        + reverse_mean * 0.34
        + reverse_p95 * 0.16
        + length_mean * 5.0
    )
    return {
        "score": round(score, 6),
        "templateToTargetMean": round(forward_mean, 6),
        "templateToTargetP95": round(forward_p95, 6),
        "targetToTemplateMean": round(reverse_mean, 6),
        "targetToTemplateP95": round(reverse_p95, 6),
        "meanLengthLogDistortion": round(length_mean, 6),
        "focusStrokeCount": len(selected),
        "focusSkeletonPoints": len(skeleton_xy),
        "snapped": snapped_lines,
    }


def _line_seed(shape: tuple[int, int], points: np.ndarray, width: int = 3) -> np.ndarray:
    seed = np.zeros(shape, dtype=np.uint8)
    rounded = np.rint(points).astype(np.int32)
    if len(rounded) == 1:
        cv2.circle(seed, tuple(rounded[0]), width, 1, -1)
    elif len(rounded) > 1:
        cv2.polylines(seed, [rounded], False, 1, width, cv2.LINE_AA)
    return seed > 0


def partition_strokes(
    target: np.ndarray, centerlines: list[np.ndarray], ambiguity_distance: float = 2.0
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    distances = []
    seeded = []
    for line in centerlines:
        seed = np.logical_and(_line_seed(target.shape, line), target)
        seeded.append(bool(seed.any()))
        distances.append(ndimage.distance_transform_edt(~seed) if seed.any() else np.full(target.shape, np.inf))
    stack = np.stack(distances)
    nearest_labels = np.argmin(stack, axis=0)
    ordered = np.partition(stack, 1, axis=0)
    ambiguous = target & ((ordered[1] - ordered[0]) <= ambiguity_distance)
    seed_labels = np.zeros(target.shape, dtype=np.int32)
    for index, line in enumerate(centerlines):
        seed = np.logical_and(_line_seed(target.shape, line), target)
        seed_labels[seed] = index + 1
    labels = geodesic_labels(target, seed_labels)
    labels[target & (labels == 0)] = nearest_labels[target & (labels == 0)] + 1
    masks = [target & (labels == index + 1) for index in range(len(centerlines))]
    return masks, ambiguous, {
        "targetPixels": int(target.sum()),
        "seededStrokes": sum(seeded),
        "strokeCount": len(centerlines),
        "ambiguousPixels": int(ambiguous.sum()),
        "ambiguousRatio": round(float(ambiguous.sum() / target.sum()), 6),
        "strokePixels": [int(mask.sum()) for mask in masks],
    }


def geodesic_labels(target: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Grow stroke labels only through ink, keeping nearby disconnected ink apart."""
    height, width = target.shape
    distances = np.full(target.shape, np.inf, dtype=np.float32)
    labels = np.zeros(target.shape, dtype=np.int32)
    queue: list[tuple[float, int, int, int]] = []
    ys, xs = np.where(target & (seeds > 0))
    for y, x in zip(ys, xs):
        label = int(seeds[y, x])
        distances[y, x] = 0
        labels[y, x] = label
        heapq.heappush(queue, (0.0, int(y), int(x), label))
    neighbors = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                 (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
    while queue:
        distance, y, x, label = heapq.heappop(queue)
        if distance != float(distances[y, x]) or label != int(labels[y, x]):
            continue
        for dy, dx, cost in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not target[ny, nx]:
                continue
            candidate = distance + cost
            if candidate < float(distances[ny, nx]):
                distances[ny, nx] = candidate
                labels[ny, nx] = label
                heapq.heappush(queue, (candidate, ny, nx, label))
    return labels


def mask_svg_path(mask: np.ndarray, canvas: int, simplify: float = 0.75) -> str:
    found = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = found[0] if len(found) == 2 else found[1]
    parts = []
    scale = 100 / canvas
    for contour in contours:
        if abs(cv2.contourArea(contour)) < 1:
            continue
        points = cv2.approxPolyDP(contour, simplify, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        commands = [f"M {points[0][0] * scale:.4f} {points[0][1] * scale:.4f}"]
        commands.extend(f"L {x * scale:.4f} {y * scale:.4f}" for x, y in points[1:])
        commands.append("Z")
        parts.append(" ".join(commands))
    return " ".join(parts)


def polyline_path(points: np.ndarray, canvas: int) -> str:
    scale = 100 / canvas
    if not len(points):
        return ""
    return "M " + " L ".join(f"{x * scale:.4f} {y * scale:.4f}" for x, y in points)


def _cluster_points(points: list[np.ndarray], radius: float) -> list[np.ndarray]:
    if not points:
        return []
    values = np.asarray(points, dtype=float)
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in cKDTree(values).query_pairs(radius):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root
    groups: dict[int, list[np.ndarray]] = {}
    for index, point in enumerate(values):
        groups.setdefault(find(index), []).append(point)
    return [np.mean(group, axis=0) for group in groups.values()]


def _point_at_fraction(line: np.ndarray, fraction: float) -> np.ndarray:
    if len(line) < 2:
        return line[0]
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    target = fraction * cumulative[-1]
    index = min(len(lengths) - 1, max(0, int(np.searchsorted(cumulative, target) - 1)))
    if lengths[index] == 0:
        return line[index]
    local = (target - cumulative[index]) / lengths[index]
    return line[index] + (line[index + 1] - line[index]) * local


def topology_markers(
    centerlines: list[np.ndarray], canvas: int, strokes: list[dict] | None = None
) -> dict[str, list[np.ndarray]]:
    endpoints = [point for line in centerlines if len(line) for point in (line[0], line[-1])]
    contacts: list[np.ndarray] = []
    dense = [resample_polyline(line, step=2.5) for line in centerlines]
    for left_index, left in enumerate(dense):
        if not len(left):
            continue
        left_tree = cKDTree(left)
        for right in dense[left_index + 1 :]:
            if not len(right):
                continue
            pairs = left_tree.query_ball_point(right, r=max(2.5, canvas * 0.006))
            for right_point, left_indices in zip(right, pairs):
                for index in left_indices:
                    contacts.append((left[index] + right_point) / 2)
    contacts = _cluster_points(contacts, radius=max(5.0, canvas * 0.014))

    bends = []
    if strokes is not None:
        for stroke, line in zip(strokes, centerlines):
            if not len(line):
                continue
            bends.extend(
                _point_at_fraction(line, fraction)
                for fraction in stroke.get("bendFractions", [])
            )
        bends = _cluster_points(bends, radius=max(3.0, canvas * 0.007))
    return {"endpoints": endpoints, "contacts": contacts, "bends": bends}


def marker_svg(centerlines: list[np.ndarray], canvas: int, strokes: list[dict]) -> str:
    markers = topology_markers(centerlines, canvas, strokes)
    scale = 100 / canvas
    parts = []
    for x, y in markers["endpoints"]:
        parts.append(f'<circle class="node endpoint" cx="{x * scale:.4f}" cy="{y * scale:.4f}" r=".62"/>')
    for x, y in markers["contacts"]:
        parts.append(f'<circle class="node contact" cx="{x * scale:.4f}" cy="{y * scale:.4f}" r=".78"/>')
    for x, y in markers["bends"]:
        x *= scale
        y *= scale
        parts.append(
            f'<path class="node bend" d="M {x - .65:.4f} {y - .65:.4f} L {x + .65:.4f} {y + .65:.4f} '
            f'M {x + .65:.4f} {y - .65:.4f} L {x - .65:.4f} {y + .65:.4f}"/>'
        )
    return "".join(parts)


def build_html(
    record: dict,
    glyph: dict,
    glyph_id: int,
    strokes: list[dict],
    centerlines: list[np.ndarray],
    masks: list[np.ndarray],
    ambiguous: np.ndarray,
    metrics: dict,
    canvas: int,
    candidate_svg: str,
) -> str:
    stroke_palette = (
        "#e11d48", "#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2",
        "#ca8a04", "#4f46e5", "#db2777", "#15803d", "#c2410c", "#7c3aed",
        "#0369a1", "#be123c", "#65a30d", "#a21caf",
    )
    layers = []
    medians = []
    for index, (stroke, centerline, mask) in enumerate(zip(strokes, centerlines, masks)):
        path_data = mask_svg_path(mask, canvas)
        label = (
            f"第 {index + 1} 笔 {stroke['feature']} · 部件 {stroke['componentId']} "
            f"· 兄弟族 {stroke['familyKey']}"
        )
        layers.append(
            f'''<path class="source-stroke" data-stroke-index="{index}" data-component-id="{stroke['componentId']}" data-label="{html.escape(label, quote=True)}" d="{path_data}" style="--component-color:{stroke['color']};--stroke-color:{stroke_palette[index % len(stroke_palette)]}" fill-rule="evenodd" clip-rule="evenodd"/>'''
        )
        median_path = polyline_path(centerline, canvas)
        medians.append(
            f'<path class="median" data-stroke-index="{index}" d="{median_path}" stroke="{stroke["color"]}"/>'
        )
    ambiguity_path = mask_svg_path(ambiguous, canvas)
    nodes = marker_svg(centerlines, canvas, strokes)
    original_use = f'''<use class="source-use" href="{glyph['useAttributes']['href']}" x="{glyph['useAttributes'].get('x', 0)}" y="{glyph['useAttributes'].get('y', 0)}"/>'''
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>U+{record['unicode']:04X} 逐笔迁移</title><style>
    body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#eef2f7;color:#172033}}header{{padding:14px 22px;background:#0f172a;color:white}}header button{{margin:10px 7px 0 0;border:1px solid #94a3b8;border-radius:6px;padding:5px 9px;background:white;cursor:pointer}}header button.active{{background:#38bdf8;border-color:#38bdf8}}.warning{{color:#fde68a;margin-top:5px}}main{{padding:18px;display:grid;grid-template-columns:repeat(4,minmax(250px,1fr));gap:14px}}article{{background:white;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden}}h3{{font-size:14px;margin:0;padding:9px;background:#f1f5f9}}svg{{display:block;width:100%;height:auto;aspect-ratio:1}}.source-fit{{fill:#111827}}.source-mask-fit{{fill:white}}.source-stroke{{fill:var(--component-color)}}body.stroke-mode .source-stroke{{fill:var(--stroke-color)}}.source-stroke:hover{{filter:drop-shadow(0 0 1.5px #111);stroke:#111;stroke-width:.25}}.median{{fill:none;stroke-width:.45;stroke-dasharray:1 1;opacity:.75}}.ambiguity{{fill:url(#hatch);opacity:.8;pointer-events:none}}.endpoint{{fill:#ef4444}}.contact{{fill:white;stroke:#2563eb;stroke-width:.38}}.bend{{fill:none;stroke:#22c55e;stroke-width:.42;stroke-linecap:round}}body.nodes-hidden .node{{display:none}}pre{{margin:0;padding:12px;white-space:pre-wrap;font-size:12px}}#tip{{position:fixed;z-index:5;display:none;pointer-events:none;background:#111827;color:white;padding:6px 8px;border-radius:6px;font-size:12px}}
    </style></head><body><header><h2>U+{record['unicode']:04X} {chr(record['unicode'])} · {record['source']} 源 · candidate {glyph_id}</h2><div>逐笔语义来自 hanzi-chai；字形轮廓来自 Unicode PDF。红实点＝端点，蓝空心点＝笔画接触/交叉，绿叉＝稀疏结构转折。</div><div class="warning">斜线区域＝笔画归属歧义；当前 {metrics['ambiguousRatio'] * 100:.2f}%，本页只用于验证，不得作为自动写入证据。</div><button id="component-mode" class="active">按递归叶部件聚色</button><button id="stroke-mode">每笔独立着色</button><button id="node-mode" class="active">显示拓扑节点</button></header><main>
    <article><h3>PDF 原始矢量轮廓</h3><svg viewBox="0 0 100 100"><defs>{glyph['definitions']}</defs><g class="source-fit">{original_use}</g></svg></article>
    <article><h3>hanzi-chai 候选（递归到叶部件）</h3>{candidate_svg}</article>
    <article><h3>PDF 逐笔互斥分区（原始矢量外沿）</h3><svg viewBox="0 0 100 100"><defs><pattern id="hatch" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="2" stroke="#111827" stroke-width=".25"/></pattern><mask id="pdf-outline-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100"><rect width="100" height="100" fill="black"/><g class="source-mask-fit">{original_use}</g></mask></defs><g mask="url(#pdf-outline-mask)">{''.join(layers)}</g><path class="ambiguity" d="{ambiguity_path}" fill-rule="evenodd"/>{nodes}</svg></article>
    <article><h3>拟合后的逐笔中心线</h3><svg viewBox="0 0 100 100">{''.join(medians)}</svg><pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre></article>
    </main><div id="tip"></div><script>
    function fit(el){{const b=el.getBBox(),s=Math.min(84/b.width,84/b.height),tx=50-s*(b.x+b.width/2),ty=50-s*(b.y+b.height/2);el.setAttribute('transform',`matrix(${{s}} 0 0 ${{s}} ${{tx}} ${{ty}})`);}}
    document.querySelectorAll('.source-fit,.source-mask-fit').forEach(fit);const tip=document.querySelector('#tip');document.querySelectorAll('.source-stroke').forEach(el=>{{el.addEventListener('pointerenter',()=>{{tip.textContent=el.dataset.label;tip.style.display='block'}});el.addEventListener('pointermove',e=>{{tip.style.left=`${{e.clientX+12}}px`;tip.style.top=`${{e.clientY+12}}px`}});el.addEventListener('pointerleave',()=>tip.style.display='none')}});const componentButton=document.querySelector('#component-mode'),strokeButton=document.querySelector('#stroke-mode'),nodeButton=document.querySelector('#node-mode');componentButton.onclick=()=>{{document.body.classList.remove('stroke-mode');componentButton.classList.add('active');strokeButton.classList.remove('active')}};strokeButton.onclick=()=>{{document.body.classList.add('stroke-mode');strokeButton.classList.add('active');componentButton.classList.remove('active')}};nodeButton.onclick=()=>{{document.body.classList.toggle('nodes-hidden');nodeButton.classList.toggle('active')}};if(new URLSearchParams(location.search).get('mode')==='stroke')strokeButton.click();
    </script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--unicode", required=True, help="hex codepoint, e.g. 6418")
    parser.add_argument("--source", required=True)
    parser.add_argument("--glyph-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=512)
    args = parser.parse_args()

    codepoint = int(args.unicode.removeprefix("U+").removeprefix("u+"), 16)
    candidate_rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    row = next(item for item in candidate_rows if item["unicode"] == codepoint)
    records, _sizes = PDF.parse_pdf_cells(args.bbox_cache)
    record = next(
        item for item in records if item["unicode"] == codepoint and item["source"] == args.source
    )
    page_svg = MATCHER.load_pdf_page_svg(args.pdf, record["page"])
    glyph = VECTOR.extract_pdf_glyph(page_svg, record["bbox"], f"u{codepoint:04x}-{args.source.lower()}")
    source_mask = MATCHER.render_pdf_vector_cell(
        page_svg,
        MATCHER.chart_glyph_bbox(record["bbox"]),
        size=args.canvas,
    ) < 224
    target = normalize_target(source_mask, args.canvas)
    strokes = candidate_strokes(row, args.glyph_id)
    fitted = fit_centerlines(strokes, args.canvas)
    snapped, snap_distances = snap_centerlines(fitted, target, max_distance=args.canvas * 0.08)
    masks, ambiguous, metrics = partition_strokes(target, snapped)
    metrics["maxSnapDistances"] = [round(value, 3) for value in snap_distances]
    metrics["emptyStrokes"] = [index for index, mask in enumerate(masks) if not mask.any()]
    document = build_html(
        record,
        glyph,
        args.glyph_id,
        strokes,
        snapped,
        masks,
        ambiguous,
        metrics,
        args.canvas,
        row["candidateSvgs"][str(args.glyph_id)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, "utf-8")
    print(json.dumps({"output": str(args.output), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
