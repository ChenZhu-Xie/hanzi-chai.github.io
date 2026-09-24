"""Transfer repository stroke semantics onto a Unicode PDF glyph outline.

This is deliberately template-guided: the repository candidate supplies stroke
order and component ownership.  The PDF outline supplies the target silhouette.
The outline alone cannot uniquely recover overlapping stroke order.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
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
                "occurrence": leaf.get("occurrence", 0),
                "hierarchy": leaf.get("hierarchy")
                or [
                    {
                        "id": leaf["leafId"],
                        "type": "component",
                        "label": "末级部件",
                        "familyKey": leaf["familyKey"],
                    },
                    {
                        "id": glyph_id,
                        "type": "glyph",
                        "label": "整字字形",
                        "familyKey": str(glyph_id),
                    },
                ],
                "points": points,
                "bendFractions": bend_fractions,
            }
    expected = len(row["candidates"][str(glyph_id)])
    if sorted(by_index) != list(range(expected)):
        raise ValueError(f"candidate {glyph_id} has incomplete stroke ownership")
    return [by_index[index] for index in range(expected)]


def load_human_annotations(
    path: Path,
    *,
    codepoint: int,
    source: str,
    glyph_id: int,
    candidate: list[dict],
    canvas: int,
) -> tuple[list[dict], dict, list[dict]]:
    """Validate and normalize human truth without modifying its source JSON."""
    document = json.loads(path.read_text("utf-8"))
    metadata = document.get("metadata", {})
    expected_unicode = f"U+{codepoint:04X}"
    if metadata.get("unicode") != expected_unicode or metadata.get("source") != source:
        raise ValueError(
            f"annotation target mismatch: expected {expected_unicode} {source}, "
            f"got {metadata.get('unicode')} {metadata.get('source')}"
        )
    if metadata.get("candidateGlyphId") != glyph_id:
        raise ValueError("annotation candidateGlyphId does not match --glyph-id")
    raw_annotations = document.get("annotations")
    if not isinstance(raw_annotations, list):
        raise ValueError("annotation JSON has no annotations array")

    by_component: dict[int, dict] = {}
    family_to_candidate: dict[str, dict] = {}
    for stroke in candidate:
        by_component.setdefault(int(stroke["componentId"]), stroke)
        family_to_candidate.setdefault(str(stroke["familyKey"]), stroke)

    corrections = []
    normalized = copy.deepcopy(document)
    normalized_annotations = normalized["annotations"]
    for index, annotation in enumerate(normalized_annotations):
        label = str(annotation.get("label", "")).strip()
        if not label.isdigit():
            continue
        component_id = int(label)
        resolved = by_component.get(component_id)
        if resolved is None:
            resolved = next(
                (
                    stroke
                    for family, stroke in family_to_candidate.items()
                    if label in family.split("/")
                ),
                None,
            )
        if resolved is None:
            continue
        expected_id = int(resolved["componentId"])
        if component_id != expected_id:
            if annotation.get("type") == "lasso":
                display = "部件圈"
            else:
                stroke_number = sum(
                    item.get("type") != "lasso"
                    for item in normalized_annotations[: index + 1]
                )
                display = f"第 {stroke_number} 笔"
            corrections.append(
                {
                    "annotationIndex": index,
                    "type": annotation.get("type"),
                    "display": display,
                    "from": component_id,
                    "to": expected_id,
                    "reason": f"selected candidate {glyph_id} uses sibling {expected_id}",
                }
            )
        annotation["label"] = str(expected_id)
        annotation["color"] = resolved["color"]

    stroke_annotations = [
        annotation
        for annotation in normalized_annotations
        if annotation.get("type") != "lasso"
    ]
    if len(stroke_annotations) != len(candidate):
        raise ValueError(
            f"human truth has {len(stroke_annotations)} strokes, candidate has {len(candidate)}"
        )
    manual = []
    scale = canvas / 100
    for index, (annotation, fallback) in enumerate(zip(stroke_annotations, candidate)):
        points = np.asarray(annotation.get("points", []), dtype=float)
        if len(points) < 2 or points.shape[1:] != (2,):
            raise ValueError(f"human stroke {index + 1} has invalid points")
        component_id = int(annotation.get("label") or fallback["componentId"])
        component = by_component.get(component_id, fallback)
        manual.append(
            {
                **fallback,
                "componentId": component_id,
                "familyKey": component["familyKey"],
                "color": component["color"],
                "hierarchy": component["hierarchy"],
                "occurrence": component["occurrence"],
                "points": points * scale,
                "annotationType": annotation.get("type", "freehand"),
            }
        )
    return manual, normalized, corrections


def partition_human_truth(
    target: np.ndarray,
    centerlines: list[np.ndarray],
    strokes: list[dict],
    annotations: list[dict],
    canvas: int,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Use human stroke medians plus component lassos to assign target ink."""
    baseline_masks, ambiguous, metrics = partition_strokes(target, centerlines)
    component_ids = list(dict.fromkeys(int(stroke["componentId"]) for stroke in strokes))
    component_number = {component_id: index + 1 for index, component_id in enumerate(component_ids)}
    component_seeds = np.zeros(target.shape, dtype=np.int32)
    overlapping = np.zeros(target.shape, dtype=bool)
    scale = canvas / 100

    for annotation in annotations:
        if annotation.get("type") != "lasso":
            continue
        label = str(annotation.get("label", ""))
        if not label.isdigit() or int(label) not in component_number:
            continue
        points = np.rint(np.asarray(annotation.get("points", []), dtype=float) * scale).astype(np.int32)
        if len(points) < 3:
            continue
        polygon = np.zeros(target.shape, dtype=np.uint8)
        cv2.fillPoly(polygon, [points], 1)
        seed = target & (polygon > 0)
        conflict = seed & (component_seeds > 0) & (
            component_seeds != component_number[int(label)]
        )
        overlapping |= conflict
        component_seeds[seed & ~conflict] = component_number[int(label)]
        component_seeds[conflict] = 0

    for stroke, line in zip(strokes, centerlines):
        seed = target & _line_seed(target.shape, line, width=2)
        component_seeds[seed] = component_number[int(stroke["componentId"])]
    component_labels = geodesic_labels(target, component_seeds)

    masks = [np.zeros_like(target) for _ in strokes]
    for component_id in component_ids:
        indices = [
            index
            for index, stroke in enumerate(strokes)
            if int(stroke["componentId"]) == component_id
        ]
        component_mask = target & (
            component_labels == component_number[component_id]
        )
        if not component_mask.any():
            for index in indices:
                masks[index] = baseline_masks[index]
            continue
        if len(indices) == 1:
            masks[indices[0]] = component_mask
            continue
        local_masks, _local_ambiguous, _local_metrics = partition_strokes(
            component_mask, [centerlines[index] for index in indices]
        )
        for index, local_mask in zip(indices, local_masks):
            masks[index] = local_mask

    uncovered = target & ~np.logical_or.reduce(masks)
    if uncovered.any():
        for index, baseline in enumerate(baseline_masks):
            masks[index] |= uncovered & baseline
    metrics.update(
        {
            "partitionMode": "human-stroke-and-component-truth",
            "humanLassoCount": sum(
                annotation.get("type") == "lasso" for annotation in annotations
            ),
            "humanComponentIds": component_ids,
            "lassoConflictPixels": int(overlapping.sum()),
            "strokePixels": [int(mask.sum()) for mask in masks],
        }
    )
    return masks, ambiguous, metrics


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


def directed_stroke_features(points: np.ndarray, canvas: int) -> dict:
    """Describe one stroke as an ordered arc from pen-down to pen-up."""
    values = np.asarray(points, dtype=float)
    if not len(values):
        return {"pointCount": 0}
    scale = 100 / canvas
    segments = np.diff(values, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    nonzero = lengths > 1e-6
    vectors = segments[nonzero]
    vector_lengths = lengths[nonzero]
    units = vectors / vector_lengths[:, None] if len(vectors) else np.empty((0, 2))

    def vector_payload(vector: np.ndarray) -> dict:
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        return {
            "dx": round(float(vector[0]), 5),
            "dy": round(float(vector[1]), 5),
            "angleDegrees": round(angle, 3),
        }

    turns = []
    for left, right in zip(units, units[1:]):
        cross = float(left[0] * right[1] - left[1] * right[0])
        dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
        turns.append(round(math.degrees(math.atan2(cross, dot)), 3))
    displacement = values[-1] - values[0]
    return {
        "pointCount": len(values),
        "direction": "points[0] -> points[-1] (pen-down -> pen-up)",
        "start": [round(float(value * scale), 4) for value in values[0]],
        "end": [round(float(value * scale), 4) for value in values[-1]],
        "arcLength": round(float(lengths.sum() * scale), 4),
        "displacement": [round(float(value * scale), 4) for value in displacement],
        "startTangent": vector_payload(units[0]) if len(units) else None,
        "endTangent": vector_payload(units[-1]) if len(units) else None,
        "segmentDirections": [vector_payload(unit) for unit in units],
        "signedTurnsDegrees": turns,
    }


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


def hierarchy_color(key: str) -> str:
    """Return a stable, readable color for non-leaf hierarchy rows."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535
    saturation = 0.56 + digest[2] / 255 * 0.14
    lightness = 0.42 + digest[3] / 255 * 0.10
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def enrich_hierarchy(hierarchy: list[dict], leaf_color: str) -> list[dict]:
    output = []
    for index, node in enumerate(hierarchy):
        family_key = str(node.get("familyKey", node["id"]))
        output.append(
            {
                **node,
                "familyKey": family_key,
                "color": leaf_color if index == 0 else hierarchy_color(family_key),
            }
        )
    return output


def interactive_candidate_svg(
    row: dict,
    glyph_id: int,
    registry: dict[str, dict],
    *,
    instance: str,
) -> str:
    """Attach real leaf/ancestor metadata to each candidate stroke path."""
    candidate_key = str(glyph_id)
    root = ET.fromstring(row["candidateSvgs"][candidate_key])
    root.set("class", "candidate-svg")
    # candidateSvgs already contain endpoint-only red circles.  Rebuild the
    # complete marker layer so candidates and PDF attribution use one topology
    # vocabulary: red endpoints, blue contacts, green structural bends.
    for child in list(root):
        if child.tag.rsplit("}", 1)[-1] == "circle":
            root.remove(child)
    paths = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "path"]
    by_stroke: dict[int, tuple[str, dict]] = {}
    for leaf in row["candidateLeafSvgs"][candidate_key]:
        part_key = (
            f"candidate:{instance}:{glyph_id}:{leaf['leafId']}:{leaf.get('occurrence', 0)}"
        )
        hierarchy = enrich_hierarchy(
            leaf.get("hierarchy")
            or [
                {
                    "id": leaf["leafId"],
                    "type": "component",
                    "label": "末级部件",
                    "familyKey": leaf["familyKey"],
                },
                {
                    "id": glyph_id,
                    "type": "glyph",
                    "label": "整字字形",
                    "familyKey": str(glyph_id),
                },
            ],
            leaf["color"],
        )
        registry[part_key] = {
            "leafId": leaf["leafId"],
            "familyKey": leaf["familyKey"],
            "glyphId": glyph_id,
            "color": leaf["color"],
            "hierarchy": hierarchy,
        }
        for stroke_index in leaf["strokeIndices"]:
            by_stroke[int(stroke_index)] = (part_key, registry[part_key])
    if len(paths) != len(row["candidates"][candidate_key]):
        raise ValueError(f"candidate {glyph_id} SVG path count mismatch")
    for stroke_index, path in enumerate(paths):
        part_key, part = by_stroke[stroke_index]
        path.set("class", "interactive-part candidate-part")
        path.set("data-part-key", part_key)
        path.set("data-family-key", str(part["familyKey"]))
        path.set("data-component-id", str(part["leafId"]))
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    markup = ET.tostring(root, encoding="unicode")
    candidate = candidate_strokes(row, glyph_id)
    markers = marker_svg(
        [stroke["points"] for stroke in candidate],
        100,
        candidate,
    )
    return markup.replace("</svg>", f'<g class="candidate-topology">{markers}</g></svg>')


def evidence_for_record(evidence: dict | None, unicode: int, source: str) -> dict | None:
    if not evidence:
        return None
    return next(
        (
            item
            for item in evidence.get("results", [])
            if item.get("unicode") == unicode and item.get("source") == source
        ),
        None,
    )


def rank_candidates(row: dict, result: dict | None, evidence: dict | None) -> list[dict]:
    candidate_ids = [int(value) for value in row["candidateSvgs"]]
    evidence_by_id = {
        int(item["id"]): item for item in (result or {}).get("candidates", [])
    }
    raw_distances = {
        glyph_id: float(evidence_by_id.get(glyph_id, {}).get("distance", 999.0))
        for glyph_id in candidate_ids
    }
    finite = [value for value in raw_distances.values() if value < 999]
    baseline = min(finite, default=0.0)
    distance_weights = {
        glyph_id: math.exp(-min(40.0, max(0.0, distance - baseline)))
        if distance < 999
        else 0.0
        for glyph_id, distance in raw_distances.items()
    }
    distance_total = sum(distance_weights.values()) or 1.0
    predicted = (result or {}).get("predictedGlyphId")
    method = (result or {}).get("predictionMethod")
    method_rows = [
        item
        for item in (evidence or {}).get("results", [])
        if method and item.get("predictionMethod") == method
    ]
    method_correct = sum(item.get("correct") is True for item in method_rows)
    method_total = len(method_rows)
    decision_weights = {
        glyph_id: distance_weights[glyph_id] / distance_total
        for glyph_id in candidate_ids
    }
    if predicted in candidate_ids and method_total:
        predicted_prior = (method_correct + 1) / (method_total + 2)
        other_ids = [glyph_id for glyph_id in candidate_ids if glyph_id != predicted]
        other_total = sum(distance_weights[glyph_id] for glyph_id in other_ids) or 1.0
        decision_weights = {
            glyph_id: (
                predicted_prior
                if glyph_id == predicted
                else (1 - predicted_prior)
                * distance_weights[glyph_id]
                / other_total
            )
            for glyph_id in candidate_ids
        }
    ordered = sorted(
        candidate_ids,
        key=lambda glyph_id: (
            -decision_weights[glyph_id],
            glyph_id,
        ),
    )
    return [
        {
            "id": glyph_id,
            "rank": index + 1,
            "recommended": glyph_id == predicted,
            "relativeWeight": decision_weights[glyph_id],
            "distanceWeight": distance_weights[glyph_id] / distance_total,
            "methodPrior": {
                "method": method,
                "correct": method_correct,
                "total": method_total,
                "laplaceAccuracy": (method_correct + 1) / (method_total + 2)
                if method_total
                else None,
            },
            **evidence_by_id.get(glyph_id, {}),
        }
        for index, glyph_id in enumerate(ordered)
    ]


def format_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def annotation_editor_script(metadata: dict, initial_annotations: list[dict] | None = None) -> str:
    payload = json.dumps(metadata, ensure_ascii=False)
    initial_payload = json.dumps(initial_annotations or [], ensure_ascii=False).replace("</", "<\\/")
    return r'''
    (() => {
      document.querySelector('.editor').id = 'human-editor';
      const metadata = __METADATA__;
      const initialAnnotations = __INITIAL_ANNOTATIONS__;
      const storageKey = `unihan-human-annotations:${metadata.reviewKey}`;
      const board = document.querySelector('#annotation-board');
      const layer = document.querySelector('#annotation-layer');
      const status = document.querySelector('#annotation-status');
      const labelInput = document.querySelector('#component-label');
      const colorInput = document.querySelector('#annotation-color');
      const fileInput = document.querySelector('#annotation-file');
      let tool = 'freehand';
      let current = null;
      let pointDraft = null;
      let previewPoint = null;
      let annotations = initialAnnotations;
      let history = [];
      let future = [];

      // A freshly supplied human-truth file is authoritative.  Do not let an
      // older browser draft silently restore the pre-correction IDs/colors.
      if (!initialAnnotations.length) {
        try {
          const raw = localStorage.getItem(storageKey);
          const saved = raw ? JSON.parse(raw) : null;
          if (Array.isArray(saved)) annotations = saved;
        } catch (_error) {}
      }

      const svgElement = (name, attributes = {}) => {
        const element = document.createElementNS('http://www.w3.org/2000/svg', name);
        for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
        return element;
      };
      const pointsText = points => points.map(point => `${point[0].toFixed(3)},${point[1].toFixed(3)}`).join(' ');
      const drawingPoint = event => {
        const point = board.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const mapped = point.matrixTransform(board.getScreenCTM().inverse());
        return [Math.max(0, Math.min(100, mapped.x)), Math.max(0, Math.min(100, mapped.y))];
      };
      const isStroke = annotation => annotation.type !== 'lasso';
      const strokeNumber = annotation => annotations.filter(isStroke).indexOf(annotation) + 1;
      const persist = () => {
        localStorage.setItem(storageKey, JSON.stringify(annotations));
        const strokes = annotations.filter(isStroke).length;
        const regions = annotations.filter(item => item.type === 'lasso').length;
        status.innerHTML = `<b>人工真值：</b>${strokes} 笔，${regions} 个部件圈。PDF 的真实笔数现在由你的 ${strokes} 条中心线定义。`;
      };
      const checkpoint = () => {
        history.push(structuredClone(annotations));
        if (history.length > 100) history.shift();
        future = [];
      };
      const mutate = callback => {
        checkpoint();
        callback();
        persist();
        render();
      };
      const centroid = points => [
        points.reduce((sum, point) => sum + point[0], 0) / points.length,
        points.reduce((sum, point) => sum + point[1], 0) / points.length,
      ];
      const appendAnnotation = (annotation, index, preview = false) => {
        if (!annotation.points.length) return;
        let shape;
        if (annotation.type === 'lasso') {
          shape = svgElement('polygon', {
            points: pointsText(annotation.points), fill: annotation.color,
            'fill-opacity': preview ? '.08' : '.16', stroke: annotation.color,
            'stroke-width': '.55', 'stroke-dasharray': '1.2 .7',
          });
        } else if (annotation.type === 'bezier' && annotation.controlPoints?.length === 4) {
          const [start, control1, control2, end] = annotation.controlPoints;
          shape = svgElement('path', {
            d: `M ${start.join(' ')} C ${control1.join(' ')} ${control2.join(' ')} ${end.join(' ')}`,
            fill: 'none', stroke: annotation.color, 'stroke-width': '1.05',
            'stroke-linecap': 'round', 'stroke-linejoin': 'round',
          });
        } else {
          shape = svgElement('polyline', {
            points: pointsText(annotation.points), fill: 'none', stroke: annotation.color,
            'stroke-width': '1.05', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
          });
        }
        shape.classList.add('annotation-shape');
        if (!preview) {
          shape.dataset.annotationIndex = String(index);
          shape.addEventListener('pointerdown', event => {
            if (tool !== 'erase') return;
            event.stopPropagation();
            mutate(() => annotations.splice(Number(shape.dataset.annotationIndex), 1));
          });
        }
        layer.append(shape);
        if (preview) return;
        const anchor = isStroke(annotation) ? annotation.points[0] : centroid(annotation.points);
        const text = svgElement('text', {
          x: anchor[0], y: anchor[1] - 1.2, fill: annotation.color, class: 'annotation-label',
          'text-anchor': isStroke(annotation) ? 'start' : 'middle',
        });
        const prefix = isStroke(annotation) ? `第${strokeNumber(annotation)}笔` : '部件';
        text.textContent = annotation.label ? `${prefix} · ${annotation.label}` : prefix;
        layer.append(text);
      };
      const render = () => {
        layer.replaceChildren();
        annotations.forEach((annotation, index) => appendAnnotation(annotation, index));
        if (current) appendAnnotation(current, -1, true);
        if (pointDraft) appendAnnotation(pointDraft, -1, true);
      };
      const simplifyPoints = points => points.filter((point, index) => {
        if (index === 0 || index === points.length - 1) return true;
        const previous = points[index - 1];
        return Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= .3;
      });

      const cubicPoints = controls => Array.from({length: 41}, (_, index) => {
        const t = index / 40, m = 1 - t;
        return [0, 1].map(axis => m ** 3 * controls[0][axis] + 3 * m ** 2 * t * controls[1][axis] + 3 * m * t ** 2 * controls[2][axis] + t ** 3 * controls[3][axis]);
      });
      const draftAnnotation = (type, points, controlPoints = undefined) => ({
        type, color: colorInput.value, label: labelInput.value.trim(), points,
        controlPoints, createdAt: new Date().toISOString(),
      });
      const finishPointDraft = () => {
        if (!pointDraft) return;
        if (pointDraft.type === 'polyline' && pointDraft.points.length >= 2) {
          mutate(() => annotations.push(pointDraft));
        }
        pointDraft = null;
        previewPoint = null;
        render();
      };

      board.addEventListener('pointerdown', event => {
        if (!['freehand', 'lasso'].includes(tool)) return;
        board.setPointerCapture(event.pointerId);
        current = draftAnnotation(tool, [drawingPoint(event)]);
        render();
      });
      board.addEventListener('pointermove', event => {
        if (!current) {
          if (['line', 'polyline', 'bezier'].includes(tool) && pointDraft) {
            previewPoint = drawingPoint(event);
            if (tool === 'line') pointDraft.points = [pointDraft.points[0], previewPoint];
            else if (tool === 'polyline') pointDraft.points = [...pointDraft.fixedPoints, previewPoint];
            else {
              const controls = [...pointDraft.fixedPoints, previewPoint];
              pointDraft.controlPoints = controls;
              pointDraft.points = controls.length === 4 ? cubicPoints(controls) : controls;
            }
            render();
          }
          return;
        }
        const point = drawingPoint(event);
        const previous = current.points[current.points.length - 1];
        if (Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= .18) {
          current.points.push(point);
          render();
        }
      });
      const finish = event => {
        if (!current) return;
        if (board.hasPointerCapture(event.pointerId)) board.releasePointerCapture(event.pointerId);
        current.points = simplifyPoints(current.points);
        if (current.points.length > 1) {
          const finished = current;
          checkpoint();
          annotations.push(finished);
        }
        current = null;
        persist();
        render();
      };
      board.addEventListener('pointerup', finish);
      board.addEventListener('pointercancel', finish);

      board.addEventListener('click', event => {
        if (!['line', 'polyline', 'bezier'].includes(tool) || event.detail > 1) return;
        const point = drawingPoint(event);
        if (!pointDraft) {
          pointDraft = draftAnnotation(tool, [point]);
          pointDraft.fixedPoints = [point];
        } else {
          pointDraft.fixedPoints.push(point);
        }
        if (tool === 'line' && pointDraft.fixedPoints.length === 2) {
          pointDraft.points = [...pointDraft.fixedPoints];
          const finished = pointDraft;
          mutate(() => annotations.push(finished));
          pointDraft = null;
        } else if (tool === 'bezier' && pointDraft.fixedPoints.length === 4) {
          pointDraft.controlPoints = [...pointDraft.fixedPoints];
          pointDraft.points = cubicPoints(pointDraft.controlPoints);
          const finished = pointDraft;
          mutate(() => annotations.push(finished));
          pointDraft = null;
        }
        render();
      });
      board.addEventListener('dblclick', event => {
        if (tool !== 'polyline') return;
        event.preventDefault();
        finishPointDraft();
      });

      document.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click', () => {
        tool = button.dataset.tool;
        current = null;
        pointDraft = null;
        previewPoint = null;
        document.querySelectorAll('[data-tool]').forEach(item => item.classList.toggle('active', item === button));
        board.style.cursor = tool === 'erase' ? 'not-allowed' : 'crosshair';
      }));
      const undo = () => {
        if (!history.length) return;
        future.push(structuredClone(annotations));
        annotations = history.pop();
        persist(); render();
      };
      const redo = () => {
        if (!future.length) return;
        history.push(structuredClone(annotations));
        annotations = future.pop();
        persist(); render();
      };
      document.querySelector('#undo-annotation').onclick = undo;
      document.querySelector('#redo-annotation').onclick = redo;
      document.querySelector('#clear-annotations').onclick = () => {
        if (!annotations.length || confirm('清空本字源的全部人工标注？')) {
          mutate(() => { annotations = []; });
        }
      };
      document.addEventListener('keydown', event => {
        const modifier = event.ctrlKey || event.metaKey;
        if (modifier && event.key.toLowerCase() === 'z') {
          event.preventDefault();
          if (event.shiftKey) redo(); else undo();
        } else if (modifier && event.key.toLowerCase() === 'y') {
          event.preventDefault(); redo();
        } else if (event.key === 'Escape') {
          current = null; pointDraft = null; previewPoint = null; render();
        } else if (event.key === 'Enter' && tool === 'polyline') {
          event.preventDefault(); finishPointDraft();
        }
      });
      document.querySelector('#export-annotations').onclick = () => {
        const output = { format: 'hanzi-chai-pdf-stroke-review-v1', metadata, annotations };
        const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${metadata.reviewKey}-annotations.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      };
      document.querySelector('#import-annotations').onclick = () => fileInput.click();
      fileInput.addEventListener('change', async () => {
        const documentValue = JSON.parse(await fileInput.files[0].text());
        if (!Array.isArray(documentValue.annotations)) throw new Error('JSON 缺少 annotations 数组');
        checkpoint();
        annotations = documentValue.annotations.map(annotation => {
          const color = window.__componentColorById?.[String(annotation.label)];
          return color ? {...annotation, color} : annotation;
        });
        future = [];
        persist();
        render();
        fileInput.value = '';
      });
      persist();
      render();
    })();
    '''.replace("__METADATA__", payload).replace("__INITIAL_ANNOTATIONS__", initial_payload)


def interaction_script(registry: dict[str, dict]) -> str:
    payload = json.dumps(registry, ensure_ascii=False).replace("</", "<\\/")
    return r'''
    (() => {
      const parts = __PARTS__;
      const tip = document.querySelector('#hierarchy-tip');
      const picker = document.querySelector('#component-picker');
      const labelInput = document.querySelector('#component-label');
      const colorInput = document.querySelector('#annotation-color');
      let activeFamily = null;

      window.__componentColorById = {};
      for (const part of Object.values(parts)) {
        window.__componentColorById[String(part.leafId)] = part.color;
        for (const node of part.hierarchy) {
          window.__componentColorById[String(node.id)] ??= node.color;
          for (const id of String(node.familyKey).split('/')) {
            window.__componentColorById[id] ??= node.color;
          }
        }
      }
      labelInput.addEventListener('input', () => {
        const color = window.__componentColorById[labelInput.value.trim()];
        if (color) colorInput.value = color;
      });

      const readableText = color => {
        const value = color.replace('#', '');
        const [r, g, b] = [0, 2, 4].map(index => parseInt(value.slice(index, index + 2), 16));
        return (r * 299 + g * 587 + b * 114) / 1000 > 150 ? '#111827' : '#fff';
      };
      const levelDescription = (node, index, total) => {
        const bottomUp = index + 1;
        const topDown = total - index;
        let kind;
        if (node.type === 'glyph') kind = index === total - 1 ? '顶级字 / 字形' : '字 / 字形';
        else if (index === 0) kind = '末级部件';
        else if (index === total - 1) kind = '顶级部件';
        else kind = `${'上'.repeat(index)}级部件`;
        return `${kind} · 自下而上第 ${bottomUp}/${total} 级 · 自上而下第 ${topDown}/${total} 级`;
      };
      const hierarchyRows = part => part.hierarchy.map((node, index) => {
        const kind = levelDescription(node, index, part.hierarchy.length);
        return `<div class="hierarchy-row" style="background:${node.color};color:${readableText(node.color)}"><span>${kind}<br>${node.label || ''}</span><b>ID ${node.id}</b></div>`;
      }).join('');
      const familyMatches = (part, family) => part && part.hierarchy.some(node => node.familyKey === family);
      const highlight = family => {
        activeFamily = family;
        document.querySelectorAll('.interactive-part').forEach(element => {
          const part = parts[element.dataset.partKey];
          element.classList.toggle('linked-highlight', familyMatches(part, family));
          element.classList.toggle('linked-dim', !familyMatches(part, family));
        });
      };
      const clearHighlight = () => {
        activeFamily = null;
        document.querySelectorAll('.interactive-part').forEach(element => element.classList.remove('linked-highlight', 'linked-dim'));
      };
      const moveFloating = (element, event) => {
        const left = Math.min(innerWidth - element.offsetWidth - 12, event.clientX + 14);
        const top = Math.min(innerHeight - element.offsetHeight - 12, event.clientY + 14);
        element.style.left = `${Math.max(8, left)}px`;
        element.style.top = `${Math.max(8, top)}px`;
      };
      const copyId = async id => {
        try { await navigator.clipboard.writeText(String(id)); }
        catch (_error) {
          const area = document.createElement('textarea');
          area.value = String(id); document.body.append(area); area.select();
          document.execCommand('copy'); area.remove();
        }
      };
      const pickerOptions = part => {
        const seen = new Set();
        const total = part.hierarchy.length;
        return part.hierarchy.flatMap((node, index) => {
          const ids = String(node.familyKey).split('/').map(Number).filter(Number.isFinite);
          const members = ids.length ? ids : [node.id];
          return members.flatMap(id => {
            const key = `${node.familyKey}:${id}`;
            if (seen.has(key)) return [];
            seen.add(key);
            return [{ ...node, id, sibling: id !== node.id, level: levelDescription(node, index, total) }];
          });
        });
      };
      const openPicker = (part, event) => {
        picker.innerHTML = '<div class="picker-title">右键层级列表 · 选中后仅复制 ID</div>';
        for (const option of pickerOptions(part)) {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'picker-option';
          button.style.background = option.color;
          button.style.color = readableText(option.color);
          button.innerHTML = `<span>${option.sibling ? '同族候选 · ' : ''}${option.level}<br>${option.label || ''}</span><b>ID ${option.id}</b>`;
          button.onclick = async clickEvent => {
            clickEvent.stopPropagation();
            await copyId(option.id);
            picker.hidden = true;
          };
          picker.append(button);
        }
        picker.hidden = false;
        moveFloating(picker, event);
      };

      document.querySelectorAll('.interactive-part').forEach(element => {
        const part = parts[element.dataset.partKey];
        if (!part) return;
        element.addEventListener('pointerenter', event => {
          highlight(part.familyKey);
          tip.innerHTML = `<div class="tip-position">鼠标：${Math.round(event.clientX)}, ${Math.round(event.clientY)} · glyph ${part.glyphId}</div>${hierarchyRows(part)}`;
          tip.hidden = false;
          moveFloating(tip, event);
        });
        element.addEventListener('pointermove', event => moveFloating(tip, event));
        element.addEventListener('pointerleave', () => {
          if (!picker.matches(':hover')) clearHighlight();
          tip.hidden = true;
        });
        element.addEventListener('click', event => {
          event.stopPropagation();
          highlight(part.familyKey);
          if (element.closest('[data-selection-mode="fill"]')) {
            labelInput.value = String(part.leafId);
            colorInput.value = part.color;
            labelInput.dispatchEvent(new Event('input', { bubbles: true }));
            colorInput.dispatchEvent(new Event('input', { bubbles: true }));
          }
        });
        element.addEventListener('contextmenu', event => {
          event.preventDefault();
          event.stopPropagation();
          highlight(part.familyKey);
          openPicker(part, event);
        });
      });
      document.addEventListener('click', event => {
        if (!picker.contains(event.target)) picker.hidden = true;
      });
      picker.addEventListener('pointerleave', () => {
        if (activeFamily) clearHighlight();
      });
    })();
    '''.replace("__PARTS__", payload)


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
    row: dict,
    evidence: dict | None,
    initial_annotations: list[dict] | None = None,
    annotation_corrections: list[dict] | None = None,
) -> str:
    stroke_palette = (
        "#e11d48", "#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2",
        "#ca8a04", "#4f46e5", "#db2777", "#15803d", "#c2410c", "#7c3aed",
        "#0369a1", "#be123c", "#65a30d", "#a21caf",
    )
    registry: dict[str, dict] = {}
    result = evidence_for_record(evidence, record["unicode"], record["source"])
    ranked = rank_candidates(row, result, evidence)
    sources_by_glyph: dict[int, list[str]] = {}
    for source, candidate_id in row.get("sourceGlyphs", {}).items():
        sources_by_glyph.setdefault(int(candidate_id), []).append(source)

    recursive = (result or {}).get("recursiveTopology") or {}
    recursive_ids = recursive.get("candidateGlyphIds", [])
    recursive_details = recursive.get("candidateDetails", [])
    recursive_by_id = {
        int(candidate_id): recursive_details[index]
        for index, candidate_id in enumerate(recursive_ids)
        if index < len(recursive_details)
    }

    def evidence_card(entry: dict, *, compact: bool, instance: str) -> str:
        candidate_id = int(entry["id"])
        candidate_svg = interactive_candidate_svg(
            row, candidate_id, registry, instance=instance
        )
        sources = "/".join(sources_by_glyph.get(candidate_id, [])) or "未分配"
        selected = " · 当前算法最终推荐" if entry.get("recommended") else ""
        details = recursive_by_id.get(candidate_id, {})
        flags = []
        if entry.get("recommended"):
            flags.append(f"方法：{(result or {}).get('predictionMethod', '无评测记录')}")
        if (result or {}).get("topologyGatePrediction") == candidate_id:
            flags.append("拓扑门控支持")
        if (result or {}).get("topologyGateOverride") and entry.get("recommended"):
            flags.append("拓扑门控覆盖了纯距离顺序")
        exact_component = recursive.get("exactComponentCandidateIndex")
        if (
            isinstance(exact_component, int)
            and exact_component < len(recursive_ids)
            and recursive_ids[exact_component] == candidate_id
        ):
            flags.append("PDF 连通分量数精确匹配")
        evidence_payload = {
            "candidate": entry,
            "recursiveCandidateDetail": details or None,
            "globalDecision": {
                "predictedGlyphId": (result or {}).get("predictedGlyphId"),
                "predictionMethod": (result or {}).get("predictionMethod"),
                "bestDistance": (result or {}).get("bestDistance"),
                "secondDistance": (result or {}).get("secondDistance"),
                "margin": (result or {}).get("margin"),
                "topologyGatePrediction": (result or {}).get("topologyGatePrediction"),
                "topologyGateOverride": (result or {}).get("topologyGateOverride"),
                "alignment": (result or {}).get("alignment"),
            },
        }
        if compact:
            return f'''<article class="editor-candidate" data-selection-mode="fill"><div class="candidate-heading"><b>#{entry['rank']} · glyph {candidate_id}</b><span>{entry['relativeWeight'] * 100:.1f}% 经验权重</span></div><div class="compact-candidate-body">{candidate_svg}<div><div>来源：{html.escape(sources)}</div><div>总距离：{format_metric(entry.get('distance'))}</div><div>纯距离权重：{entry['distanceWeight'] * 100:.1f}%</div><div>{html.escape('；'.join(flags) or '未获最终规则加权')}</div></div></div></article>'''
        prior = entry["methodPrior"]
        prior_text = (
            f"{prior['correct']}/{prior['total']}；Laplace {prior['laplaceAccuracy'] * 100:.1f}%"
            if prior["total"]
            else "无同方法历史样本"
        )
        return f'''<article class="candidate-card" data-selection-mode="fill"><h3>#{entry['rank']} · glyph {candidate_id} · 来源 {html.escape(sources)}{html.escape(selected)}</h3><div class="candidate-visual">{candidate_svg}</div><div class="weight"><b>{entry['relativeWeight'] * 100:.1f}%</b> 经验决策权重（非校准概率）</div><dl class="evidence-grid"><dt>总距离</dt><dd>{format_metric(entry.get('distance'))}</dd><dt>纯距离权重</dt><dd>{entry['distanceWeight'] * 100:.1f}%</dd><dt>视觉距离</dt><dd>{format_metric(entry.get('visualDistance'))}</dd><dt>拓扑距离</dt><dd>{format_metric(entry.get('topologyDistance'))}</dd><dt>递归拓扑</dt><dd>{format_metric(entry.get('recursiveTopologyDistance'))}</dd><dt>方法历史命中</dt><dd>{html.escape(prior_text)}</dd><dt>递归 forward</dt><dd>{format_metric(details.get('forward'))}</dd><dt>递归 reverse</dt><dd>{format_metric(details.get('reverse'))}</dd></dl><div class="evidence-flags">{html.escape('；'.join(flags) or '当前候选没有最终规则加权')}</div><details><summary>全部原始证据参数</summary><pre>{html.escape(json.dumps(evidence_payload, ensure_ascii=False, indent=2))}</pre></details></article>'''

    top_candidates = "".join(
        evidence_card(entry, compact=False, instance=f"top-{entry['id']}")
        for entry in ranked
    )
    layers = []
    medians = []
    for index, (stroke, centerline, mask) in enumerate(zip(strokes, centerlines, masks)):
        path_data = mask_svg_path(mask, canvas)
        label = (
            f"第 {index + 1} 笔 {stroke['feature']} · 部件 {stroke['componentId']} "
            f"· 兄弟族 {stroke['familyKey']}"
        )
        part_key = f"pdf:{glyph_id}:{stroke['componentId']}:{stroke['occurrence']}"
        if part_key not in registry:
            registry[part_key] = {
                "leafId": stroke["componentId"],
                "familyKey": stroke["familyKey"],
                "glyphId": glyph_id,
                "color": stroke["color"],
                "hierarchy": enrich_hierarchy(stroke["hierarchy"], stroke["color"]),
            }
        layers.append(
            f'''<path class="source-stroke interactive-part" data-part-key="{part_key}" data-family-key="{stroke['familyKey']}" data-stroke-index="{index}" data-component-id="{stroke['componentId']}" data-label="{html.escape(label, quote=True)}" d="{path_data}" style="--component-color:{stroke['color']};--stroke-color:{stroke_palette[index % len(stroke_palette)]}" fill-rule="evenodd" clip-rule="evenodd"/>'''
        )
        median_path = polyline_path(centerline, canvas)
        medians.append(
            f'<path class="median interactive-part" data-part-key="{part_key}" data-family-key="{stroke["familyKey"]}" data-stroke-index="{index}" d="{median_path}" stroke="{stroke["color"]}"/>'
        )
    ambiguity_path = mask_svg_path(ambiguous, canvas)
    nodes = marker_svg(centerlines, canvas, strokes)
    review_key = f"U+{record['unicode']:04X}-{record['source']}-{glyph_id}"
    candidate_order = "；".join(
        f"{index + 1}. {stroke['feature']}" for index, stroke in enumerate(strokes)
    )
    annotation_js = annotation_editor_script(
        {
            "reviewKey": review_key,
            "unicode": f"U+{record['unicode']:04X}",
            "character": chr(record["unicode"]),
            "source": record["source"],
            "candidateGlyphId": glyph_id,
            "candidateStrokeCount": len(strokes),
            "candidateStrokeOrder": [stroke["feature"] for stroke in strokes],
            "strokePointSemantics": "points[0] is pen-down; points[-1] is pen-up; order is directed",
        },
        initial_annotations,
    )
    original_use = f'''<use class="source-use" href="{glyph['useAttributes']['href']}" x="{glyph['useAttributes'].get('x', 0)}" y="{glyph['useAttributes'].get('y', 0)}"/>'''
    interaction_js = interaction_script(registry)
    decision_method = (result or {}).get("predictionMethod", "未载入历史评测证据")
    decision_margin = format_metric((result or {}).get("margin"))
    human_driven = metrics.get("partitionMode") == "human-stroke-and-component-truth"
    partition_title = (
        "人工真值重新分区的 PDF 墨迹归属"
        if human_driven
        else "候选驱动的 PDF 墨迹归属假设（不是已识别笔画）"
    )
    warning = (
        f"已载入人工真值：{len(strokes)} 笔、{metrics.get('humanLassoCount', 0)} 个部件圈；"
        f"按真值重分区。ID 自动纠正 {len(annotation_corrections or [])} 项。"
        if human_driven
        else f"候选强制为 {len(strokes)} 笔，但分区产生 {metrics.get('totalPartitionInkFragments', '待统计')} 个连通墨迹片；斜线歧义 {metrics['ambiguousRatio'] * 100:.2f}%。因此不得自动写入。"
    )
    correction_note = (
        "；".join(
            f"{item['display']} {item['from']}→{item['to']}"
            for item in (annotation_corrections or [])
        )
        or "无 ID 纠正"
    )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>U+{record['unicode']:04X} 逐笔迁移</title><style>
    *{{box-sizing:border-box}}body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#eef2f7;color:#172033}}button,input{{font:inherit}}header{{padding:13px 20px;background:#0f172a;color:white}}header h2{{margin:0 0 5px}}header button,.toolbar button{{margin:8px 6px 0 0;border:1px solid #94a3b8;border-radius:6px;padding:5px 9px;background:white;cursor:pointer}}header button.active,.toolbar button.active{{background:#38bdf8;border-color:#38bdf8}}.warning{{color:#fde68a;margin-top:4px}}main{{padding:14px;display:flex;flex-direction:column;gap:14px}}.review-section{{background:white;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden}}.section-title,h3{{font-size:14px;margin:0;padding:9px 11px;background:#f1f5f9}}.section-note{{padding:8px 11px;font-size:12px;color:#475569;border-bottom:1px solid #e2e8f0}}svg{{display:block;width:100%;height:auto;aspect-ratio:1}}.candidate-row{{display:flex;gap:12px;padding:12px;overflow-x:auto;align-items:stretch}}.candidate-card{{flex:1 0 340px;max-width:460px;border:1px solid #cbd5e1;border-radius:10px;overflow:hidden;background:#fff}}.candidate-visual{{height:250px;display:flex;justify-content:center}}.candidate-visual svg{{height:250px;width:auto}}.weight{{padding:7px 10px;background:#ecfeff;border-top:1px solid #a5f3fc}}.evidence-grid{{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:4px 8px;margin:0;padding:8px 10px;font-size:12px}}.evidence-grid dt{{color:#64748b}}.evidence-grid dd{{margin:0;font-variant-numeric:tabular-nums}}.evidence-flags{{padding:7px 10px;font-size:12px;background:#fefce8}}details{{border-top:1px solid #e2e8f0}}summary{{cursor:pointer;padding:7px 10px;font-size:12px}}pre{{margin:0;padding:10px;white-space:pre-wrap;font-size:11px;max-height:260px;overflow:auto}}.vector-row{{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:12px;padding:12px;align-items:start}}.vector-card{{border:1px solid #cbd5e1;border-radius:10px;overflow:hidden;background:#fff}}.vector-card>svg{{max-height:390px}}.source-fit{{fill:#111827}}.source-mask-fit{{fill:white}}.source-stroke{{fill:var(--component-color)}}body.stroke-mode .source-stroke{{fill:var(--stroke-color)}}.source-original-overlay{{display:none;fill:#111827}}body.original-source-mode .source-attribution-layer,body.original-source-mode .ambiguity,body.original-source-mode .node{{display:none}}body.original-source-mode .source-original-overlay{{display:block}}.median{{fill:none;stroke-width:.45;stroke-dasharray:1 1;opacity:.9}}.ambiguity{{fill:url(#hatch);opacity:.8;pointer-events:none}}.node{{pointer-events:none}}.endpoint{{fill:#ef4444}}.contact{{fill:white;stroke:#2563eb;stroke-width:.38}}.bend{{fill:none;stroke:#22c55e;stroke-width:.42;stroke-linecap:round}}body.nodes-hidden .node{{display:none}}.interactive-part{{cursor:pointer;transition:opacity .12s,filter .12s,stroke-width .12s}}.interactive-part.linked-highlight{{filter:drop-shadow(0 0 1.2px #020617);stroke:#020617!important;stroke-width:4.6!important;opacity:1!important}}.source-stroke.linked-highlight{{stroke-width:.5!important}}.median.linked-highlight{{stroke-width:1!important}}.interactive-part.linked-dim{{opacity:.13!important}}.editor{{scroll-margin-top:10px}}.toolbar{{padding:0 8px 7px;background:#f8fafc;border-bottom:1px solid #cbd5e1}}.toolbar button{{font-size:11px;padding:4px 6px;margin:6px 3px 0 0}}.toolbar label{{display:inline-flex;align-items:center;gap:4px;margin:6px 4px 0 0;font-size:11px}}.toolbar input[type=text]{{width:112px;padding:4px;border:1px solid #94a3b8;border-radius:5px}}.board-wrap{{height:330px;border-bottom:1px solid #cbd5e1;background:white;overflow:hidden;display:flex;justify-content:center}}#annotation-board{{height:100%;width:auto;max-width:100%;touch-action:none;cursor:crosshair;user-select:none}}.annotation-reference{{fill:#111827;opacity:.18;pointer-events:none}}.annotation-shape{{vector-effect:non-scaling-stroke}}.annotation-label{{font-size:3.2px;font-weight:700;paint-order:stroke;stroke:white;stroke-width:.7px;pointer-events:none}}.editor-help{{font-size:11px;line-height:1.35;padding:7px;display:grid;gap:6px}}#annotation-status,.help-box{{padding:6px;background:#ecfeff;border:1px solid #67e8f9;border-radius:6px}}.hierarchy-float{{position:fixed;z-index:30;min-width:330px;max-width:520px;border-radius:8px;overflow:hidden;box-shadow:0 12px 35px #0f172a55;background:#fff}}#hierarchy-tip{{pointer-events:none}}.tip-position,.picker-title{{padding:6px 8px;background:#0f172a;color:#fff;font-size:11px}}.hierarchy-row,.picker-option{{width:100%;display:flex;justify-content:space-between;gap:12px;padding:6px 8px;border:0;font-size:12px;text-align:left}}.hierarchy-row b,.picker-option b{{white-space:nowrap;align-self:center}}.picker-option{{cursor:pointer;border-top:1px solid #ffffff44}}.picker-option:hover{{outline:3px solid #38bdf8;outline-offset:-3px}}.jump{{display:inline-block;margin:8px 6px 0 0;border:1px solid #94a3b8;border-radius:6px;padding:5px 9px;background:white;color:#172033;text-decoration:none}}@media(max-width:1050px){{.vector-row{{grid-template-columns:1fr}}.board-wrap{{height:60vh}}}}
    </style></head><body><header><h2>U+{record['unicode']:04X} {chr(record['unicode'])} · {record['source']} 源 · candidate {glyph_id}</h2><div>PDF 只有最终复合轮廓；候选提供引用树，人工标注可提供真实笔画与部件边界。</div><div class="warning">{html.escape(warning)}</div><button id="component-mode" class="active">按递归叶部件聚色</button><button id="stroke-mode">按逐笔槽着色</button><button id="node-mode" class="active">显示拓扑节点</button><button id="source-view-mode">归属图 / 原始 PDF</button><a class="jump" href="#human-editor">跳到人工标注板 ↓</a></header><main>
    <section class="review-section"><h2 class="section-title">第 1 行 · 所有可能拆法（经验决策权重由高到低）</h2><div class="section-note">先以同一最终决策方法在已复核样本中的 Laplace 平滑命中率作为推荐候选的先验，其余权重再按 exp(−(总距离−最小总距离)) 分配；同时单列纯距离权重。这仍是可审计的经验估计，不是已校准概率。当前方法：{html.escape(str(decision_method))}；margin：{decision_margin}。</div><div class="candidate-row">{top_candidates}</div></section>
    <section class="review-section"><h2 class="section-title">第 2 行 · 人工真值标注、PDF 重分区、真值中心线</h2><div class="vector-row"><article class="vector-card editor"><h3>人工真值标注板（左键第一行候选，直接选择叶部件）</h3><div class="toolbar"><button type="button" data-tool="freehand" class="active">自由线</button><button type="button" data-tool="line">直线</button><button type="button" data-tool="polyline">折线</button><button type="button" data-tool="bezier">贝塞尔</button><button type="button" data-tool="lasso">圈部件</button><button type="button" data-tool="erase">删除</button><label>部件标签 <input id="component-label" type="text" placeholder="点上方候选"></label><label>颜色 <input id="annotation-color" type="color" value="#ef4444"></label><button type="button" id="undo-annotation">撤销</button><button type="button" id="redo-annotation">重做</button><button type="button" id="clear-annotations">清空</button><button type="button" id="export-annotations">导出</button><button type="button" id="import-annotations">导入</button><input id="annotation-file" type="file" accept="application/json" hidden></div><div class="board-wrap"><svg id="annotation-board" viewBox="0 0 100 100" aria-label="PDF 字源人工标注板"><defs>{glyph['definitions']}</defs><g class="annotation-reference source-fit">{original_use}</g><g id="annotation-layer"></g></svg></div><div class="editor-help"><div id="annotation-status"></div><div class="help-box"><b>快捷键：</b>Ctrl+Z 撤销；Ctrl+Y / Ctrl+Shift+Z 重做；Esc 取消当前线；折线按 Enter 或双击完成。<br><b>ID：</b>{html.escape(correction_note)}。</div></div></article><article class="vector-card"><h3>{html.escape(partition_title)}</h3><svg viewBox="0 0 100 100"><defs><pattern id="hatch" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="2" stroke="#111827" stroke-width=".25"/></pattern><mask id="pdf-outline-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100"><rect width="100" height="100" fill="black"/><g class="source-mask-fit">{original_use}</g></mask></defs><g class="source-attribution-layer" mask="url(#pdf-outline-mask)">{''.join(layers)}</g><g class="source-original-overlay source-fit">{original_use}</g><path class="ambiguity" d="{ambiguity_path}" fill-rule="evenodd"/>{nodes}</svg><div class="section-note">hover 查看层级；右键才打开只读复制列表；页首可切换原始 PDF。</div></article><article class="vector-card"><h3>{'人工真值拟合后的逐笔中心线' if human_driven else '拟合后的逐笔中心线'}</h3><svg viewBox="0 0 100 100">{''.join(medians)}</svg><details><summary>分区与拟合指标</summary><pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre></details></article></div></section>
    </main><div id="hierarchy-tip" class="hierarchy-float" hidden></div><div id="component-picker" class="hierarchy-float" hidden></div><script>
    function fit(el){{const b=el.getBBox(),s=Math.min(84/b.width,84/b.height),tx=50-s*(b.x+b.width/2),ty=50-s*(b.y+b.height/2);el.setAttribute('transform',`matrix(${{s}} 0 0 ${{s}} ${{tx}} ${{ty}})`);}}
    document.querySelectorAll('.source-fit,.source-mask-fit').forEach(fit);const componentButton=document.querySelector('#component-mode'),strokeButton=document.querySelector('#stroke-mode'),nodeButton=document.querySelector('#node-mode'),sourceButton=document.querySelector('#source-view-mode');componentButton.onclick=()=>{{document.body.classList.remove('stroke-mode');componentButton.classList.add('active');strokeButton.classList.remove('active')}};strokeButton.onclick=()=>{{document.body.classList.add('stroke-mode');strokeButton.classList.add('active');componentButton.classList.remove('active')}};nodeButton.onclick=()=>{{document.body.classList.toggle('nodes-hidden');nodeButton.classList.toggle('active')}};sourceButton.onclick=()=>{{document.body.classList.toggle('original-source-mode');sourceButton.classList.toggle('active')}};if(new URLSearchParams(location.search).get('mode')==='stroke')strokeButton.click();
    {annotation_js}
    {interaction_js}
    </script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="optional reviewed vector benchmark with per-candidate scoring evidence",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="optional exported human-truth JSON used to repartition PDF ink",
    )
    parser.add_argument("--unicode", required=True, help="hex codepoint, e.g. 6418")
    parser.add_argument("--source", required=True)
    parser.add_argument("--glyph-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=512)
    args = parser.parse_args()

    codepoint = int(args.unicode.removeprefix("U+").removeprefix("u+"), 16)
    candidate_rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    evidence = (
        json.loads(args.evidence.read_text("utf-8"))
        if args.evidence and args.evidence.exists()
        else None
    )
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
    candidate = candidate_strokes(row, args.glyph_id)
    normalized_annotations = None
    annotation_corrections = []
    if args.annotations:
        strokes, normalized_document, annotation_corrections = load_human_annotations(
            args.annotations,
            codepoint=codepoint,
            source=args.source,
            glyph_id=args.glyph_id,
            candidate=candidate,
            canvas=args.canvas,
        )
        normalized_annotations = normalized_document["annotations"]
        snapped, snap_distances = snap_centerlines(
            [stroke["points"] for stroke in strokes],
            target,
            max_distance=args.canvas * 0.045,
        )
        masks, ambiguous, metrics = partition_human_truth(
            target,
            snapped,
            strokes,
            normalized_annotations,
            args.canvas,
        )
    else:
        strokes = candidate
        fitted = fit_centerlines(strokes, args.canvas)
        snapped, snap_distances = snap_centerlines(
            fitted, target, max_distance=args.canvas * 0.08
        )
        masks, ambiguous, metrics = partition_strokes(target, snapped)
    fragment_counts = []
    for mask in masks:
        component_count, _labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8
        )
        fragment_counts.append(max(0, int(component_count) - 1))
    metrics["candidateStrokeCount"] = len(strokes)
    metrics["candidateStrokeOrder"] = [stroke["feature"] for stroke in strokes]
    metrics["pdfActualStrokeCount"] = len(strokes) if args.annotations else None
    metrics["pdfActualStrokeOrder"] = (
        normalized_document.get("metadata", {}).get("candidateStrokeOrder")
        if args.annotations
        else None
    )
    metrics["annotationLabelCorrections"] = annotation_corrections
    metrics["directedStrokeFeatures"] = [
        {
            "strokeNumber": index + 1,
            "componentId": int(stroke["componentId"]),
            "feature": stroke["feature"],
            **directed_stroke_features(centerline, args.canvas),
        }
        for index, (stroke, centerline) in enumerate(zip(strokes, snapped))
    ]
    metrics["partitionInkFragments"] = fragment_counts
    metrics["totalPartitionInkFragments"] = sum(fragment_counts)
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
        row,
        evidence,
        normalized_annotations,
        annotation_corrections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, "utf-8")
    print(json.dumps({"output": str(args.output), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
