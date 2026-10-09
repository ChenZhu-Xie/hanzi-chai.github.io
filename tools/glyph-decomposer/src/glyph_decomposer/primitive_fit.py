"""Fit one minimal geometric primitive program to one directed skeleton route.

The candidate's SVG command sequence is a hard grammar. Fitting may move the
control points, but it may not add another segment or explain a spur with an
extra stroke. This module is deterministic and has no learned parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

Point = tuple[float, float]

_SAMPLES_PER_COMMAND = {"h": 1, "v": 1, "l": 1, "c": 16, "z": 16, "a": 24}
_LINE_COMMANDS = frozenset(("h", "v", "l"))
_CUBIC_COMMANDS = frozenset(("c", "z"))


@dataclass(frozen=True)
class FittedPrimitive:
    command: str
    kind: str
    controls: tuple[Point, ...]
    samples: tuple[Point, ...]
    route_start: int
    route_end: int
    squared_error: float


@dataclass(frozen=True)
class FittedStroke:
    commands: tuple[str, ...]
    primitives: tuple[FittedPrimitive, ...]
    route: tuple[Point, ...]
    fitted_path: tuple[Point, ...]
    rmse: float
    maximum_error: float
    control_point_count: int
    compression_ratio: float


def _lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros(len(points)), 0.0
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return cumulative, float(cumulative[-1])


def resample_polyline(points, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot resample an empty route")
    if len(values) == 1 or count <= 1:
        return values[:1].copy()
    cumulative, total = _lengths(values)
    if total <= 1e-12:
        return np.repeat(values[:1], count, axis=0)
    targets = np.linspace(0.0, total, count)
    output = np.empty((count, 2), dtype=float)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 == len(cumulative):
            output[index] = values[-1]
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        ratio = 0.0 if span <= 1e-12 else (target - cumulative[segment]) / span
        output[index] = values[segment] * (1 - ratio) + values[segment + 1] * ratio
    return output


def _parameterize(points: np.ndarray) -> np.ndarray:
    cumulative, total = _lengths(points)
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _fit_line(points: np.ndarray, command: str, start: int, end: int):
    # A line primitive represents the complete directed pen backbone, not a
    # chord aimed from one calligraphic spur to another.  Total least squares
    # keeps exactly two controls while allowing a short serif at either end to
    # contribute ink without rotating the whole backbone towards that serif.
    centre = points.mean(axis=0)
    _u, _singular, vectors = np.linalg.svd(points - centre, full_matrices=False)
    axis = vectors[0]
    if float(np.dot(axis, points[-1] - points[0])) < 0:
        axis = -axis
    parameters = (points - centre) @ axis
    first = centre + float(parameters.min()) * axis
    last = centre + float(parameters.max()) * axis
    controls = (tuple(first), tuple(last))
    prediction = centre + parameters[:, None] * axis
    error = float(np.sum((points - prediction) ** 2))
    return FittedPrimitive(
        command,
        "line",
        controls,
        tuple(map(tuple, prediction)),
        start,
        end,
        error,
    )


def _fit_cubic(points: np.ndarray, command: str, start: int, end: int):
    p0, p3 = points[0], points[-1]
    t = _parameterize(points)
    chord_controls = np.vstack((p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3))
    p1, p2 = chord_controls
    regularization = 0.001 * len(points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    control_minimum = minimum - span
    control_maximum = maximum + span
    for _ in range(8):
        u = 1 - t
        matrix = np.column_stack((3 * u * u * t, 3 * u * t * t))
        fixed = u[:, None] ** 3 * p0 + t[:, None] ** 3 * p3
        target = points - fixed
        lhs = matrix.T @ matrix + regularization * np.eye(2)
        rhs = matrix.T @ target + regularization * chord_controls
        p1, p2 = np.linalg.solve(lhs, rhs)
        p1 = np.clip(p1, control_minimum, control_maximum)
        p2 = np.clip(p2, control_minimum, control_maximum)
        prediction = (
            u[:, None] ** 3 * p0
            + 3 * u[:, None] ** 2 * t[:, None] * p1
            + 3 * u[:, None] * t[:, None] ** 2 * p2
            + t[:, None] ** 3 * p3
        )
        first_derivative = (
            3 * u[:, None] ** 2 * (p1 - p0)
            + 6 * u[:, None] * t[:, None] * (p2 - p1)
            + 3 * t[:, None] ** 2 * (p3 - p2)
        )
        second_derivative = (
            6 * u[:, None] * (p2 - 2 * p1 + p0)
            + 6 * t[:, None] * (p3 - 2 * p2 + p1)
        )
        residual = prediction - points
        numerator = np.sum(residual * first_derivative, axis=1)
        denominator = np.sum(first_derivative * first_derivative, axis=1) + np.sum(
            residual * second_derivative, axis=1
        )
        safe = np.where(np.abs(denominator) < 1e-9, 1.0, denominator)
        updated = np.clip(t - numerator / safe, 0.0, 1.0)
        updated[0], updated[-1] = 0.0, 1.0
        # Closest-point updates must not reverse the directed pen trajectory.
        t = np.maximum.accumulate(updated)
        t[-1] = 1.0
    u = 1 - t
    prediction = (
        u[:, None] ** 3 * p0
        + 3 * u[:, None] ** 2 * t[:, None] * p1
        + 3 * u[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )
    error = float(np.sum((points - prediction) ** 2))
    return FittedPrimitive(
        command,
        "cubic",
        tuple(map(tuple, (p0, p1, p2, p3))),
        tuple(map(tuple, prediction)),
        start,
        end,
        error,
    )


def _fit_arc(points: np.ndarray, command: str, start: int, end: int):
    x, y = points[:, 0], points[:, 1]
    matrix = np.column_stack((x, y, np.ones(len(points))))
    solution, *_ = np.linalg.lstsq(matrix, -(x * x + y * y), rcond=None)
    centre = np.asarray((-solution[0] / 2, -solution[1] / 2))
    radius = float(max(0.0, np.mean(np.linalg.norm(points - centre, axis=1))))
    angles = np.unwrap(np.arctan2(y - centre[1], x - centre[0]))
    sampled_angles = np.linspace(float(angles[0]), float(angles[-1]), len(points))
    prediction = centre + radius * np.column_stack(
        (np.cos(sampled_angles), np.sin(sampled_angles))
    )
    error = float(np.sum((points - prediction) ** 2))
    return FittedPrimitive(
        command,
        "arc",
        (tuple(centre), (radius, float(angles[0])), (radius, float(angles[-1]))),
        tuple(map(tuple, prediction)),
        start,
        end,
        error,
    )


def _fit_segment(points: np.ndarray, command: str, start: int, end: int):
    if command in _LINE_COMMANDS:
        return _fit_line(points, command, start, end)
    if command in _CUBIC_COMMANDS:
        return _fit_cubic(points, command, start, end)
    if command == "a":
        return _fit_arc(points, command, start, end)
    raise ValueError(f"unsupported stroke primitive command {command!r}")


def _command_spans(
    commands: tuple[str, ...], expected_points
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    values = np.asarray(expected_points, dtype=float)
    counts = [_SAMPLES_PER_COMMAND[command] for command in commands]
    if len(values) != 1 + sum(counts):
        raise ValueError("candidate samples do not match its command grammar")
    spans = []
    cursor = 0
    lengths = []
    for count in counts:
        part = values[cursor : cursor + count + 1]
        spans.append(part)
        lengths.append(_lengths(part)[1])
        cursor += count
    total = sum(lengths)
    ratios = tuple(value / total if total else 1 / len(lengths) for value in lengths)
    return tuple(spans), ratios


def _direction_penalty(observed: np.ndarray, expected: np.ndarray) -> float:
    observed_length = float(np.linalg.norm(observed))
    expected_length = float(np.linalg.norm(expected))
    if observed_length <= 1e-9 or expected_length <= 1e-9:
        return 4.0
    cosine = float(
        np.clip(
            np.dot(observed, expected) / (observed_length * expected_length), -1, 1
        )
    )
    return 2.0 * (1.0 - cosine)


def _sample_primitive(primitive: FittedPrimitive, count: int = 101) -> np.ndarray:
    controls = np.asarray(primitive.controls, dtype=float)
    if primitive.kind == "line":
        t = np.linspace(0.0, 1.0, count)[:, None]
        return controls[0] * (1 - t) + controls[1] * t
    if primitive.kind == "cubic":
        t = np.linspace(0.0, 1.0, count)
        u = 1 - t
        return (
            u[:, None] ** 3 * controls[0]
            + 3 * u[:, None] ** 2 * t[:, None] * controls[1]
            + 3 * u[:, None] * t[:, None] ** 2 * controls[2]
            + t[:, None] ** 3 * controls[3]
        )
    centre = controls[0]
    radius, first = controls[1]
    _, last = controls[2]
    angle = np.linspace(first, last, count)
    return centre + radius * np.column_stack((np.cos(angle), np.sin(angle)))


def _distances_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    best = np.full(len(points), math.inf)
    for start, end in pairwise(polyline):
        vector = end - start
        denominator = float(np.dot(vector, vector))
        if denominator <= 1e-12:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            ratios = np.clip(((points - start) @ vector) / denominator, 0, 1)
            projections = start + ratios[:, None] * vector
            distance = np.linalg.norm(points - projections, axis=1)
        best = np.minimum(best, distance)
    return best


def fit_stroke_primitives(
    route_points,
    commands: tuple[str, ...],
    expected_points,
    *,
    expected_vectors=None,
    segment_ratios=None,
    maximum_fit_samples: int = 81,
) -> FittedStroke:
    """Fit exactly one candidate command program to a directed route."""
    if not commands:
        raise ValueError("a stroke must have at least one primitive command")
    original = np.asarray(route_points, dtype=float)
    if len(original) < len(commands) + 1:
        raise ValueError("route has fewer edges than the fixed primitive grammar")
    route = (
        resample_polyline(original, maximum_fit_samples)
        if len(original) > maximum_fit_samples
        else original.copy()
    )
    if expected_vectors is None or segment_ratios is None:
        expected_spans, default_ratios = _command_spans(commands, expected_points)
        default_vectors = tuple(part[-1] - part[0] for part in expected_spans)
        expected_vectors = expected_vectors or default_vectors
        segment_ratios = segment_ratios or default_ratios
    expected_vectors = tuple(np.asarray(vector, dtype=float) for vector in expected_vectors)
    expected_ratios = tuple(float(value) for value in segment_ratios)
    if len(expected_vectors) != len(commands) or len(expected_ratios) != len(commands):
        raise ValueError("stroke grammar vectors and ratios must match its commands")
    cumulative, total_length = _lengths(route)
    expected_boundaries = np.cumsum(expected_ratios)

    cache: dict[tuple[int, int, int], tuple[float, FittedPrimitive]] = {}

    def option(command_index: int, start: int, end: int):
        key = command_index, start, end
        if key in cache:
            return cache[key]
        primitive = _fit_segment(
            route[start : end + 1], commands[command_index], start, end
        )
        direction = _direction_penalty(
            route[end] - route[start], expected_vectors[command_index]
        )
        boundary = 0.0
        if command_index + 1 < len(commands) and total_length:
            observed_fraction = cumulative[end] / total_length
            boundary = (
                4.0
                * total_length
                * (observed_fraction - expected_boundaries[command_index]) ** 2
            )
        score = primitive.squared_error + direction + boundary
        cache[key] = score, primitive
        return cache[key]

    states: dict[int, tuple[float, tuple[FittedPrimitive, ...]]] = {0: (0.0, ())}
    last_index = len(route) - 1
    for command_index in range(len(commands)):
        remaining = len(commands) - command_index - 1
        next_states: dict[int, tuple[float, tuple[FittedPrimitive, ...]]] = {}
        for start, (prior_score, prior) in states.items():
            minimum_end = start + 1
            maximum_end = last_index - remaining
            for end in range(minimum_end, maximum_end + 1):
                if command_index == len(commands) - 1 and end != last_index:
                    continue
                score, primitive = option(command_index, start, end)
                proposed = prior_score + score
                current = next_states.get(end)
                if current is None or proposed < current[0]:
                    next_states[end] = proposed, (*prior, primitive)
        states = next_states
    if last_index not in states:
        raise ValueError("fixed primitive grammar has no valid route partition")
    primitives = states[last_index][1]
    dense_parts = [_sample_primitive(primitive) for primitive in primitives]
    fitted = np.vstack(
        [part if index == 0 else part[1:] for index, part in enumerate(dense_parts)]
    )
    distances = _distances_to_polyline(original, fitted)
    point_count = 1 + sum(
        1 if item.kind == "line" else 3 if item.kind == "cubic" else 2
        for item in primitives
    )
    return FittedStroke(
        commands=commands,
        primitives=primitives,
        route=tuple(map(tuple, original)),
        fitted_path=tuple(map(tuple, fitted)),
        rmse=float(np.sqrt(np.mean(distances * distances))),
        maximum_error=float(np.max(distances)),
        control_point_count=point_count,
        compression_ratio=len(original) / point_count,
    )


def symmetric_chamfer(first, second) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    first_distance = _distances_to_polyline(first_values, second_values)
    second_distance = _distances_to_polyline(second_values, first_values)
    return float((first_distance.mean() + second_distance.mean()) / 2)
