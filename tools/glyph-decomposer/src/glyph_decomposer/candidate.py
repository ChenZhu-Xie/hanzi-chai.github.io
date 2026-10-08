"""Compile candidate glyph strokes into directed, leaf-owned vector seeds."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grammar import GlyphRepository

IDS_INTERVALS = {
    "⿰": ((0, 0, 50, 100), (50, 0, 100, 100)),
    "⿱": ((0, 0, 100, 50), (0, 50, 100, 100)),
    "⿸": ((0, 0, 100, 100), (40, 40, 90, 90)),
}


@dataclass(frozen=True)
class StrokeSeed:
    leaf_id: int
    occurrence: int
    feature: str
    points: tuple[tuple[float, float], ...]
    component_path: tuple[int, ...] = ()


def _point(transform, point):
    sx, sy, tx, ty = transform
    return point[0] * sx + tx, point[1] * sy + ty


def _compose(transform, interval):
    sx, sy, tx, ty = transform
    x0, y0, x1, y1 = interval
    return sx * (x1 - x0) / 100, sy * (y1 - y0) / 100, tx + sx * x0, ty + sy * y0


def _cubic(start, values, steps: int = 16):
    x0, y0 = start
    x1, y1, x2, y2, x3, y3 = values
    control1 = (x0 + x1, y0 + y1)
    control2 = (x0 + x2, y0 + y2)
    end = (x0 + x3, y0 + y3)
    output = []
    for index in range(1, steps + 1):
        t = index / steps
        u = 1 - t
        output.append(
            (
                u**3 * x0
                + 3 * u * u * t * control1[0]
                + 3 * u * t * t * control2[0]
                + t**3 * end[0],
                u**3 * y0
                + 3 * u * u * t * control1[1]
                + 3 * u * t * t * control2[1]
                + t**3 * end[1],
            )
        )
    return output, end


def stroke_points(stroke: dict) -> list[tuple[float, float]]:
    current = tuple(map(float, stroke["start"]))
    points = [current]
    for curve in stroke.get("curveList", ()):
        command = curve["command"]
        values = list(map(float, curve["parameterList"]))
        if command == "h":
            current = current[0] + values[0], current[1]
            points.append(current)
        elif command == "v":
            current = current[0], current[1] + values[0]
            points.append(current)
        elif command == "l":
            current = current[0] + values[0], current[1] + values[1]
            points.append(current)
        elif command in {"c", "z"}:
            sampled, current = _cubic(current, values)
            points.extend(sampled)
        elif command == "a":
            radius = values[0]
            points.extend(
                (
                    current[0] + radius * math.cos(2 * math.pi * index / 24),
                    current[1] + radius * math.sin(2 * math.pi * index / 24),
                )
                for index in range(1, 25)
            )
    return points


def compile_stroke_seeds(
    repository: GlyphRepository, glyph_id: int
) -> tuple[StrokeSeed, ...]:
    occurrence_count: dict[int, int] = {}

    def compile_one(
        identifier: int,
        transform,
        active: tuple[int, ...],
        component_path: tuple[int, ...],
    ):
        if identifier in active:
            raise ValueError(f"cyclic glyph references through {identifier}")
        record = repository.record(identifier)
        if record["type"] == "component":
            occurrence = occurrence_count.get(identifier, 0)
            occurrence_count[identifier] = occurrence + 1
            return [
                StrokeSeed(
                    leaf_id=identifier,
                    occurrence=occurrence,
                    feature=stroke.get("feature", "unknown"),
                    points=tuple(
                        _point(transform, point) for point in stroke_points(stroke)
                    ),
                    component_path=component_path,
                )
                for stroke in record.get("strokes", ())
            ]
        operator = record.get("operator")
        intervals = IDS_INTERVALS.get(operator)
        references = record.get("references", ())
        if intervals is None or len(intervals) != len(references):
            raise ValueError(f"stroke seed compiler does not support {operator!r}")
        parts = [
            compile_one(
                int(reference["id"]),
                _compose(transform, interval),
                (*active, identifier),
                (*component_path, index),
            )
            for index, (reference, interval) in enumerate(zip(references, intervals))
        ]
        selectors = record.get("strokes")
        if not selectors:
            return [seed for part in parts for seed in part]
        output = []
        for selector in selectors:
            part = parts[int(selector["index"])]
            start = int(selector.get("from", 0))
            end = int(selector.get("to", len(part) - 1)) + 1
            output.extend(part[start:end])
        return output

    return tuple(compile_one(glyph_id, (1.0, 1.0, 0.0, 0.0), (), ()))
