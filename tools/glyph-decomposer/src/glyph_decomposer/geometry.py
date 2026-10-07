"""Vector path conversion and geometry normalization."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise

from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from svgpathtools import Path as SvgPath
from svgpathtools import parse_path


def _signed_area(points: list[tuple[float, float]]) -> float:
    return (
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in pairwise(points)
        )
        / 2
    )


def _flatten(subpath: SvgPath, tolerance: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for segment in subpath:
        try:
            length = float(segment.length(error=tolerance / 4))
        except (ValueError, ZeroDivisionError):
            length = abs(segment.end - segment.start)
        steps = max(1, min(256, math.ceil(length / tolerance)))
        sampled = [segment.point(index / steps) for index in range(steps + 1)]
        if points:
            sampled = sampled[1:]
        points.extend((float(point.real), float(point.imag)) for point in sampled)
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def svg_path_geometry(path_data: str, tolerance: float = 0.08):
    """Convert one non-zero-filled SVG path into a Shapely geometry."""
    path = parse_path(path_data)
    rings: list[tuple[list[tuple[float, float]], int]] = []
    lines = []
    for subpath in path.continuous_subpaths():
        points = _flatten(subpath, tolerance)
        if len(points) < 4:
            continue
        area = _signed_area(points)
        if abs(area) <= tolerance * tolerance:
            continue
        rings.append((points, 1 if area > 0 else -1))
        lines.append(LineString(points))
    if not rings:
        return GeometryCollection()

    faces = list(polygonize(unary_union(lines)))
    filled = []
    for face in faces:
        probe = face.representative_point()
        winding = 0
        for points, direction in rings:
            if Polygon(points).covers(probe):
                winding += direction
        if winding:
            filled.append(face)
    return unary_union(filled) if filled else GeometryCollection()


def union_geometries(geometries: Iterable):
    present = [geometry for geometry in geometries if not geometry.is_empty]
    return unary_union(present) if present else GeometryCollection()


def normalize_geometry(geometry, size: float = 100.0, margin: float = 2.0):
    if geometry.is_empty:
        return geometry
    min_x, min_y, max_x, max_y = geometry.bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("cannot normalize degenerate geometry")
    target = size - 2 * margin
    scale = min(target / width, target / height)
    moved = affinity.translate(geometry, xoff=-min_x, yoff=-min_y)
    scaled = affinity.scale(moved, xfact=scale, yfact=scale, origin=(0, 0))
    out_x = margin + (target - width * scale) / 2
    out_y = margin + (target - height * scale) / 2
    return affinity.translate(scaled, xoff=out_x, yoff=out_y)


def polygon_parts(geometry) -> int:
    if geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return 1
    if geometry.geom_type == "MultiPolygon":
        return len(geometry.geoms)
    return sum(1 for item in geometry.geoms if item.geom_type == "Polygon")


def clip_axis(geometry, axis: str, cut: float):
    world = 1_000_000.0
    if axis == "x":
        return (
            geometry.intersection(box(-world, -world, cut, world)),
            geometry.intersection(box(cut, -world, world, world)),
        )
    if axis == "y":
        return (
            geometry.intersection(box(-world, -world, world, cut)),
            geometry.intersection(box(-world, cut, world, world)),
        )
    raise ValueError(f"unsupported axis {axis!r}")
