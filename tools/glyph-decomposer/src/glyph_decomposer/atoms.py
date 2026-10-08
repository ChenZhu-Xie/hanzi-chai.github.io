"""Mutually exclusive vector ink atoms induced by candidate stroke seeds."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

from shapely import make_valid, voronoi_polygons
from shapely.geometry import LineString, MultiPoint, box
from shapely.ops import unary_union

from .candidate import StrokeSeed
from .domain import PartitionEvidence
from .solver import RootPartition


def _align_seeds(seeds: tuple[StrokeSeed, ...], geometry) -> tuple[StrokeSeed, ...]:
    all_points = [point for seed in seeds for point in seed.points]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    target = geometry.bounds
    scale_x = (target[2] - target[0]) / max(max_x - min_x, 1e-9)
    scale_y = (target[3] - target[1]) / max(max_y - min_y, 1e-9)
    return tuple(
        StrokeSeed(
            seed.leaf_id,
            seed.occurrence,
            seed.feature,
            tuple(
                (
                    target[0] + (point[0] - min_x) * scale_x,
                    target[1] + (point[1] - min_y) * scale_y,
                )
                for point in seed.points
            ),
        )
        for seed in seeds
    )


def _sample_seed(seed: StrokeSeed, spacing: float = 1.5):
    line = LineString(seed.points)
    count = max(2, round(line.length / spacing) + 1)
    return [
        line.interpolate(index / (count - 1), normalized=True) for index in range(count)
    ]


def _transform_seeds(seeds, center, scale_x, scale_y, offset_x, offset_y):
    return tuple(
        StrokeSeed(
            seed.leaf_id,
            seed.occurrence,
            seed.feature,
            tuple(
                (
                    center[0] + (point[0] - center[0]) * scale_x + offset_x,
                    center[1] + (point[1] - center[1]) * scale_y + offset_y,
                )
                for point in seed.points
            ),
        )
        for seed in seeds
    )


def _alignment_iou(geometry, seeds) -> float:
    lines = [LineString(seed.points) for seed in seeds]
    total_length = sum(line.length for line in lines)
    radius = min(7.0, max(1.0, geometry.area / max(2 * total_length, 1e-9)))
    rendered = unary_union(
        [line.buffer(radius, cap_style="square", join_style="round") for line in lines]
    )
    union_area = rendered.union(geometry).area
    return rendered.intersection(geometry).area / union_area if union_area else 0.0


def _optimize_alignment(geometry, seeds):
    min_x, min_y, max_x, max_y = geometry.bounds
    center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    span_x = max_x - min_x
    span_y = max_y - min_y
    candidates = []
    for scale_x, scale_y, offset_x, offset_y in product(
        (0.9, 1.0, 1.1),
        (0.9, 1.0, 1.1),
        (-0.04 * span_x, 0.0, 0.04 * span_x),
        (-0.04 * span_y, 0.0, 0.04 * span_y),
    ):
        transformed = _transform_seeds(
            seeds, center, scale_x, scale_y, offset_x, offset_y
        )
        candidates.append((_alignment_iou(geometry, transformed), transformed))
    return max(candidates, key=lambda item: item[0])


def _seed_owner(seed: StrokeSeed, child_leaf_groups) -> int:
    matches = [
        index for index, group in enumerate(child_leaf_groups) if seed.leaf_id in group
    ]
    if len(matches) != 1:
        raise ValueError(
            f"stroke leaf {seed.leaf_id} must belong to exactly one child, got {matches}"
        )
    return matches[0]


def _partition_from_seeds(geometry, aligned, child_leaf_groups):
    points = []
    owners = []
    seen = set()
    for seed in aligned:
        owner = _seed_owner(seed, child_leaf_groups)
        for point in _sample_seed(seed):
            key = round(point.x, 4), round(point.y, 4)
            if key in seen:
                continue
            seen.add(key)
            points.append(point)
            owners.append(owner)
    if len(points) < 2 or len(set(owners)) != 2:
        raise ValueError("both children need distinct vector seed points")

    min_x, min_y, max_x, max_y = geometry.bounds
    extent = box(min_x - 10, min_y - 10, max_x + 10, max_y + 10)
    cells = voronoi_polygons(MultiPoint(points), extend_to=extent, ordered=True).geoms
    by_owner = defaultdict(list)
    for cell, owner in zip(cells, owners):
        atom = make_valid(cell).intersection(geometry)
        if not atom.is_empty:
            by_owner[owner].append(atom)
    children = tuple(unary_union(by_owner[index]) for index in range(2))
    return children, points, cells


def assign_vector_atoms(
    geometry,
    seeds: tuple[StrokeSeed, ...],
    child_leaf_groups: tuple[frozenset[int], frozenset[int]],
) -> RootPartition:
    if not seeds:
        raise ValueError("candidate contains no stroke seeds")
    initial = _align_seeds(seeds, geometry)
    alignment_iou, aligned = _optimize_alignment(geometry, initial)
    children, points, cells = _partition_from_seeds(
        geometry, aligned, child_leaf_groups
    )
    # One EM-like self-consistency pass: each child may adapt to the ink it
    # currently owns, but the pass is accepted only when the global, truth-free
    # rendered-centerline IoU improves.
    refined_groups = []
    for index, child in enumerate(children):
        group = tuple(
            seed for seed in aligned if _seed_owner(seed, child_leaf_groups) == index
        )
        refined_groups.extend(_align_seeds(group, child))
    refined = tuple(refined_groups)
    refined_iou = _alignment_iou(geometry, refined)
    if refined_iou > alignment_iou + 1e-6:
        aligned = refined
        alignment_iou = refined_iou
        children, points, cells = _partition_from_seeds(
            geometry, aligned, child_leaf_groups
        )
    if any(child.is_empty for child in children):
        raise ValueError("vector atom assignment produced an empty child")

    covered = unary_union(children)
    reconstruction_error = covered.symmetric_difference(geometry).area / geometry.area
    overlap_ratio = children[0].intersection(children[1]).area / geometry.area
    if reconstruction_error > 1e-8 or overlap_ratio > 1e-8:
        raise ValueError("vector atoms are not a complete mutually exclusive partition")
    seed_coverage = sum(geometry.buffer(1.5).covers(point) for point in points) / len(
        points
    )
    expected_ratio = sum(seed.leaf_id in child_leaf_groups[0] for seed in seeds) / len(
        seeds
    )
    observed_ratios = tuple(child.area / geometry.area for child in children)
    minimum_expected = min(expected_ratio, 1 - expected_ratio) * 0.25
    if seed_coverage < 0.75 or alignment_iou < 0.25:
        raise ValueError(
            "candidate centerlines do not align with enough PDF ink "
            f"(seed coverage {seed_coverage:.3%}, alignment IoU {alignment_iou:.3%})"
        )
    if min(observed_ratios) < minimum_expected:
        raise ValueError("vector atom assignment collapses a real child program")
    balance_error = abs(children[0].area / geometry.area - expected_ratio)
    return RootPartition(
        evidence=PartitionEvidence(
            axis="atoms",
            cut=None,
            score=(1 - seed_coverage) + balance_error,
            crossingRatio=overlap_ratio,
            balanceError=balance_error,
            reconstructionError=reconstruction_error,
            solverStatus="VECTOR_VORONOI",
            seedCoverage=seed_coverage,
            alignmentIoU=alignment_iou,
            atomCount=len(cells),
        ),
        children=children,
    )
