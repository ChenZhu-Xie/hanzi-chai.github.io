"""Mutually exclusive vector ink atoms induced by candidate stroke seeds."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

from shapely import make_valid, voronoi_polygons
from shapely.geometry import LineString, MultiPoint, Point, box
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
            seed.component_path,
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
            seed.component_path,
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


def _intersection_points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [geometry.interpolate(0.5, normalized=True)]
    return [point for part in geometry.geoms for point in _intersection_points(part)]


def _polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if not hasattr(geometry, "geoms"):
        return []
    return [part for item in geometry.geoms for part in _polygon_parts(item)]


def _inertial_atom_owners(geometry, cells, site_stroke_owners, lines):
    """Grow mutually exclusive ink ownership from whole-stroke vector cores.

    A Voronoi cell may intersect several disconnected pieces of PDF ink.  The
    old point-site assignment gave every such piece the site's colour, which
    produced remote speckles.  Here each connected vector atom is considered
    separately.  Atoms touching a complete directed candidate line form the
    trusted stroke cores; all remaining ink grows from the nearest such core.
    """
    atoms = []
    initial_owners = []
    for cell, owner in zip(cells, site_stroke_owners):
        for part in _polygon_parts(make_valid(cell).intersection(geometry)):
            if part.area > 1e-9:
                atoms.append(part)
                initial_owners.append(owner)
    anchored_by_stroke = [[] for _ in lines]
    core_tubes = [line.buffer(1.0, cap_style="flat") for line in lines]
    is_anchored = []
    for atom, owner in zip(atoms, initial_owners):
        # The PDF centerline can sit a fraction outside a serif outline after
        # truth-free alignment.  A narrow tube keeps that numerical detail
        # from breaking an otherwise continuous stroke core.
        anchored = core_tubes[owner].intersects(atom)
        is_anchored.append(anchored)
        if anchored:
            anchored_by_stroke[owner].append(atom)

    missing = [index for index, owned in enumerate(anchored_by_stroke) if not owned]
    if missing:
        raise ValueError(f"candidate strokes have no PDF-ink anchor: {missing}")
    cores = tuple(unary_union(owned) for owned in anchored_by_stroke)

    owners = []
    orphan_area = 0.0
    inertial_area = 0.0
    for atom, owner, anchored in zip(atoms, initial_owners, is_anchored):
        probe = atom.representative_point()
        if anchored:
            inertial_area += atom.area
        else:
            orphan_area += atom.area
            owner = min(
                range(len(lines)),
                key=lambda index: (probe.distance(cores[index]), -index),
            )
        owners.append(owner)
    return (
        atoms,
        owners,
        inertial_area / geometry.area,
        orphan_area / geometry.area,
    )


def _partition_from_seeds(geometry, aligned, child_leaf_groups):
    sites = {}
    lines = [LineString(seed.points) for seed in aligned]
    for stroke_index, seed in enumerate(aligned):
        owner = _seed_owner(seed, child_leaf_groups)
        for point in _sample_seed(seed):
            key = round(point.x, 4), round(point.y, 4)
            # Later strokes overwrite an exact duplicate site. This models
            # calligraphic writing order without merging the two stroke IDs.
            sites[key] = (point, stroke_index, owner)
    junctions = []
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            for point in _intersection_points(first.intersection(lines[second_index])):
                key = round(point.x, 4), round(point.y, 4)
                later = second_index
                sites[key] = (
                    Point(point.x, point.y),
                    later,
                    _seed_owner(aligned[later], child_leaf_groups),
                )
                junctions.append(point)
    points = [site[0] for site in sites.values()]
    stroke_owners = [site[1] for site in sites.values()]
    child_owners = [site[2] for site in sites.values()]
    if len(points) < 2 or len(set(child_owners)) != 2:
        raise ValueError("both children need distinct vector seed points")

    min_x, min_y, max_x, max_y = geometry.bounds
    extent = box(min_x - 10, min_y - 10, max_x + 10, max_y + 10)
    cells = voronoi_polygons(MultiPoint(points), extend_to=extent, ordered=True).geoms
    atoms, stroke_owners, stroke_inertia, orphan_ink_ratio = _inertial_atom_owners(
        geometry, cells, stroke_owners, lines
    )
    by_stroke = defaultdict(list)
    for atom, stroke_owner in zip(atoms, stroke_owners):
        by_stroke[stroke_owner].append(atom)
    stroke_regions = tuple(
        unary_union(by_stroke[index]) for index in range(len(aligned))
    )
    children = tuple(
        unary_union(
            [
                region
                for seed, region in zip(aligned, stroke_regions)
                if _seed_owner(seed, child_leaf_groups) == child_index
            ]
        )
        for child_index in range(2)
    )
    return (
        children,
        points,
        atoms,
        stroke_regions,
        tuple(junctions),
        stroke_inertia,
        orphan_ink_ratio,
    )


def _stroke_metrics(geometry, aligned, stroke_regions, junctions):
    lines = [LineString(seed.points) for seed in aligned]
    total_length = sum(line.length for line in lines)
    junction_radius = min(7.0, max(1.0, geometry.area / max(2 * total_length, 1e-9)))
    junction_area = unary_union([point.buffer(junction_radius) for point in junctions])
    continuity_scores = []
    for line, region in zip(lines, stroke_regions):
        visible = line.intersection(geometry.buffer(1.0))
        if visible.length <= 1e-9:
            continuity_scores.append(0.0)
            continue
        allowed = region.buffer(1.0).union(junction_area)
        continuity_scores.append(visible.intersection(allowed).length / visible.length)
    overlap = 0.0
    for index, region in enumerate(stroke_regions):
        for other in stroke_regions[index + 1 :]:
            overlap += region.intersection(other).area
    return min(continuity_scores, default=0.0), max(0.0, 1 - overlap / geometry.area)


def assign_vector_atoms(
    geometry,
    seeds: tuple[StrokeSeed, ...],
    child_leaf_groups: tuple[frozenset[int], frozenset[int]],
    *,
    minimum_seed_coverage: float = 0.75,
) -> RootPartition:
    if not seeds:
        raise ValueError("candidate contains no stroke seeds")
    initial = _align_seeds(seeds, geometry)
    alignment_iou, aligned = _optimize_alignment(geometry, initial)
    (
        children,
        points,
        atoms,
        stroke_regions,
        junctions,
        stroke_inertia,
        orphan_ink_ratio,
    ) = _partition_from_seeds(geometry, aligned, child_leaf_groups)
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
        (
            children,
            points,
            atoms,
            stroke_regions,
            junctions,
            stroke_inertia,
            orphan_ink_ratio,
        ) = _partition_from_seeds(geometry, aligned, child_leaf_groups)
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
    if seed_coverage < minimum_seed_coverage or alignment_iou < 0.25:
        raise ValueError(
            "candidate centerlines do not align with enough PDF ink "
            f"(seed coverage {seed_coverage:.3%}, alignment IoU {alignment_iou:.3%})"
        )
    if min(observed_ratios) < minimum_expected:
        raise ValueError("vector atom assignment collapses a real child program")
    balance_error = abs(children[0].area / geometry.area - expected_ratio)
    stroke_continuity, stroke_independence = _stroke_metrics(
        geometry, aligned, stroke_regions, junctions
    )
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
            atomCount=len(atoms),
            strokeCount=len(aligned),
            junctionCount=len(junctions),
            strokeContinuity=stroke_continuity,
            strokeIndependence=stroke_independence,
            strokeInertia=stroke_inertia,
            orphanInkRatio=orphan_ink_ratio,
        ),
        children=children,
        stroke_regions=stroke_regions,
        stroke_labels=tuple(
            f"{seed.leaf_id} · 第{index + 1}笔 · {seed.feature}"
            for index, seed in enumerate(aligned)
        ),
    )
