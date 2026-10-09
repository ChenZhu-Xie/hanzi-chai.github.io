"""Blind, deterministic affine hypotheses for recursive IDS leaf components."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .candidate_graph import CandidateGraph
from .primitive_fit import resample_polyline
from .route_cover import _align_points, _candidate_bounds, _skeleton_bounds


@dataclass(frozen=True)
class LeafAlignment:
    component_path: tuple[int, ...]
    bounds: tuple[float, float, float, float]
    score: float
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    claimed_pixels: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class JointLeafAlignmentSolution:
    alignments: tuple[LeafAlignment, ...]
    score: float
    repeated_claim_count: int
    ids_structure_cost: float


def _leaf_points(
    candidate: CandidateGraph, component_path: tuple[int, ...]
) -> np.ndarray:
    samples = []
    for stroke in candidate.strokes:
        if stroke.component_path != component_path:
            continue
        samples.extend(resample_polyline(stroke.points, 33))
    if not samples:
        raise ValueError(f"candidate has no strokes for leaf path {component_path}")
    return np.asarray(samples, dtype=float)


def _score_points(
    normalized: np.ndarray,
    bounds: tuple[float, float, float, float],
    distances: np.ndarray,
) -> float:
    x0, y0, x1, y1 = bounds
    points = np.column_stack(
        (x0 + normalized[:, 0] * (x1 - x0), y0 + normalized[:, 1] * (y1 - y0))
    )
    height, width = distances.shape
    xs = np.clip(np.rint(points[:, 0] * width / 100 - 0.5), 0, width - 1).astype(int)
    ys = np.clip(np.rint(points[:, 1] * height / 100 - 0.5), 0, height - 1).astype(int)
    values = distances[ys, xs] * 100 / height
    return float(values.mean() + 0.35 * np.quantile(values, 0.9))


def _transformed_points(
    normalized: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return np.column_stack(
        (x0 + normalized[:, 0] * (x1 - x0), y0 + normalized[:, 1] * (y1 - y0))
    )


def _hypothesis_bounds(
    base: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = base
    center_x = (x0 + x1) / 2 + offset_x
    center_y = (y0 + y1) / 2 + offset_y
    half_width = (x1 - x0) * scale_x / 2
    half_height = (y1 - y0) * scale_y / 2
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _inside_canvas(bounds: tuple[float, float, float, float], margin: float = 3) -> bool:
    return (
        bounds[0] >= -margin
        and bounds[1] >= -margin
        and bounds[2] <= 100 + margin
        and bounds[3] <= 100 + margin
    )


def _diverse(
    hypothesis: LeafAlignment, selected: list[LeafAlignment], threshold: float = 5
) -> bool:
    x0, y0, x1, y1 = hypothesis.bounds
    center = np.asarray(((x0 + x1) / 2, (y0 + y1) / 2))
    for other in selected:
        ox0, oy0, ox1, oy1 = other.bounds
        other_center = np.asarray(((ox0 + ox1) / 2, (oy0 + oy1) / 2))
        scale_delta = abs(math.log(hypothesis.scale_x / other.scale_x)) + abs(
            math.log(hypothesis.scale_y / other.scale_y)
        )
        if np.linalg.norm(center - other_center) < threshold and scale_delta < 0.22:
            return False
    return True


def search_leaf_alignments(
    skeleton: np.ndarray,
    candidate: CandidateGraph,
    component_path: tuple[int, ...],
    *,
    count: int = 8,
    allow_extreme_scales: bool = False,
) -> tuple[LeafAlignment, ...]:
    """Find leaf-local affine placements without reading annotation truth.

    Candidate absolute layout is only the search origin. The score is a one-way
    distance from the transformed candidate stroke grammar to the PDF skeleton,
    so unrelated glyph ink is treated as an outlier rather than a target.
    """
    points = _leaf_points(candidate, component_path)
    point_minimum = points.min(axis=0)
    point_maximum = points.max(axis=0)
    span = np.maximum(point_maximum - point_minimum, 1e-6)
    normalized = (points - point_minimum) / span
    aligned = _align_points(
        tuple(map(tuple, points)), _candidate_bounds(candidate), _skeleton_bounds(skeleton)
    )
    minimum = aligned.min(axis=0)
    maximum = aligned.max(axis=0)
    base = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    distances, nearest = distance_transform_edt(~skeleton, return_indices=True)
    scales = (
        (0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5)
        if allow_extreme_scales
        else (0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5)
    )
    offsets = tuple(float(value) for value in range(-30, 31, 5))
    hypotheses = []
    for scale_x in scales:
        for scale_y in scales:
            scale_penalty = 0.04 * (
                abs(math.log(scale_x)) + abs(math.log(scale_y))
            ) + 0.5 * abs(math.log(scale_x / scale_y))
            for offset_x in offsets:
                for offset_y in offsets:
                    bounds = _hypothesis_bounds(
                        base, scale_x, scale_y, offset_x, offset_y
                    )
                    if not _inside_canvas(bounds):
                        continue
                    score = _score_points(normalized, bounds, distances) + scale_penalty
                    transformed = _transformed_points(normalized, bounds)
                    height, width = skeleton.shape
                    xs = np.clip(
                        np.rint(transformed[:, 0] * width / 100 - 0.5),
                        0,
                        width - 1,
                    ).astype(int)
                    ys = np.clip(
                        np.rint(transformed[:, 1] * height / 100 - 0.5),
                        0,
                        height - 1,
                    ).astype(int)
                    claimed = tuple(
                        sorted(
                            {
                                (int(nearest[0, y, x]), int(nearest[1, y, x]))
                                for y, x in zip(ys, xs)
                            }
                        )
                    )
                    hypotheses.append(
                        LeafAlignment(
                            component_path,
                            bounds,
                            score,
                            scale_x,
                            scale_y,
                            offset_x,
                            offset_y,
                            claimed,
                        )
                    )
    hypotheses.sort(key=lambda item: (item.score, item.bounds))
    # Preserve spatial/scale modes before global pruning. This prevents a simple
    # leaf from filling the whole pool with tiny matches to one unrelated area.
    bucketed = {}
    for hypothesis in hypotheses:
        x0, y0, x1, y1 = hypothesis.bounds
        signature = (
            round((x0 + x1) / 20),
            round((y0 + y1) / 20),
            hypothesis.scale_x,
            hypothesis.scale_y,
        )
        bucketed.setdefault(signature, hypothesis)
    ordered = sorted(bucketed.values(), key=lambda item: (item.score, item.bounds))
    selected: list[LeafAlignment] = []
    for hypothesis in ordered:
        if _diverse(hypothesis, selected):
            selected.append(hypothesis)
            if len(selected) == count:
                break
    return tuple(selected)


def _alignment_ids_specs(candidate: CandidateGraph, leaf_paths):
    output = []

    def visit(node):
        pairs = (
            ((node.operator, node.children[0], node.children[1]),)
            if len(node.children) == 2
            and node.operator in {"⿰", "⿱", "⿸", "⿹"}
            else tuple(
                (
                    "⿰" if node.operator == "⿲" else "⿱",
                    node.children[index],
                    node.children[index + 1],
                )
                for index in range(2)
            )
            if len(node.children) == 3 and node.operator in {"⿲", "⿳"}
            else ()
        )
        for operator, first_child, second_child in pairs:
            output.append(
                (
                    operator,
                    frozenset(
                        path
                        for path in leaf_paths
                        if path[: len(first_child.path)] == first_child.path
                    ),
                    frozenset(
                        path
                        for path in leaf_paths
                        if path[: len(second_child.path)] == second_child.path
                    ),
                )
            )
        for child in node.children:
            visit(child)

    visit(candidate.root)
    return tuple(output)


def _alignment_ids_cost(specs, selected: tuple[LeafAlignment, ...]) -> float:
    by_path = {item.component_path: item for item in selected}
    cost = 0.0
    for operator, first_paths, second_paths in specs:
        first = [by_path[path] for path in first_paths if path in by_path]
        second = [by_path[path] for path in second_paths if path in by_path]
        # A recursive IDS relation describes two complete subtrees. Scoring a
        # partially built subtree during beam search substitutes one early leaf
        # for its whole parent and can irreversibly prune the correct layout.
        if len(first) != len(first_paths) or len(second) != len(second_paths):
            continue

        def center(items):
            return (
                sum((item.bounds[0] + item.bounds[2]) / 2 for item in items)
                / len(items),
                sum((item.bounds[1] + item.bounds[3]) / 2 for item in items)
                / len(items),
            )

        first_x, first_y = center(first)
        second_x, second_y = center(second)
        first_bounds = (
            min(item.bounds[0] for item in first),
            min(item.bounds[1] for item in first),
            max(item.bounds[2] for item in first),
            max(item.bounds[3] for item in first),
        )
        second_bounds = (
            min(item.bounds[0] for item in second),
            min(item.bounds[1] for item in second),
            max(item.bounds[2] for item in second),
            max(item.bounds[3] for item in second),
        )

        def orthogonal_coherence(
            first_interval: tuple[float, float],
            second_interval: tuple[float, float],
        ) -> float:
            """Penalize siblings that obey IDS order but miss each other sideways.

            A pure centre-order check accepts a top child at the far left and a
            bottom child at the far right as ⿱.  The orthogonal axis is weaker
            evidence than the IDS axis, but the two subtrees should normally
            share a visual column/row.  Normalizing by their union keeps this
            usable for optically asymmetric components.
            """
            first_start, first_end = first_interval
            second_start, second_end = second_interval
            union_span = max(first_end, second_end) - min(
                first_start, second_start
            )
            if union_span <= 1e-6:
                return 0.0
            first_center = (first_start + first_end) / 2
            second_center = (second_start + second_end) / 2
            gap = max(
                0.0,
                max(first_start, second_start) - min(first_end, second_end),
            )
            return 0.5 * abs(first_center - second_center) / union_span + gap / union_span

        def ordered_axis_overlap(
            first_interval: tuple[float, float],
            second_interval: tuple[float, float],
        ) -> float:
            """Softly reject IDS siblings that occupy nearly the same band."""
            first_start, first_end = first_interval
            second_start, second_end = second_interval
            overlap = max(
                0.0,
                min(first_end, second_end) - max(first_start, second_start),
            )
            smaller_span = min(
                first_end - first_start,
                second_end - second_start,
            )
            if smaller_span <= 1e-6:
                return 0.0
            return 0.5 * overlap / smaller_span

        if operator == "⿰":
            cost += max(0.0, first_x - second_x + 1.0) / 100
            cost += ordered_axis_overlap(
                (first_bounds[0], first_bounds[2]),
                (second_bounds[0], second_bounds[2]),
            )
            cost += orthogonal_coherence(
                (first_bounds[1], first_bounds[3]),
                (second_bounds[1], second_bounds[3]),
            )
        elif operator == "⿱":
            cost += max(0.0, first_y - second_y + 1.0) / 100
            cost += ordered_axis_overlap(
                (first_bounds[1], first_bounds[3]),
                (second_bounds[1], second_bounds[3]),
            )
            cost += orthogonal_coherence(
                (first_bounds[0], first_bounds[2]),
                (second_bounds[0], second_bounds[2]),
            )
        elif operator == "⿸":
            cost += max(0.0, first_x - second_x + 1.0) / 100
            cost += max(0.0, first_y - second_y + 1.0) / 100
            # The first child is the enclosing top-left frame. Its aggregate
            # extent should contain the inner child's four extrema, allowing a
            # small optical overhang at the open bottom/right edges.
            tolerance = 3.0
            cost += max(0.0, first_bounds[0] - second_bounds[0]) / 100
            cost += max(0.0, first_bounds[1] - second_bounds[1]) / 100
            cost += max(
                0.0, second_bounds[2] - first_bounds[2] - tolerance
            ) / 100
            cost += max(
                0.0, second_bounds[3] - first_bounds[3] - tolerance
            ) / 100
        elif operator == "⿹":
            # Upper-right enclosure: the first child is the frame. The inner
            # child must lie below its top and left of its right edge. Bottom
            # and left can overhang slightly because the frame is open there.
            tolerance = 3.0
            cost += max(0.0, first_y - second_y + 1.0) / 100
            cost += max(0.0, second_x - first_x + 1.0) / 100
            cost += max(0.0, first_bounds[1] - second_bounds[1]) / 100
            cost += max(
                0.0, second_bounds[2] - first_bounds[2] - tolerance
            ) / 100
    return cost


def alignment_ids_cost(
    candidate: CandidateGraph, selected: tuple[LeafAlignment, ...]
) -> float:
    """Return recursive IDS cost for a complete or partial leaf placement."""
    paths = tuple(item.component_path for item in selected)
    return _alignment_ids_cost(_alignment_ids_specs(candidate, paths), selected)


def _global_envelope_cost(
    selected: tuple[LeafAlignment, ...],
    target: tuple[float, float, float, float],
) -> float:
    """Measure how completely all leaves explain the glyph's outer envelope.

    Every terminal IDS leaf belongs to the same glyph, so their union should
    reach the four stable extrema of the PDF skeleton.  This rejects locally
    attractive subset matches without assigning every pixel to a leaf yet.
    """
    if not selected:
        return 0.0
    union = (
        min(item.bounds[0] for item in selected),
        min(item.bounds[1] for item in selected),
        max(item.bounds[2] for item in selected),
        max(item.bounds[3] for item in selected),
    )
    target_width = max(target[2] - target[0], 1e-6)
    target_height = max(target[3] - target[1], 1e-6)
    return (
        abs(union[0] - target[0]) / target_width
        + abs(union[2] - target[2]) / target_width
        + abs(union[1] - target[1]) / target_height
        + abs(union[3] - target[3]) / target_height
    )


def select_joint_leaf_alignments(
    alignments_by_path: dict[tuple[int, ...], tuple[LeafAlignment, ...]],
    candidate: CandidateGraph,
    *,
    skeleton: np.ndarray | None = None,
    beam_width: int = 400,
    repeated_claim_penalty: float = 0.8,
    new_claim_reward: float = 0.12,
    ids_structure_penalty: float = 40.0,
    global_envelope_penalty: float = 15.0,
    global_envelope_trigger: float = 0.30,
    free_contact_claims: int = 4,
) -> JointLeafAlignmentSolution | None:
    """Select leaf placements jointly under IDS order and exclusive ownership.

    The ordinary local/IDS solve remains authoritative when its leaf union
    already covers the glyph reasonably well.  A second envelope-aware solve
    is only activated for gross subset matches, preserving optical whitespace
    in otherwise valid recursive layouts.
    """
    if not alignments_by_path or any(not items for items in alignments_by_path.values()):
        return None
    paths = tuple(sorted(alignments_by_path))
    specs = _alignment_ids_specs(candidate, paths)
    target_bounds = _skeleton_bounds(skeleton) if skeleton is not None else None

    def solve(envelope_penalty: float) -> JointLeafAlignmentSolution:
        # local score, selected, occupied claims, repeated count, IDS cost
        states = [(0.0, (), frozenset(), 0, 0.0)]
        for path in paths:
            expanded = []
            for score, selected, occupied, repeated, _structure in states:
                for alignment in alignments_by_path[path]:
                    claims = frozenset(alignment.claimed_pixels)
                    overlap = len(claims & occupied)
                    charged_overlap = max(0, overlap - free_contact_claims)
                    new_claims = len(claims - occupied)
                    proposed_selected = (*selected, alignment)
                    structure = _alignment_ids_cost(specs, proposed_selected)
                    envelope = (
                        _global_envelope_cost(proposed_selected, target_bounds)
                        if target_bounds is not None
                        and len(proposed_selected) == len(paths)
                        else 0.0
                    )
                    proposed_score = (
                        score
                        + alignment.score
                        + repeated_claim_penalty * charged_overlap
                        - new_claim_reward * new_claims
                        + envelope_penalty * envelope
                    )
                    expanded.append(
                        (
                            proposed_score,
                            proposed_selected,
                            occupied | claims,
                            repeated + charged_overlap,
                            structure,
                        )
                    )
            expanded.sort(
                key=lambda item: (
                    item[0] + ids_structure_penalty * item[4],
                    item[3],
                    tuple(alignment.bounds for alignment in item[1]),
                )
            )
            states = expanded[:beam_width]
        score, selected, _occupied, repeated, structure = min(
            states,
            key=lambda item: (item[0] + ids_structure_penalty * item[4], item[3]),
        )
        return JointLeafAlignmentSolution(
            selected,
            score + ids_structure_penalty * structure,
            repeated,
            structure,
        )

    ordinary = solve(0.0)
    if target_bounds is None or global_envelope_penalty <= 0:
        return ordinary
    if _global_envelope_cost(ordinary.alignments, target_bounds) <= global_envelope_trigger:
        return ordinary
    return solve(global_envelope_penalty)
