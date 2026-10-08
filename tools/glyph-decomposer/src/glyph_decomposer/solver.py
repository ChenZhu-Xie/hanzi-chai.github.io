"""Bounded exact selection of an IDS-constrained root partition."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import box

from .domain import ComponentProgram, PartitionEvidence
from .geometry import clip_axis


@dataclass(frozen=True)
class RootPartition:
    evidence: PartitionEvidence
    children: tuple[object, object]
    stroke_regions: tuple[object, ...] = ()
    stroke_labels: tuple[str, ...] = ()


def _axis_for(operator: str | None) -> str:
    if operator == "⿰":
        return "x"
    if operator == "⿱":
        return "y"
    raise ValueError(f"initial root solver supports ⿰ and ⿱, not {operator!r}")


def solve_root_partition(
    geometry,
    program: ComponentProgram,
    *,
    minimum: float = 18.0,
    maximum: float = 82.0,
    step: float = 0.5,
    band_width: float = 0.75,
) -> RootPartition:
    if program.kind != "compound" or len(program.children) != 2:
        raise ValueError("initial root solver requires a binary compound")
    axis = _axis_for(program.operator)
    total_area = geometry.area
    if total_area <= 0:
        raise ValueError("source geometry has no area")

    child_strokes = [max(1, child.stroke_count()) for child in program.children]
    expected_ratio = child_strokes[0] / sum(child_strokes)
    options = []
    count = round((maximum - minimum) / step)
    for index in range(count + 1):
        cut = minimum + index * step
        left, right = clip_axis(geometry, axis, cut)
        if left.area <= 0 or right.area <= 0:
            continue
        if axis == "x":
            band = box(cut - band_width, -1_000, cut + band_width, 1_000)
        else:
            band = box(-1_000, cut - band_width, 1_000, cut + band_width)
        crossing_ratio = geometry.intersection(band).area / total_area
        observed_ratio = left.area / total_area
        balance_error = abs(observed_ratio - expected_ratio)
        edge_penalty = abs(cut - 50.0) / 50.0
        # The gap is primary. Stroke-count balance only prevents extreme cuts;
        # candidate coordinates do not enter the objective.
        score = crossing_ratio * 100.0 + balance_error * 3.0 + edge_penalty * 0.05
        options.append((cut, left, right, crossing_ratio, balance_error, score))
    if not options:
        raise ValueError("no feasible root cut candidates")

    # The initial problem is a bounded one-dimensional search.  Sorting every
    # feasible cut is both exact and substantially more reliable than crossing
    # a native MILP/CP boundary.  The tuple supplies deterministic tie-breaks.
    cut, first, second, crossing_ratio, balance_error, score = min(
        options,
        key=lambda option: (
            round(option[-1], 12),
            round(option[3], 12),
            round(option[4], 12),
            abs(option[0] - 50.0),
            option[0],
        ),
    )
    reconstruction_error = (
        first.union(second).symmetric_difference(geometry).area / total_area
    )
    return RootPartition(
        evidence=PartitionEvidence(
            axis=axis,
            cut=cut,
            score=score,
            crossingRatio=crossing_ratio,
            balanceError=balance_error,
            reconstructionError=reconstruction_error,
            solverStatus="EXACT_ENUMERATION",
        ),
        children=(first, second),
    )


def solve_surround_partition(
    geometry,
    program: ComponentProgram,
    *,
    grid_steps: int = 36,
    band_width: float = 0.75,
) -> RootPartition:
    if program.kind != "compound" or len(program.children) != 2:
        raise ValueError("surround solver requires a binary compound")
    if program.operator != "⿸":
        raise ValueError(
            f"initial surround solver supports ⿸, not {program.operator!r}"
        )
    total_area = geometry.area
    if total_area <= 0:
        raise ValueError("source geometry has no area")
    min_x, min_y, max_x, max_y = geometry.bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    child_strokes = [max(1, child.stroke_count()) for child in program.children]
    expected_ratio = child_strokes[0] / sum(child_strokes)
    options = []
    for x_index in range(1, grid_steps):
        x_cut = min_x + span_x * x_index / grid_steps
        for y_index in range(1, grid_steps):
            y_cut = min_y + span_y * y_index / grid_steps
            lower_right = box(x_cut, y_cut, max_x + 1, max_y + 1)
            second = geometry.intersection(lower_right)
            first = geometry.difference(second)
            if first.area <= 0 or second.area <= 0:
                continue
            # A real child program cannot collapse into a serif-sized speck.
            # This is deliberately based on program existence, not annotation.
            if min(first.area, second.area) / total_area < 0.10:
                continue
            vertical_boundary = box(
                x_cut - band_width,
                y_cut,
                x_cut + band_width,
                max_y + 1,
            )
            horizontal_boundary = box(
                x_cut,
                y_cut - band_width,
                max_x + 1,
                y_cut + band_width,
            )
            boundary = vertical_boundary.union(horizontal_boundary)
            crossing_ratio = geometry.intersection(boundary).area / total_area
            observed_ratio = first.area / total_area
            balance_error = abs(observed_ratio - expected_ratio)
            corner_penalty = (
                abs(x_cut - (min_x + span_x * 0.45)) / span_x
                + abs(y_cut - (min_y + span_y * 0.45)) / span_y
            )
            score = crossing_ratio * 25.0 + balance_error * 6.0 + corner_penalty * 0.03
            options.append(
                (
                    x_cut,
                    y_cut,
                    first,
                    second,
                    crossing_ratio,
                    balance_error,
                    score,
                )
            )
    if not options:
        raise ValueError("no feasible upper-left surround candidates")
    x_cut, y_cut, first, second, crossing_ratio, balance_error, score = min(
        options,
        key=lambda option: (
            round(option[-1], 12),
            round(option[4], 12),
            round(option[5], 12),
            option[1],
            option[0],
        ),
    )
    if crossing_ratio > 0.01:
        raise ValueError(
            f"best ⿸ boundary crosses {crossing_ratio:.3%} of source ink; "
            "stroke-level assignment is required"
        )
    reconstruction_error = (
        first.union(second).symmetric_difference(geometry).area / total_area
    )
    return RootPartition(
        evidence=PartitionEvidence(
            axis="xy",
            cut=y_cut,
            secondaryCut=x_cut,
            score=score,
            crossingRatio=crossing_ratio,
            balanceError=balance_error,
            reconstructionError=reconstruction_error,
            solverStatus="EXACT_ENUMERATION",
        ),
        children=(first, second),
    )
