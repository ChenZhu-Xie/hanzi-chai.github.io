"""Deterministic route alternatives scored by a fixed stroke grammar."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import cache
from itertools import product

import numpy as np

from .candidate_graph import CandidateComponent, CandidateGraph, CandidateStroke
from .leaf_alignment import LeafAlignment
from .primitive_fit import FittedStroke, fit_stroke_primitives, symmetric_chamfer
from .route_cover import (
    Pixel,
    _align_points,
    _candidate_bounds,
    _components,
    _distance_to_polyline,
    _pixel_neighbours,
    _rank_components,
    _skeleton_bounds,
)
from .stroke_grammar import StrokeGrammar, canonical_stroke_grammar


@dataclass(frozen=True)
class PrimitiveRoute:
    stroke_index: int
    pixels: tuple[Pixel, ...]
    fit: FittedStroke
    score: float
    endpoint_cost: float
    guide_cost: float
    length_cost: float
    direction_cost: float
    region_cost: float
    placement_confidence: float
    alternative_count: int


@dataclass(frozen=True)
class JointRouteSolution:
    routes: tuple[PrimitiveRoute, ...]
    score: float
    covered_pixel_count: int
    repeated_pixel_count: int
    residual_pixel_count: int
    ids_structure_cost: float
    stroke_relation_cost: float


def _xy(pixel: Pixel, scale: float) -> tuple[float, float]:
    return (pixel[1] + 0.5) * scale, (pixel[0] + 0.5) * scale


def _endpoint_candidates(
    component: frozenset[Pixel],
    target: np.ndarray,
    scale: float,
    *,
    count: int = 7,
    separation: float = 3.0,
    diverse: bool = False,
) -> tuple[Pixel, ...]:
    offsets = tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
    )
    critical = [
        pixel
        for pixel in component
        if sum((pixel[0] + dy, pixel[1] + dx) in component for dy, dx in offsets) != 2
    ]
    ordered = sorted(
        critical or component,
        key=lambda pixel: (
            float(np.linalg.norm(np.asarray(_xy(pixel, scale)) - target)),
            pixel,
        ),
    )
    nearest = min(
        component,
        key=lambda pixel: (
            float(np.linalg.norm(np.asarray(_xy(pixel, scale)) - target)),
            pixel,
        ),
    )
    if not diverse:
        selected = []
        for pixel in ordered:
            point = np.asarray(_xy(pixel, scale))
            if any(
                np.linalg.norm(point - np.asarray(_xy(other, scale))) < separation
                for other in selected
            ):
                continue
            selected.append(pixel)
            if len(selected) == count:
                break
        if all(
            np.linalg.norm(
                np.asarray(_xy(nearest, scale)) - np.asarray(_xy(other, scale))
            )
            >= separation
            for other in selected
        ):
            selected.append(nearest)
        return tuple(selected)
    separated = []
    for pixel in (nearest, *ordered):
        point = np.asarray(_xy(pixel, scale))
        if any(
            np.linalg.norm(point - np.asarray(_xy(other, scale))) < separation
            for other in separated
        ):
            continue
        separated.append(pixel)
    selected = []
    local_count = max(1, count - 2)
    for pixel in separated:
        selected.append(pixel)
        if len(selected) == local_count:
            break
    # Candidate geometry is a prior, not truth. Reserve two slots for spatial
    # modes that are farther from its endpoint but still in the near frontier.
    # This keeps route count bounded while avoiding pure nearest-neighbour
    # pruning of a real source-font branch.
    frontier = separated[: max(count * 2, count)]
    while len(selected) < count:
        eligible = [
            pixel
            for pixel in frontier
            if pixel not in selected
            and all(
                np.linalg.norm(
                    np.asarray(_xy(pixel, scale)) - np.asarray(_xy(other, scale))
                )
                >= separation
                for other in selected
            )
        ]
        if not eligible:
            break
        pixel = min(
            eligible,
            key=lambda item: (
                -min(
                    np.linalg.norm(
                        np.asarray(_xy(item, scale)) - np.asarray(_xy(other, scale))
                    )
                    for other in selected
                ),
                float(np.linalg.norm(np.asarray(_xy(item, scale)) - target)),
                item,
            ),
        )
        selected.append(pixel)
    return tuple(selected)


def _angle_cost(first: np.ndarray, second: np.ndarray) -> float:
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 2.0
    cosine = float(
        np.clip(np.dot(first, second) / (first_length * second_length), -1, 1)
    )
    return 1.0 - cosine


def _direction_cost(route: np.ndarray, grammar: StrokeGrammar) -> float:
    if len(route) < 3:
        return 2.0
    window = min(5, len(route) - 1)
    start_vector = route[window] - route[0]
    end_vector = route[-1] - route[-1 - window]
    return _angle_cost(start_vector, np.asarray(grammar.direction_vectors[0])) + (
        _angle_cost(end_vector, np.asarray(grammar.direction_vectors[-1]))
        if len(grammar.commands) > 1
        else 0.0
    )


def _single_curve_direction_compatible(
    fitted_path: tuple[tuple[float, float], ...],
    grammar: StrokeGrammar,
    candidate_path: tuple[tuple[float, float], ...] | None = None,
    *,
    minimum_cosine: float = 0.80,
) -> bool:
    """Reject a one-piece curve drawn in the wrong semantic direction.

    A 点/撇/捺 may curve substantially, but its directed start-to-end chord
    cannot reverse the canonical stroke class.  Treating this as a soft local
    score lets a nearby skeleton branch beat the actual directed stroke before
    topology and coverage are considered.
    """
    if grammar.commands != ("c",) or len(fitted_path) < 2:
        return True
    chord = np.asarray(fitted_path[-1]) - np.asarray(fitted_path[0])
    expected = (
        np.asarray(candidate_path[-1]) - np.asarray(candidate_path[0])
        if candidate_path is not None and len(candidate_path) >= 2
        else np.asarray(grammar.direction_vectors[0], dtype=float)
    )
    denominator = float(np.linalg.norm(chord) * np.linalg.norm(expected))
    if denominator <= 1e-9:
        return False
    cosine = float(np.dot(chord, expected) / denominator)
    return cosine >= minimum_cosine


def _single_curve_turn_compatible(
    fit: FittedStroke,
    grammar: StrokeGrammar,
    *,
    sample_count: int = 33,
) -> bool:
    """Reject a materially inflected 捺/平捺 centreline.

    Brush outline details can widen or taper a stroke, but the directed main
    path of a 捺 keeps turning to one side. Tiny fitted sign changes are
    tolerated because a least-squares cubic over a pixel skeleton can put its
    controls just across the chord. Controls well onto opposite sides of the
    chord, or a very long control polygon, identify a real S-shaped detour
    assembled from unrelated skeleton branches.
    """
    if grammar.feature not in {"捺", "平捺"} or grammar.commands != ("c",):
        return True
    if len(fit.primitives) != 1 or fit.primitives[0].kind != "cubic":
        return False
    controls = np.asarray(fit.primitives[0].controls, dtype=float)
    if controls.shape != (4, 2):
        return False
    span = controls.max(axis=0) - controls.min(axis=0)
    tolerance = max(1e-6, 1e-3 * float(np.dot(span, span)))
    signs = set()
    for t in np.linspace(0.0, 1.0, sample_count):
        u = 1.0 - t
        first = 3 * (
            u * u * (controls[1] - controls[0])
            + 2 * u * t * (controls[2] - controls[1])
            + t * t * (controls[3] - controls[2])
        )
        second = 6 * (
            u * (controls[2] - 2 * controls[1] + controls[0])
            + t * (controls[3] - 2 * controls[2] + controls[1])
        )
        signed_curvature = float(first[0] * second[1] - first[1] * second[0])
        if signed_curvature > tolerance:
            signs.add(1)
        elif signed_curvature < -tolerance:
            signs.add(-1)
    if len(signs) <= 1:
        return True
    chord = controls[3] - controls[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-9:
        return False
    signed_control_distances = tuple(
        float(
            chord[0] * (control - controls[0])[1]
            - chord[1] * (control - controls[0])[0]
        )
        / chord_length
        for control in controls[1:3]
    )
    opposite_sides = signed_control_distances[0] * signed_control_distances[1] < 0
    material_crossing = (
        opposite_sides
        and min(abs(value) for value in signed_control_distances) > 0.1 * chord_length
    )
    control_polygon_length = float(
        np.linalg.norm(np.diff(controls, axis=0), axis=1).sum()
    )
    material_detour = control_polygon_length > 1.5 * chord_length
    return not (material_crossing or material_detour)


def _has_interior_same_leaf_contact(
    candidate: CandidateGraph, stroke: CandidateStroke
) -> bool:
    """Whether another stroke meets the interior of this directed stroke."""
    for relation in candidate.relations:
        if not relation.exact_intersection or not relation.same_leaf_occurrence:
            continue
        if relation.first_stroke == stroke.index:
            position = relation.first_position
        elif relation.second_stroke == stroke.index:
            position = relation.second_position
        else:
            continue
        if 0.15 < position < 0.85:
            return True
    return False


def _guided_shortest_paths(
    component: frozenset[Pixel],
    shape: tuple[int, int],
    start: Pixel,
    goals: tuple[Pixel, ...],
    guide_distance: dict[Pixel, float],
    *,
    cache: dict | None = None,
    cache_key: object | None = None,
) -> dict[Pixel, tuple[Pixel, ...]]:
    state = cache.get(cache_key) if cache is not None else None
    if state is None:
        distance = {start: 0.0}
        previous: dict[Pixel, Pixel] = {}
        queue = [(0.0, start)]
        settled: set[Pixel] = set()
        state = (distance, previous, queue, settled)
        if cache is not None:
            cache[cache_key] = state
    else:
        distance, previous, queue, settled = state
    remaining = set(goals) - settled
    while queue and remaining:
        cost, point = heapq.heappop(queue)
        if cost != distance.get(point):
            continue
        settled.add(point)
        remaining.discard(point)
        for neighbour in _pixel_neighbours(point, shape):
            if neighbour not in component:
                continue
            step = math.hypot(neighbour[0] - point[0], neighbour[1] - point[1])
            proximity = (guide_distance[point] + guide_distance[neighbour]) / 2
            proposed = cost + step * (1 + min(proximity / 3.0, 5.0))
            if proposed < distance.get(neighbour, math.inf):
                distance[neighbour] = proposed
                previous[neighbour] = point
                heapq.heappush(queue, (proposed, neighbour))
    output = {}
    for goal in goals:
        if goal == start:
            output[goal] = (start,)
            continue
        if goal not in previous:
            continue
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        output[goal] = tuple(reversed(path))
    return output


def _route_score(
    route: tuple[Pixel, ...],
    fit: FittedStroke,
    aligned: np.ndarray,
    grammar: StrokeGrammar,
    scale: float,
    region_bounds: tuple[float, float, float, float],
    placement_confidence: float,
) -> tuple[float, float, float, float, float, float]:
    route_xy = np.asarray([_xy(pixel, scale) for pixel in route])
    endpoint_cost = float(
        np.linalg.norm(route_xy[0] - aligned[0])
        + np.linalg.norm(route_xy[-1] - aligned[-1])
    )
    probes = route_xy[
        np.linspace(0, len(route_xy) - 1, min(25, len(route_xy))).astype(int)
    ]
    guide_cost = float(
        np.mean([_distance_to_polyline(point, aligned) for point in probes])
    )
    route_length = float(np.linalg.norm(np.diff(route_xy, axis=0), axis=1).sum())
    candidate_length = float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum())
    length_cost = abs(math.log(max(route_length, 1e-6) / max(candidate_length, 1e-6)))
    direction_cost = _direction_cost(route_xy, grammar)
    x0, y0, x1, y1 = region_bounds
    outside_x = np.maximum(x0 - probes[:, 0], 0) + np.maximum(probes[:, 0] - x1, 0)
    outside_y = np.maximum(y0 - probes[:, 1], 0) + np.maximum(probes[:, 1] - y1, 0)
    region_cost = float(np.mean(np.hypot(outside_x, outside_y)))
    endpoint_weight = 0.05 + 0.23 * placement_confidence
    guide_weight = 0.03 + 0.13 * placement_confidence
    length_weight = 0.5 + 1.5 * placement_confidence
    region_weight = 2.5 * placement_confidence
    score = (
        1.0 * fit.rmse
        + 0.2 * fit.maximum_error
        # Candidate placement proposes routes but is not ground truth. Keep its
        # endpoints, guide, and implied length as weak priors only; recursive
        # IDS layouts can substantially compress or move a real source leaf.
        + endpoint_weight * endpoint_cost
        + guide_weight * guide_cost
        + length_weight * length_cost
        + 1.5 * direction_cost
        + region_weight * region_cost
    )
    return score, endpoint_cost, guide_cost, length_cost, direction_cost, region_cost


def _component_bounds(candidate: CandidateGraph, path: tuple[int, ...]):
    node = candidate.root
    for index in path:
        node = node.children[index]
    return node.bounds


def _leaf_stroke_bounds(
    candidate: CandidateGraph, path: tuple[int, ...]
) -> tuple[float, float, float, float]:
    points = np.asarray(
        [
            point
            for stroke in candidate.strokes
            if stroke.component_path == path
            for point in stroke.points
        ],
        dtype=float,
    )
    if not len(points):
        raise ValueError(f"candidate has no leaf strokes at {path}")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _align_bounds(
    bounds: tuple[float, float, float, float],
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    *,
    margin: float = 3.0,
) -> tuple[float, float, float, float]:
    corners = _align_points(
        ((bounds[0], bounds[1]), (bounds[2], bounds[3])), source, target
    )
    return (
        float(corners[0, 0] - margin),
        float(corners[0, 1] - margin),
        float(corners[1, 0] + margin),
        float(corners[1, 1] + margin),
    )


def enumerate_primitive_routes(
    skeleton: np.ndarray,
    candidate: CandidateGraph,
    stroke: CandidateStroke,
    *,
    endpoint_count: int = 7,
    leaf_alignment: LeafAlignment | None = None,
    shortest_path_cache: dict | None = None,
) -> tuple[PrimitiveRoute, ...]:
    """Enumerate a small, deterministic route set without annotation truth."""
    scale = 100.0 / skeleton.shape[0]
    source_bounds = _candidate_bounds(candidate)
    target_bounds = _skeleton_bounds(skeleton)
    if leaf_alignment is None:
        aligned = _align_points(stroke.points, source_bounds, target_bounds)
        region_bounds = _align_bounds(
            _component_bounds(candidate, stroke.component_path),
            source_bounds,
            target_bounds,
        )
    else:
        if leaf_alignment.component_path != stroke.component_path:
            raise ValueError("leaf alignment path does not own the candidate stroke")
        aligned = _align_points(
            stroke.points,
            _leaf_stroke_bounds(candidate, stroke.component_path),
            leaf_alignment.bounds,
        )
        region_bounds = leaf_alignment.bounds
    ranked_components = _rank_components(aligned, _components(skeleton), scale)
    best_component_score = ranked_components[0][0]
    plausible_components = tuple(
        component
        for component_score, component in ranked_components[:2]
        if component_score <= best_component_score + 4.0
    )
    grammar = canonical_stroke_grammar(stroke.feature, stroke.commands)
    leaf_stroke_count = sum(
        item.component_path == stroke.component_path for item in candidate.strokes
    )
    diverse_endpoints = leaf_stroke_count >= 5 and grammar.commands == ("c",)
    interior_contact = _has_interior_same_leaf_contact(candidate, stroke)
    effective_endpoint_count = (
        max(endpoint_count, 11)
        if leaf_stroke_count >= 5 and stroke.feature == "点" and interior_contact
        else endpoint_count
    )
    stroke_evidence = min(1.0, max(0.0, (leaf_stroke_count - 2) / 3))
    scale_evidence = (
        math.exp(
            -abs(math.log(leaf_alignment.scale_x))
            - abs(math.log(leaf_alignment.scale_y))
        )
        if leaf_alignment is not None and leaf_stroke_count >= 3
        else 0.0
    )
    placement_confidence = (
        min(1.0, stroke_evidence + 0.5 * scale_evidence) * scale_evidence
        if leaf_stroke_count >= 3
        else stroke_evidence
    )
    alternatives = []
    seen: set[tuple[Pixel, ...]] = set()
    for component in plausible_components:
        starts = _endpoint_candidates(
            component,
            aligned[0],
            scale,
            count=effective_endpoint_count,
            diverse=diverse_endpoints,
        )
        ends = _endpoint_candidates(
            component,
            aligned[-1],
            scale,
            count=effective_endpoint_count,
            diverse=diverse_endpoints,
        )
        guide_distance = {
            point: _distance_to_polyline(np.asarray(_xy(point, scale)), aligned)
            for point in component
        }
        paths = {
            (start, end): route
            for start in starts
            for end, route in _guided_shortest_paths(
                component,
                skeleton.shape,
                start,
                ends,
                guide_distance,
                cache=shortest_path_cache,
                cache_key=(
                    component,
                    start,
                    tuple(float(value) for value in aligned.ravel()),
                ),
            ).items()
        }
        for start, end in product(starts, ends):
            if start == end:
                continue
            route = paths.get((start, end), ())
            if len(route) < len(grammar.commands) + 1 or route in seen:
                continue
            seen.add(route)
            route_xy = [_xy(pixel, scale) for pixel in route]
            try:
                fit = fit_stroke_primitives(
                    route_xy,
                    grammar.commands,
                    stroke.points,
                    expected_vectors=grammar.direction_vectors,
                    segment_ratios=grammar.segment_ratios,
                )
            except ValueError:
                continue
            if not _single_curve_direction_compatible(
                fit.fitted_path,
                grammar,
                stroke.points,
                minimum_cosine=0.85 if interior_contact else 0.80,
            ):
                continue
            if interior_contact and not _single_curve_direction_compatible(
                fit.fitted_path, grammar, minimum_cosine=0.85
            ):
                continue
            if not _single_curve_turn_compatible(fit, grammar):
                continue
            score, endpoint, guide, length, direction, region = _route_score(
                route,
                fit,
                aligned,
                grammar,
                scale,
                region_bounds,
                stroke_evidence,
            )
            alternatives.append(
                PrimitiveRoute(
                    stroke.index,
                    route,
                    fit,
                    score,
                    endpoint,
                    guide,
                    length,
                    direction,
                    region,
                    placement_confidence,
                    0,
                )
            )
    alternatives.sort(key=lambda item: (item.score, item.pixels))
    count = len(alternatives)
    return tuple(
        PrimitiveRoute(
            item.stroke_index,
            item.pixels,
            item.fit,
            item.score,
            item.endpoint_cost,
            item.guide_cost,
            item.length_cost,
            item.direction_cost,
            item.region_cost,
            item.placement_confidence,
            count,
        )
        for item in alternatives
    )


def best_primitive_route(
    skeleton: np.ndarray, candidate: CandidateGraph, stroke: CandidateStroke
) -> PrimitiveRoute | None:
    alternatives = enumerate_primitive_routes(skeleton, candidate, stroke)
    return alternatives[0] if alternatives else None


def route_truth_distance(route: PrimitiveRoute, truth_points) -> float:
    """Held-out evaluator; never call this while choosing a route."""
    return symmetric_chamfer(route.fit.fitted_path, truth_points)


@cache
def _route_moments(pixels: tuple[Pixel, ...]) -> tuple[int, float, float]:
    return (
        len(pixels),
        sum(pixel[1] for pixel in pixels),
        sum(pixel[0] for pixel in pixels),
    )


def _ids_structure_specs(
    root: CandidateComponent,
) -> tuple[tuple[str, frozenset[int], frozenset[int]], ...]:
    output = []

    def visit(node: CandidateComponent):
        pairs = (
            ((node.operator, node.children[0], node.children[1]),)
            if len(node.children) == 2 and node.operator in {"⿰", "⿱", "⿸", "⿹"}
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
                    frozenset(first_child.stroke_indices),
                    frozenset(second_child.stroke_indices),
                )
            )
        for child in node.children:
            visit(child)

    visit(root)
    return tuple(output)


def _selected_moments(
    indices: frozenset[int], routes: dict[int, PrimitiveRoute]
) -> tuple[int, float, float]:
    count = x_sum = y_sum = 0
    for index in indices & routes.keys():
        route_count, route_x, route_y = _route_moments(routes[index].pixels)
        count += route_count
        x_sum += route_x
        y_sum += route_y
    return count, x_sum, y_sum


def _ids_structure_cost(
    specs: tuple[tuple[str, frozenset[int], frozenset[int]], ...],
    selected: tuple[PrimitiveRoute, ...],
) -> float:
    """Score recursive IDS ordering without trusting candidate absolute geometry."""
    routes = {route.stroke_index: route for route in selected}
    all_moments = [_route_moments(route.pixels) for route in selected]
    total_count = sum(item[0] for item in all_moments)
    if not total_count:
        return 0.0
    all_x = [
        pixel[1] for route in selected for pixel in (route.pixels[0], route.pixels[-1])
    ]
    all_y = [
        pixel[0] for route in selected for pixel in (route.pixels[0], route.pixels[-1])
    ]
    width = max(max(all_x) - min(all_x), 1)
    height = max(max(all_y) - min(all_y), 1)
    cost = 0.0
    for operator, first_indices, second_indices in specs:
        first_count, first_x_sum, first_y_sum = _selected_moments(first_indices, routes)
        second_count, second_x_sum, second_y_sum = _selected_moments(
            second_indices, routes
        )
        if not first_count or not second_count:
            continue
        first_x = first_x_sum / first_count
        first_y = first_y_sum / first_count
        second_x = second_x_sum / second_count
        second_y = second_y_sum / second_count
        if operator == "⿰":
            cost += max(0.0, first_x - second_x + 1.0) / width
        elif operator == "⿱":
            cost += max(0.0, first_y - second_y + 1.0) / height
        elif operator == "⿸":
            cost += max(0.0, first_x - second_x + 1.0) / width
            cost += max(0.0, first_y - second_y + 1.0) / height
        elif operator == "⿹":
            cost += max(0.0, first_y - second_y + 1.0) / height
            cost += max(0.0, second_x - first_x + 1.0) / width
    return cost


def _new_stroke_relation_cost(
    candidate: CandidateGraph,
    previous: tuple[PrimitiveRoute, ...],
    current: PrimitiveRoute,
    cache: dict[tuple[tuple[Pixel, ...], tuple[Pixel, ...], float, float], float],
) -> float:
    previous_by_index = {route.stroke_index: route for route in previous}
    cost = 0.0
    for relation in candidate.relations:
        if not relation.exact_intersection or not relation.same_leaf_occurrence:
            continue
        if relation.second_stroke == current.stroke_index:
            other = previous_by_index.get(relation.first_stroke)
            expected_other = relation.first_position
            expected_current = relation.second_position
        elif relation.first_stroke == current.stroke_index:
            other = previous_by_index.get(relation.second_stroke)
            expected_other = relation.second_position
            expected_current = relation.first_position
        else:
            continue
        if other is None:
            continue
        key = (
            other.pixels,
            current.pixels,
            expected_other,
            expected_current,
        )
        if key in cache:
            cost += cache[key]
            continue
        other_points = np.asarray(other.pixels, dtype=float)
        current_points = np.asarray(current.pixels, dtype=float)
        distances = np.linalg.norm(
            other_points[:, None, :] - current_points[None, :, :], axis=2
        )
        other_index, current_index = np.unravel_index(
            int(np.argmin(distances)), distances.shape
        )
        separation = float(distances[other_index, current_index])
        other_position = other_index / max(len(other_points) - 1, 1)
        current_position = current_index / max(len(current_points) - 1, 1)
        pair_cost = separation / 3.0
        pair_cost += abs(other_position - expected_other)
        pair_cost += abs(current_position - expected_current)
        cache[key] = pair_cost
        cost += pair_cost
    return cost


def select_joint_routes(
    alternatives_by_stroke: tuple[tuple[PrimitiveRoute, ...], ...],
    skeleton_pixel_count: int,
    *,
    candidate: CandidateGraph | None = None,
    beam_width: int = 400,
    free_contact_pixels: int = 5,
    repeated_pixel_penalty: float = 0.32,
    new_pixel_reward: float = 0.03,
    residual_pixel_penalty: float = 0.03,
    ids_structure_penalty: float = 30.0,
    stroke_relation_penalty: float = 3.0,
    local_score_slack: float = 2.0,
) -> JointRouteSolution | None:
    """Choose one route per stroke under whole-glyph exclusive coverage pressure.

    A few shared pixels are free because real strokes meet. Longer repeated
    traversal is penalized, while covering previously unexplained skeleton is
    rewarded. No annotation coordinates participate in this search.
    """
    if any(not alternatives for alternatives in alternatives_by_stroke):
        return None
    ids_specs = _ids_structure_specs(candidate.root) if candidate is not None else ()
    relation_cache: dict[
        tuple[tuple[Pixel, ...], tuple[Pixel, ...], float, float], float
    ] = {}
    # route score, routes, occupied pixels, repeated count, IDS cost, relation cost
    states = [(0.0, (), frozenset(), 0, 0.0, 0.0)]
    for alternatives in alternatives_by_stroke:
        placement_confidence = alternatives[0].placement_confidence
        adaptive_slack = local_score_slack * (1.0 - placement_confidence)
        local_limit = alternatives[0].score + adaptive_slack
        eligible_alternatives = tuple(
            route for route in alternatives if route.score <= local_limit
        )
        expanded = []
        for score, routes, occupied, repeated, _structure, relation_cost in states:
            for route in eligible_alternatives:
                pixels = frozenset(route.pixels)
                overlap = len(pixels & occupied)
                newly_covered = len(pixels - occupied)
                charged_overlap = max(0, overlap - free_contact_pixels)
                proposed_routes = (*routes, route)
                structure = (
                    _ids_structure_cost(ids_specs, proposed_routes)
                    if ids_specs
                    else 0.0
                )
                new_relation_cost = (
                    _new_stroke_relation_cost(candidate, routes, route, relation_cache)
                    if candidate is not None
                    else 0.0
                )
                proposed = (
                    score
                    + route.score
                    + repeated_pixel_penalty * charged_overlap
                    - new_pixel_reward * newly_covered
                )
                expanded.append(
                    (
                        proposed,
                        proposed_routes,
                        occupied | pixels,
                        repeated + charged_overlap,
                        structure,
                        relation_cost + new_relation_cost,
                    )
                )
        expanded.sort(
            key=lambda item: (
                item[0]
                + ids_structure_penalty * item[4]
                + stroke_relation_penalty * item[5],
                item[3],
                -len(item[2]),
                tuple(route.pixels for route in item[1]),
            )
        )
        states = expanded[:beam_width]
    best = min(
        states,
        key=lambda item: (
            item[0]
            + residual_pixel_penalty * (skeleton_pixel_count - len(item[2]))
            + ids_structure_penalty * item[4]
            + stroke_relation_penalty * item[5],
            item[3],
            -len(item[2]),
        ),
    )
    score, routes, occupied, repeated, structure, relation_cost = best
    residual = max(0, skeleton_pixel_count - len(occupied))
    return JointRouteSolution(
        routes,
        score
        + residual_pixel_penalty * residual
        + ids_structure_penalty * structure
        + stroke_relation_penalty * relation_cost,
        len(occupied),
        repeated,
        residual,
        structure,
        relation_cost,
    )
