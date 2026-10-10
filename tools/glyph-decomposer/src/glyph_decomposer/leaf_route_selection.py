"""Coarse-to-fine leaf placement validation using realizable stroke routes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .candidate_graph import CandidateGraph
from .leaf_alignment import (
    JointLeafAlignmentSolution,
    LeafAlignment,
    alignment_ids_cost,
)
from .primitive_route import (
    JointRouteSolution,
    Pixel,
    enumerate_primitive_routes,
    select_joint_routes,
)


@dataclass(frozen=True)
class LeafRouteRefinement:
    component_path: tuple[int, ...]
    original: LeafAlignment
    selected: LeafAlignment
    original_route_score: float | None
    selected_route_score: float | None
    evaluated_count: int


@dataclass(frozen=True)
class LeafRouteGlobalRefinement:
    component_path: tuple[int, ...]
    original: LeafAlignment
    selected: LeafAlignment
    baseline_score: float
    selected_score: float
    evaluated_count: int
    locked_paths: tuple[tuple[int, ...], ...]

    @property
    def changed(self) -> bool:
        return self.original != self.selected


def _avoids_locked_segments(
    route: tuple[Pixel, ...],
    locked_pixels: frozenset[Pixel],
    *,
    maximum_shared_run: int = 3,
) -> bool:
    """Allow contact with locked leaves, but never travel along their ink."""
    shared_run = 0
    for pixel in route:
        if pixel in locked_pixels:
            shared_run += 1
            if shared_run > maximum_shared_run:
                return False
        else:
            shared_run = 0
    return True


def _select_refinement_trial(
    original: LeafAlignment,
    baseline: JointRouteSolution,
    trials: tuple[tuple[LeafAlignment, JointRouteSolution], ...],
    *,
    maximum_ids_regression: float = 0.05,
) -> tuple[LeafAlignment, JointRouteSolution]:
    """Accept only a strict whole-glyph improvement with no safety regression."""
    eligible = tuple(
        (alignment, solution)
        for alignment, solution in trials
        if solution.score < baseline.score - 1e-9
        and solution.repeated_pixel_count <= baseline.repeated_pixel_count
        and solution.ids_structure_cost
        <= baseline.ids_structure_cost + maximum_ids_regression
    )
    if not eligible:
        return original, baseline
    return min(
        eligible,
        key=lambda item: (
            item[1].score,
            item[1].repeated_pixel_count,
            item[1].residual_pixel_count,
            item[0].score,
            item[0].bounds,
        ),
    )


def _refinement_shortlist(
    pool: tuple[LeafAlignment, ...], current: LeafAlignment
) -> tuple[LeafAlignment, ...]:
    by_scale: dict[tuple[float, float], list[LeafAlignment]] = {}
    for item in pool:
        if not (0.8 <= item.scale_x <= 1.7 and 0.8 <= item.scale_y <= 1.7):
            continue
        by_scale.setdefault((item.scale_x, item.scale_y), []).append(item)
    selected = [current]
    for scale in sorted(by_scale):
        for item in by_scale[scale][:2]:
            if item not in selected:
                selected.append(item)
    return tuple(selected)


def _needs_route_refinement(alignment: LeafAlignment, stroke_count: int) -> bool:
    uniform_scale = math.sqrt(alignment.scale_x * alignment.scale_y)
    suspicious_scale = (
        uniform_scale <= 0.75
        or min(alignment.scale_x, alignment.scale_y) <= 0.8
        or max(alignment.scale_x, alignment.scale_y) >= 1.7
    )
    suspicious_fit = alignment.score >= 3.0
    if stroke_count < 2 or not (suspicious_scale or suspicious_fit):
        return False
    if stroke_count == 2:
        return max(alignment.scale_x, alignment.scale_y) >= 1.7 or suspicious_fit
    return True


def refine_suspicious_leaf_alignments(
    skeleton: np.ndarray,
    candidate: CandidateGraph,
    pools: dict[tuple[int, ...], tuple[LeafAlignment, ...]],
    solution: JointLeafAlignmentSolution,
    *,
    shortest_path_cache: dict | None = None,
    maximum_ids_regression: float = 0.05,
) -> tuple[dict[tuple[int, ...], LeafAlignment], tuple[LeafRouteRefinement, ...]]:
    """Validate suspicious shrunken leaves using cheap, real stroke routing."""
    selected = {item.component_path: item for item in solution.alignments}
    refinements = []
    for path, original in tuple(selected.items()):
        strokes = tuple(
            stroke for stroke in candidate.strokes if stroke.component_path == path
        )
        if not _needs_route_refinement(original, len(strokes)):
            continue
        shortlist = _refinement_shortlist(pools[path], original)
        scored = []
        baseline_structure = alignment_ids_cost(candidate, tuple(selected.values()))
        for alignment in shortlist:
            proposed_alignments = tuple(
                alignment if selected_path == path else item
                for selected_path, item in selected.items()
            )
            proposed_structure = alignment_ids_cost(candidate, proposed_alignments)
            # Leaf-local routing may fit well by stealing a dense neighbouring
            # component. Preserve the structural validity established by the
            # preceding joint solve.
            if proposed_structure > baseline_structure + maximum_ids_regression:
                continue
            alternatives = tuple(
                enumerate_primitive_routes(
                    skeleton,
                    candidate,
                    stroke,
                    endpoint_count=2,
                    leaf_alignment=alignment,
                    shortest_path_cache=shortest_path_cache,
                )
                for stroke in strokes
            )
            routed = select_joint_routes(
                alternatives,
                int(skeleton.sum()),
                candidate=candidate,
                beam_width=50,
            )
            if routed is not None:
                scored.append((routed.score, alignment))
        scored.sort(key=lambda item: (item[0], item[1].score, item[1].bounds))
        replacement = scored[0][1] if scored else original
        scores = {alignment: score for score, alignment in scored}
        selected[path] = replacement
        refinements.append(
            LeafRouteRefinement(
                path,
                original,
                replacement,
                scores.get(original),
                scores.get(replacement),
                len(scored),
            )
        )
    return selected, tuple(refinements)


def refine_low_confidence_leaf_routes(
    skeleton: np.ndarray,
    candidate: CandidateGraph,
    pools: dict[tuple[int, ...], tuple[LeafAlignment, ...]],
    selected_alignments: dict[tuple[int, ...], LeafAlignment],
    alternatives_by_stroke: tuple[tuple, ...],
    joint: JointRouteSolution,
    *,
    shortest_path_cache: dict | None = None,
    route_enumerator=enumerate_primitive_routes,
    alignment_score_trigger: float = 3.0,
    route_score_trigger: float = 5.0,
    joint_relation_trigger: float = 10.0,
    joint_repeated_pixel_trigger: int = 8,
    maximum_locked_shared_run: int = 3,
    shortlist_limit: int = 16,
    maximum_refined_leaves: int = 1,
) -> tuple[
    dict[tuple[int, ...], LeafAlignment],
    tuple[tuple, ...],
    JointRouteSolution,
    tuple[LeafRouteGlobalRefinement, ...],
]:
    """Re-open weak leaves while keeping confident routes reserved.

    Locked routes remain singleton alternatives. A target leaf may touch them
    at a junction, but a route that follows their ink for a sustained run is
    excluded. A replacement is accepted only by whole-glyph joint scoring.
    """
    if (
        joint.stroke_relation_cost < joint_relation_trigger
        and joint.repeated_pixel_count < joint_repeated_pixel_trigger
    ):
        return selected_alignments, alternatives_by_stroke, joint, ()
    strokes_by_path: dict[tuple[int, ...], tuple] = {}
    for path in selected_alignments:
        strokes_by_path[path] = tuple(
            stroke for stroke in candidate.strokes if stroke.component_path == path
        )
    routes_by_index = {route.stroke_index: route for route in joint.routes}

    def severity(path: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
        route_score = max(
            (routes_by_index[stroke.index].score for stroke in strokes_by_path[path]),
            default=0.0,
        )
        return (-route_score, -selected_alignments[path].score, path)

    suspicious = sorted(
        (
            path
            for path, alignment in selected_alignments.items()
            if alignment.score >= alignment_score_trigger
            or any(
                routes_by_index[stroke.index].score >= route_score_trigger
                for stroke in strokes_by_path[path]
            )
        ),
        key=severity,
    )[:maximum_refined_leaves]
    current_alignments = dict(selected_alignments)
    current_alternatives = list(alternatives_by_stroke)
    current_joint = joint
    refinements = []
    for path in suspicious:
        original = current_alignments[path]
        baseline_score = current_joint.score
        current_routes = {route.stroke_index: route for route in current_joint.routes}
        locked_routes = tuple(
            route
            for route in current_joint.routes
            if candidate.strokes[route.stroke_index].component_path != path
        )
        locked_pixels = frozenset(
            pixel for route in locked_routes for pixel in route.pixels
        )
        locked_paths = tuple(
            sorted(
                {
                    candidate.strokes[route.stroke_index].component_path
                    for route in locked_routes
                }
            )
        )
        shortlist = [original]
        for alignment in pools[path]:
            if alignment in shortlist:
                continue
            if not (0.8 <= alignment.scale_x <= 1.7):
                continue
            if not (0.8 <= alignment.scale_y <= 1.7):
                continue
            shortlist.append(alignment)
            if len(shortlist) == shortlist_limit:
                break

        trials = []
        trial_alternatives = {}
        for alignment in shortlist:
            target_alternatives = {}
            for stroke in strokes_by_path[path]:
                alternatives = route_enumerator(
                    skeleton,
                    candidate,
                    stroke,
                    leaf_alignment=alignment,
                    shortest_path_cache=shortest_path_cache,
                )
                alternatives = tuple(
                    route
                    for route in alternatives
                    if _avoids_locked_segments(
                        route.pixels,
                        locked_pixels,
                        maximum_shared_run=maximum_locked_shared_run,
                    )
                )
                if not alternatives:
                    break
                target_alternatives[stroke.index] = alternatives
            if len(target_alternatives) != len(strokes_by_path[path]):
                continue
            proposed = tuple(
                target_alternatives[stroke.index]
                if stroke.component_path == path
                else (current_routes[stroke.index],)
                for stroke in candidate.strokes
            )
            routed = select_joint_routes(
                proposed,
                int(skeleton.sum()),
                candidate=candidate,
                beam_width=100,
                local_score_slack=6.0,
            )
            if routed is None:
                continue
            trials.append((alignment, routed))
            trial_alternatives[alignment] = target_alternatives

        selected, selected_joint = _select_refinement_trial(
            original, current_joint, tuple(trials)
        )
        if selected != original:
            current_alignments[path] = selected
            for stroke_index, alternatives in trial_alternatives[selected].items():
                current_alternatives[stroke_index] = alternatives
            current_joint = selected_joint
        refinements.append(
            LeafRouteGlobalRefinement(
                path,
                original,
                selected,
                baseline_score,
                selected_joint.score,
                len(trials),
                locked_paths,
            )
        )
    return (
        current_alignments,
        tuple(current_alternatives),
        current_joint,
        tuple(refinements),
    )
