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
from .primitive_route import enumerate_primitive_routes, select_joint_routes


@dataclass(frozen=True)
class LeafRouteRefinement:
    component_path: tuple[int, ...]
    original: LeafAlignment
    selected: LeafAlignment
    original_route_score: float | None
    selected_route_score: float | None
    evaluated_count: int


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
            if proposed_structure > baseline_structure + 0.10:
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
