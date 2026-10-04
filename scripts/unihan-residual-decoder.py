"""Sequential beam decoder over directed roads and residual PDF ink."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


def _load(filename: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DIRECTED = _load("unihan-directed-skeleton.py", "unihan_directed_skeleton_for_decoder")
RESIDUAL = _load("unihan-residual-ink.py", "unihan_residual_ink_for_decoder")
ORDER = _load("unihan-stroke-order.py", "unihan_stroke_order_for_decoder")


class RankedRoute(Protocol):
    points: np.ndarray
    pixels: frozenset[tuple[int, int]]
    score: float
    evidence: dict


@dataclass(frozen=True)
class ResidualDecoderState:
    stroke_index: int
    routes: tuple[RankedRoute, ...]
    route_edge_ids: tuple[tuple[int, ...], ...]
    regions: tuple[object, ...]
    ledger: object
    closed_occurrences: frozenset[tuple[int, int]]
    score: float
    evidence: tuple[dict, ...]


@dataclass(frozen=True)
class ResidualDecodeResult:
    routes: tuple[RankedRoute, ...]
    regions: tuple[object, ...]
    ledger: object
    status: str
    review_reasons: tuple[str, ...]
    best_score: float
    runner_up_margin: float | None
    steps: tuple[dict, ...]


def _nearest_point_index(graph, point: np.ndarray) -> int:
    points = graph.points[:, ::-1].astype(float)
    return int(np.argmin(np.linalg.norm(points - np.asarray(point, dtype=float), axis=1)))


def _route_hard_rejections(
    route: RankedRoute,
    route_edge_ids: tuple[int, ...],
    expectation,
    state: ResidualDecoderState,
    directed,
    graph,
    pen_candidates: dict[int, object],
) -> list[str]:
    rejections = []
    if not route_edge_ids:
        return ["route-does-not-follow-directed-roads"]
    first_edge = directed.edges[route_edge_ids[0]]
    if (
        not ORDER.sector_compatible(
            expectation.expected_sector, first_edge.start_sector
        )
    ):
        rejections.append("wrong-start-direction")

    start_index = _nearest_point_index(graph, route.points[0])
    candidate = pen_candidates.get(start_index)
    if candidate is None:
        rejections.append("start-is-not-a-gated-road-end")
    else:
        rejections.extend(candidate.hard_rejections)
        if first_edge.edge_id not in candidate.outgoing_edge_ids:
            rejections.append("start-edge-incompatible-with-stroke")

    consumed_roads = {
        directed.edges[edge_id].road_id for edge_id in state.ledger.consumed_edge_ids
    }
    route_roads = {directed.edges[edge_id].road_id for edge_id in route_edge_ids}
    if consumed_roads & route_roads:
        rejections.append("route-reuses-consumed-road")

    key = (expectation.component_id, expectation.occurrence)
    if key in state.closed_occurrences:
        rejections.append("closed-component-reentry")

    route_turns = tuple(route.evidence.get("routeJunctionTurns", ()))
    if expectation.expected_turns and route_turns != expectation.expected_turns:
        rejections.append("forbidden-junction-exit")

    for position, (incoming_id, outgoing_id) in enumerate(
        zip(route_edge_ids, route_edge_ids[1:])
    ):
        incoming = directed.edges[incoming_id]
        outgoing = directed.edges[outgoing_id]
        if incoming.end_gate != outgoing.start_gate:
            rejections.append("disconnected-route-edges")
            continue
        if incoming.road_id == outgoing.road_id:
            rejections.append("route-reverses-at-gate")
            continue
        if position < len(expectation.expected_turns):
            legal = DIRECTED.legal_exit_edges(
                directed,
                incoming_id,
                expected_sector=outgoing.start_sector,
                expected_turns=(expectation.expected_turns[position],),
            )
            if outgoing_id not in legal:
                rejections.append("forbidden-junction-exit")
    return list(dict.fromkeys(rejections))


def _partial_result(
    state: ResidualDecoderState,
    reason: str,
) -> ResidualDecodeResult:
    return ResidualDecodeResult(
        routes=state.routes,
        regions=state.regions,
        ledger=state.ledger,
        status="needs-review",
        review_reasons=(reason,),
        best_score=state.score,
        runner_up_margin=None,
        steps=state.evidence,
    )


def decode_residual_routes(
    target: np.ndarray,
    graph,
    strokes: list[dict],
    ranked_routes: list[list[RankedRoute]],
    source: str,
    codepoint: int,
    learned_model: dict | None = None,
    normative_catalog=None,
    beam_width: int = 350,
) -> ResidualDecodeResult:
    del learned_model  # learned costs are already bounded inside ranked routes
    directed = DIRECTED.build_directed_skeleton(graph)
    expectations = ORDER.compile_stroke_expectations(
        strokes, source, codepoint, normative_catalog
    )
    initial = ResidualDecoderState(
        stroke_index=0,
        routes=(),
        route_edge_ids=(),
        regions=(),
        ledger=RESIDUAL.initial_ledger(target),
        closed_occurrences=frozenset(),
        score=0.0,
        evidence=(),
    )
    beam = [initial]
    full_ledger = RESIDUAL.initial_ledger(target)
    region_partition_cache: dict[tuple[int, ...], object] = {}

    for index, (expectation, options) in enumerate(zip(expectations, ranked_routes)):
        next_beam = []
        rejection_counts: dict[str, int] = {}
        for state in beam:
            candidates = ORDER.rank_pen_down_candidates(
                expectation, state.ledger, directed, graph
            )
            candidates_by_point = {item.point_index: item for item in candidates}
            for option_rank, route in enumerate(options):
                edge_ids = DIRECTED.trace_route_edges(directed, graph, route.points)
                rejections = _route_hard_rejections(
                    route,
                    edge_ids,
                    expectation,
                    state,
                    directed,
                    graph,
                    candidates_by_point,
                )
                if rejections:
                    for reason in rejections:
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                base_region = region_partition_cache.get(edge_ids)
                if base_region is None:
                    base_region = RESIDUAL.recover_stroke_region(
                        target,
                        graph,
                        directed,
                        edge_ids,
                        route.points,
                        full_ledger,
                    )
                    region_partition_cache[edge_ids] = base_region
                available = state.ledger.available
                region = RESIDUAL.StrokeRegion(
                    mask=base_region.mask & available,
                    contact_mask=base_region.contact_mask & available,
                    forbidden_leak_mask=base_region.forbidden_leak_mask & available,
                )
                if not region.mask.any():
                    rejection_counts["empty-residual-region"] = (
                        rejection_counts.get("empty-residual-region", 0) + 1
                    )
                    continue
                if region.forbidden_leak_mask.any():
                    rejection_counts["forbidden-branch-leakage"] = (
                        rejection_counts.get("forbidden-branch-leakage", 0) + 1
                    )
                    continue

                start_index = _nearest_point_index(graph, route.points[0])
                pen = candidates_by_point[start_index]
                ledger = RESIDUAL.advance_ledger(
                    state.ledger, region, index, edge_ids
                )
                key = (expectation.component_id, expectation.occurrence)
                closed = state.closed_occurrences | ({key} if expectation.closes_component else set())
                step = {
                    "stroke": index + 1,
                    "componentId": expectation.component_id,
                    "occurrence": expectation.occurrence,
                    "routeRank": option_rank + 1,
                    "routeEdgeIds": list(edge_ids),
                    "routeScore": float(route.score),
                    "penDownScore": float(pen.score),
                    "hardRejections": [],
                    "penDownAlternatives": [
                        {
                            "pointIndex": int(item.point_index),
                            "outgoingEdgeIds": list(item.outgoing_edge_ids),
                            "score": float(item.score),
                            "evidence": list(item.evidence),
                            "hardRejections": list(item.hard_rejections),
                            "accepted": item.point_index == start_index,
                        }
                        for item in candidates[:8]
                    ],
                    "orderEvidence": [item.__dict__ for item in expectation.evidence],
                    "unexplainedInk": int(ledger.unexplained.sum()),
                    "remainingLaterStrokeFeasible": True,
                }
                next_beam.append(
                    ResidualDecoderState(
                        stroke_index=index + 1,
                        routes=state.routes + (route,),
                        route_edge_ids=state.route_edge_ids + (edge_ids,),
                        regions=state.regions + (region,),
                        ledger=ledger,
                        closed_occurrences=frozenset(closed),
                        score=state.score + float(route.score) + 0.2 * float(pen.score),
                        evidence=state.evidence + (step,),
                    )
                )
        if not next_beam:
            best_partial = min(beam, key=lambda item: item.score)
            key = (expectation.component_id, expectation.occurrence)
            reason = (
                f"closed-component-reentry-stroke-{index + 1}"
                if key in best_partial.closed_occurrences
                else f"no-feasible-route-stroke-{index + 1}"
            )
            result = _partial_result(best_partial, reason)
            if rejection_counts:
                steps = result.steps + (
                    {
                        "stroke": index + 1,
                        "hardRejectionCounts": rejection_counts,
                    },
                )
                return ResidualDecodeResult(**{**result.__dict__, "steps": steps})
            return result
        for state in next_beam:
            if state.evidence:
                state.evidence[-1]["hardRejectionCounts"] = dict(rejection_counts)
        next_beam.sort(key=lambda item: item.score)
        beam = next_beam[:beam_width]

    best = beam[0]
    runner_up_margin = beam[1].score - best.score if len(beam) > 1 else None
    review_reasons = []
    if any(
        evidence.rule == "source-order-unavailable"
        for expectation in expectations
        for evidence in expectation.evidence
    ):
        review_reasons.append("source-order-unavailable")
    unexplained_ratio = float(best.ledger.unexplained.sum()) / max(1, int(np.asarray(target).sum()))
    if unexplained_ratio >= 0.15:
        review_reasons.append("large-unexplained-ink")
    if runner_up_margin is not None and runner_up_margin < 0.005:
        review_reasons.append("near-tied-global-solutions")
    return ResidualDecodeResult(
        routes=best.routes,
        regions=best.regions,
        ledger=best.ledger,
        status="needs-review" if review_reasons else "safe-candidate",
        review_reasons=tuple(review_reasons),
        best_score=best.score,
        runner_up_margin=runner_up_margin,
        steps=best.evidence,
    )
