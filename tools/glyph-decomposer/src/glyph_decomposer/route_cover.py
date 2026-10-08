"""Route candidate leaf strokes over a certified skeleton and expose semantic cuts.

Real glyph ink may physically join two logical components. The target is not
one connected component per IDS leaf. Candidate strokes are instead routed on
the skeleton; a semantic cut is requested wherever different leaf occurrences
share one skeleton junction.
"""

from __future__ import annotations

import heapq
import html
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .candidate_graph import (
    CandidateGraph,
    CandidateStroke,
    compile_candidate_graph,
    compile_catalog_candidate_graph,
)
from .domain import DecompositionRequest
from .grammar import GlyphRepository
from .multiscale import MultiScaleCertificate, certify_multiscale
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .skeleton import generate_skeleton
from .topology import SkeletonTopology, compress_skeleton
from .topology_certificate import TopologyCertificate, certify_skeleton

Pixel = tuple[int, int]
LeafKey = tuple[tuple[int, ...], int]

_NEIGHBOURS = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
)
_COLOURS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#dc2626",
    "#0d9488",
)


@dataclass(frozen=True)
class StrokeRoute:
    stroke_index: int
    feature: str
    leaf_key: LeafKey
    pixels: tuple[Pixel, ...]
    snapped_landmarks: tuple[Pixel, ...]
    mean_candidate_distance: float
    endpoint_snap_distance: float
    route_length: float
    candidate_length: float


@dataclass(frozen=True)
class SemanticCut:
    node_id: int
    position: tuple[float, float]
    leaf_keys: tuple[LeafKey, ...]
    stroke_indices: tuple[int, ...]
    shared_pixel_count: int
    reason: str
    auto_accepted: bool = False


@dataclass(frozen=True)
class LeafEdgeAssignment:
    edge_id: int
    leaf_key: LeafKey
    cost: float
    margin: float


@dataclass(frozen=True)
class RouteCoverAudit:
    request: DecompositionRequest
    skeleton: np.ndarray
    topology: SkeletonTopology
    topology_certificate: TopologyCertificate
    multiscale_certificate: MultiScaleCertificate
    candidate: CandidateGraph
    routes: tuple[StrokeRoute, ...]
    route_cuts: tuple[SemanticCut, ...]
    edge_assignments: tuple[LeafEdgeAssignment, ...]
    cuts: tuple[SemanticCut, ...]
    expected_leaf_count: int
    raw_skeleton_components: int
    candidate_cross_leaf_contacts: int
    observed_cross_leaf_contacts: int
    unmatched_strokes: tuple[int, ...]


def _pixel_neighbours(point: Pixel, shape: tuple[int, int]):
    y, x = point
    for dy, dx in _NEIGHBOURS:
        neighbour = y + dy, x + dx
        if 0 <= neighbour[0] < shape[0] and 0 <= neighbour[1] < shape[1]:
            yield neighbour


def _components(skeleton: np.ndarray) -> tuple[frozenset[Pixel], ...]:
    available = {tuple(point) for point in np.argwhere(skeleton)}
    output = []
    while available:
        seed = min(available)
        available.remove(seed)
        component = {seed}
        pending = deque([seed])
        while pending:
            point = pending.popleft()
            for neighbour in _pixel_neighbours(point, skeleton.shape):
                if neighbour in available:
                    available.remove(neighbour)
                    component.add(neighbour)
                    pending.append(neighbour)
        output.append(frozenset(component))
    return tuple(output)


def _candidate_bounds(candidate: CandidateGraph) -> tuple[float, float, float, float]:
    points = np.asarray(
        [point for stroke in candidate.strokes for point in stroke.points], dtype=float
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def _skeleton_bounds(skeleton: np.ndarray) -> tuple[float, float, float, float]:
    pixels = np.argwhere(skeleton)
    scale = 100.0 / skeleton.shape[0]
    return (
        float((pixels[:, 1].min() + 0.5) * scale),
        float((pixels[:, 0].min() + 0.5) * scale),
        float((pixels[:, 1].max() + 0.5) * scale),
        float((pixels[:, 0].max() + 0.5) * scale),
    )


def _align_points(
    points: tuple[tuple[float, float], ...],
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    sx0, sy0, sx1, sy1 = source_bounds
    tx0, ty0, tx1, ty1 = target_bounds
    sx = (tx1 - tx0) / max(sx1 - sx0, 1e-9)
    sy = (ty1 - ty0) / max(sy1 - sy0, 1e-9)
    values = np.asarray(points, dtype=float)
    return np.column_stack(
        (tx0 + (values[:, 0] - sx0) * sx, ty0 + (values[:, 1] - sy0) * sy)
    )


def _distance_to_polyline(point_xy: np.ndarray, polyline: np.ndarray) -> float:
    best = math.inf
    for start, end in pairwise(polyline):
        vector = end - start
        denominator = float(np.dot(vector, vector))
        ratio = (
            0.0
            if denominator == 0
            else float(np.clip(np.dot(point_xy - start, vector) / denominator, 0, 1))
        )
        best = min(best, float(np.linalg.norm(point_xy - (start + ratio * vector))))
    return best


def _nearest_pixel(point_xy: np.ndarray, pixels: frozenset[Pixel], scale: float):
    return min(
        pixels,
        key=lambda pixel: (
            (pixel[1] * scale - point_xy[0]) ** 2
            + (pixel[0] * scale - point_xy[1]) ** 2
        ),
    )


def _choose_component(
    polyline: np.ndarray,
    components: tuple[frozenset[Pixel], ...],
    scale: float,
) -> frozenset[Pixel]:
    probes = polyline[
        np.linspace(0, len(polyline) - 1, min(7, len(polyline))).astype(int)
    ]

    def score(component: frozenset[Pixel]):
        coordinates = np.asarray([(x * scale, y * scale) for y, x in component])
        return float(
            np.mean(
                [
                    np.sqrt(np.min(np.sum((coordinates - point) ** 2, axis=1)))
                    for point in probes
                ]
            )
        )

    return min(components, key=score)


def _shortest_guided_path(
    skeleton_pixels: frozenset[Pixel],
    shape: tuple[int, int],
    start: Pixel,
    goal: Pixel,
    guide: np.ndarray,
    scale: float,
) -> tuple[Pixel, ...]:
    if start == goal:
        return (start,)
    guide_distance = {
        point: _distance_to_polyline(
            np.asarray((point[1] * scale, point[0] * scale)), guide
        )
        for point in skeleton_pixels
    }
    distance = {start: 0.0}
    previous: dict[Pixel, Pixel] = {}
    queue = [(0.0, start)]
    while queue:
        cost, point = heapq.heappop(queue)
        if point == goal:
            break
        if cost != distance.get(point):
            continue
        for neighbour in _pixel_neighbours(point, shape):
            if neighbour not in skeleton_pixels:
                continue
            step = math.hypot(neighbour[0] - point[0], neighbour[1] - point[1])
            proximity = (guide_distance[point] + guide_distance[neighbour]) / 2
            proposed = cost + step * (1 + min(proximity / 3.0, 5.0))
            if proposed < distance.get(neighbour, math.inf):
                distance[neighbour] = proposed
                previous[neighbour] = point
                heapq.heappush(queue, (proposed, neighbour))
    if goal not in previous:
        return ()
    output = [goal]
    while output[-1] != start:
        output.append(previous[output[-1]])
    return tuple(reversed(output))


def _stroke_route(
    stroke: CandidateStroke,
    aligned: np.ndarray,
    skeleton: np.ndarray,
    components: tuple[frozenset[Pixel], ...],
) -> StrokeRoute | None:
    scale = 100.0 / skeleton.shape[0]
    component = _choose_component(aligned, components, scale)
    landmark_indices = np.unique(
        np.linspace(0, len(aligned) - 1, min(9, len(aligned))).round().astype(int)
    )
    landmarks_xy = aligned[landmark_indices]
    landmarks = tuple(_nearest_pixel(point, component, scale) for point in landmarks_xy)
    route: list[Pixel] = []
    for start, end in pairwise(landmarks):
        segment = _shortest_guided_path(
            component, skeleton.shape, start, end, aligned, scale
        )
        if not segment:
            return None
        route.extend(segment if not route else segment[1:])
    if not route:
        route = [landmarks[0]]
    route_xy = np.asarray([(x * scale, y * scale) for y, x in route])
    mean_distance = float(
        np.mean([_distance_to_polyline(point, aligned) for point in route_xy])
    )
    endpoint_distance = float(
        np.mean(
            [
                np.linalg.norm(route_xy[0] - aligned[0]),
                np.linalg.norm(route_xy[-1] - aligned[-1]),
            ]
        )
    )
    route_length = float(
        sum(math.hypot(b[0] - a[0], b[1] - a[1]) * scale for a, b in pairwise(route))
    )
    candidate_length = float(
        sum(np.linalg.norm(end - start) for start, end in pairwise(aligned))
    )
    return StrokeRoute(
        stroke_index=stroke.index,
        feature=stroke.feature,
        leaf_key=(stroke.component_path, stroke.leaf_id),
        pixels=tuple(route),
        snapped_landmarks=landmarks,
        mean_candidate_distance=mean_distance,
        endpoint_snap_distance=endpoint_distance,
        route_length=route_length,
        candidate_length=candidate_length,
    )


def match_candidate_routes(
    skeleton: np.ndarray, candidate: CandidateGraph
) -> tuple[tuple[StrokeRoute, ...], tuple[int, ...]]:
    """Produce deterministic route hypotheses without annotation truth."""
    components = _components(skeleton)
    source_bounds = _candidate_bounds(candidate)
    target_bounds = _skeleton_bounds(skeleton)
    routes = []
    unmatched = []
    for stroke in candidate.strokes:
        aligned = _align_points(stroke.points, source_bounds, target_bounds)
        route = _stroke_route(stroke, aligned, skeleton, components)
        if route is None:
            unmatched.append(stroke.index)
        else:
            routes.append(route)
    return tuple(routes), tuple(unmatched)


def assign_topology_edges(
    topology: SkeletonTopology, candidate: CandidateGraph, size: int
) -> tuple[LeafEdgeAssignment, ...]:
    """Assign every maximal skeleton chain to exactly one IDS leaf occurrence.

    This is deliberately a complete, mutually exclusive partition. Candidate
    geometry supplies unary evidence only; no PDF fill geometry or truth path
    is available at this stage.
    """
    source_bounds = _candidate_bounds(candidate)
    skeleton_points = np.asarray(
        [point for edge in topology.edges for point in edge.path], dtype=float
    )
    target_bounds = (
        float(skeleton_points[:, 0].min() * 100 / size),
        float(skeleton_points[:, 1].min() * 100 / size),
        float(skeleton_points[:, 0].max() * 100 / size),
        float(skeleton_points[:, 1].max() * 100 / size),
    )
    aligned_by_leaf: dict[LeafKey, list[np.ndarray]] = defaultdict(list)
    for stroke in candidate.strokes:
        key = stroke.component_path, stroke.leaf_id
        aligned_by_leaf[key].append(
            _align_points(stroke.points, source_bounds, target_bounds)
        )
    leaves = tuple(sorted(aligned_by_leaf))
    candidate_length = {
        leaf: sum(
            float(sum(np.linalg.norm(end - start) for start, end in pairwise(guide)))
            for guide in guides
        )
        for leaf, guides in aligned_by_leaf.items()
    }
    costs: dict[int, dict[LeafKey, float]] = {}
    for edge in topology.edges:
        # Long chains receive more samples than short serifs, but a hard cap
        # keeps the cost table small and deterministic.
        path = np.asarray(edge.path, dtype=float) * (100 / size)
        indices = np.linspace(0, len(path) - 1, min(25, len(path))).astype(int)
        probes = path[indices]
        costs[edge.id] = {
            leaf: float(
                np.mean(
                    [
                        min(
                            _distance_to_polyline(point, guide)
                            for guide in aligned_by_leaf[leaf]
                        )
                        for point in probes
                    ]
                )
            )
            for leaf in leaves
        }

    total_edge_length = sum(edge.length for edge in topology.edges)
    total_candidate_length = sum(candidate_length.values())
    target_length = {
        leaf: total_edge_length * candidate_length[leaf] / total_candidate_length
        for leaf in leaves
    }
    multipliers = {leaf: 0.0 for leaf in leaves}
    best_objective = math.inf
    best_owners = None
    # Soft capacities stop a centrally located candidate leaf from swallowing
    # the graph. They guide a global partition; they are not an ink-area claim.
    for _ in range(120):
        owners = {
            edge.id: min(
                leaves,
                key=lambda leaf: (costs[edge.id][leaf] + multipliers[leaf], leaf),
            )
            for edge in topology.edges
        }
        owned_length = {leaf: 0.0 for leaf in leaves}
        for edge in topology.edges:
            owned_length[owners[edge.id]] += edge.length
        unary = sum(
            edge.length * costs[edge.id][owners[edge.id]] for edge in topology.edges
        ) / max(total_edge_length, 1e-9)
        capacity_error = sum(
            abs(owned_length[leaf] - target_length[leaf]) for leaf in leaves
        ) / max(total_edge_length, 1e-9)
        objective = unary + 8.0 * capacity_error
        if objective < best_objective and all(owned_length[leaf] for leaf in leaves):
            best_objective = objective
            best_owners = owners.copy()
        for leaf in leaves:
            relative_error = (owned_length[leaf] - target_length[leaf]) / max(
                target_length[leaf], 1e-9
            )
            multipliers[leaf] += 0.35 * relative_error
        centre = sum(multipliers.values()) / len(multipliers)
        multipliers = {leaf: value - centre for leaf, value in multipliers.items()}
    owners = best_owners or owners

    output = []
    for edge in topology.edges:
        ordered = sorted(costs[edge.id].items(), key=lambda item: (item[1], item[0]))
        chosen = owners[edge.id]
        chosen_cost = costs[edge.id][chosen]
        alternative = min(
            (cost for leaf, cost in ordered if leaf != chosen), default=chosen_cost
        )
        output.append(
            LeafEdgeAssignment(
                edge_id=edge.id,
                leaf_key=chosen,
                cost=chosen_cost,
                margin=alternative - chosen_cost,
            )
        )
    return tuple(output)


def find_partition_cuts(
    topology: SkeletonTopology,
    assignments: tuple[LeafEdgeAssignment, ...],
    size: int,
) -> tuple[SemanticCut, ...]:
    owner_by_edge = {
        assignment.edge_id: assignment.leaf_key for assignment in assignments
    }
    cuts = []
    for node in topology.nodes:
        owners = {owner_by_edge[edge_id] for edge_id in node.incident_edges}
        if len(owners) < 2:
            continue
        cuts.append(
            SemanticCut(
                node_id=node.id,
                position=(node.centre[0] * 100 / size, node.centre[1] * 100 / size),
                leaf_keys=tuple(sorted(owners)),
                stroke_indices=(),
                shared_pixel_count=len(node.pixels),
                reason="incident maximal chains have different IDS leaf owners",
            )
        )
    return tuple(cuts)


def _candidate_cross_leaf_contacts(candidate: CandidateGraph) -> int:
    strokes = {stroke.index: stroke for stroke in candidate.strokes}
    return sum(
        relation.exact_intersection
        and (
            strokes[relation.first_stroke].component_path
            != strokes[relation.second_stroke].component_path
        )
        for relation in candidate.relations
    )


def find_semantic_cuts(
    topology: SkeletonTopology,
    routes: tuple[StrokeRoute, ...],
    size: int,
) -> tuple[SemanticCut, ...]:
    """Find physical nodes used by multiple logical IDS leaf occurrences."""
    routes_by_pixel: dict[Pixel, list[StrokeRoute]] = defaultdict(list)
    for route in routes:
        for pixel in set(route.pixels):
            routes_by_pixel[pixel].append(route)
    cuts = []
    for node in topology.nodes:
        nearby = set(node.pixels)
        if not nearby:
            nearby.add((round(node.centre[1]), round(node.centre[0])))
        owners = {
            route.leaf_key
            for pixel in nearby
            for route in routes_by_pixel.get(pixel, ())
        }
        if len(owners) < 2:
            continue
        strokes = {
            route.stroke_index
            for pixel in nearby
            for route in routes_by_pixel.get(pixel, ())
        }
        shared_pixels = sum(
            len({route.leaf_key for route in routes_by_pixel.get(pixel, ())}) >= 2
            for pixel in nearby
        )
        cuts.append(
            SemanticCut(
                node_id=node.id,
                position=(node.centre[0] * 100 / size, node.centre[1] * 100 / size),
                leaf_keys=tuple(sorted(owners)),
                stroke_indices=tuple(sorted(strokes)),
                shared_pixel_count=shared_pixels,
                reason="different IDS leaf routes share one physical skeleton node",
            )
        )
    return tuple(cuts)


def build_route_cover_audit(
    request: DecompositionRequest, size: int = 256
) -> RouteCoverAudit:
    cells = parse_cells(Path(request.bbox_path))
    codepoint = int(request.unicode.removeprefix("U+"), 16)
    cell = find_cell(cells, codepoint, request.source)
    page = export_page_svg(Path(request.pdf_path), cell.page)
    geometry = extract_cell_geometry(page, cell)
    multiscale = certify_multiscale(geometry)
    ink, skeleton = generate_skeleton(geometry, size)
    certificate = certify_skeleton(ink, skeleton)
    if not certificate.certified or not multiscale.certified:
        raise ValueError("skeleton failed topology or multi-scale certification")
    # Strict boundary: routing receives only the frozen, certified skeleton.
    topology = compress_skeleton(skeleton)
    repository = GlyphRepository.load(Path(request.glyph_data_path))
    try:
        candidate = compile_candidate_graph(repository, request.candidate_glyph_id)
    except KeyError:
        if not request.candidate_catalog_path:
            raise
        catalog = json.loads(
            Path(request.candidate_catalog_path).read_text(encoding="utf-8")
        )
        candidate = compile_catalog_candidate_graph(
            catalog, codepoint, request.candidate_glyph_id
        )
    routes, unmatched = match_candidate_routes(skeleton, candidate)
    route_cuts = find_semantic_cuts(topology, routes, size)
    assignments = assign_topology_edges(topology, candidate, size)
    cuts = find_partition_cuts(topology, assignments, size)
    return RouteCoverAudit(
        request=request,
        skeleton=skeleton,
        topology=topology,
        topology_certificate=certificate,
        multiscale_certificate=multiscale,
        candidate=candidate,
        routes=routes,
        route_cuts=route_cuts,
        edge_assignments=assignments,
        cuts=cuts,
        expected_leaf_count=len({route.leaf_key for route in routes}),
        raw_skeleton_components=len(_components(skeleton)),
        candidate_cross_leaf_contacts=_candidate_cross_leaf_contacts(candidate),
        observed_cross_leaf_contacts=len(cuts),
        unmatched_strokes=unmatched,
    )


def _leaf_label(key: LeafKey) -> str:
    path, glyph_id = key
    return f"{glyph_id}@{'.'.join(map(str, path)) or 'root'}"


def render_route_cover_audit(audit: RouteCoverAudit) -> str:
    size = audit.skeleton.shape[0]
    scale = 100 / size
    leaf_keys = sorted({route.leaf_key for route in audit.routes})
    colours = {
        key: _COLOURS[index % len(_COLOURS)] for index, key in enumerate(leaf_keys)
    }
    skeleton_path = "".join(
        f"M{x * scale:.3f},{y * scale:.3f}h{scale:.3f}v{scale:.3f}h-{scale:.3f}z"
        for y, x in np.argwhere(audit.skeleton)
    )
    route_paths = []
    rows = []
    for route in audit.routes:
        colour = colours[route.leaf_key]
        points = " ".join(
            f"{(x + 0.5) * scale:.3f},{(y + 0.5) * scale:.3f}" for y, x in route.pixels
        )
        route_paths.append(
            f'<polyline class="route" style="--route:{colour}" points="{points}"><title>'
            f"stroke {route.stroke_index} {html.escape(route.feature)} · "
            f"{_leaf_label(route.leaf_key)}</title></polyline>"
        )
        rows.append(
            f"<tr><td>{route.stroke_index}</td><td>{html.escape(route.feature)}</td>"
            f'<td><span style="color:{colour}">■</span> {_leaf_label(route.leaf_key)}</td>'
            f"<td>{route.mean_candidate_distance:.2f}</td>"
            f"<td>{route.endpoint_snap_distance:.2f}</td>"
            f"<td>{route.route_length:.1f}/{route.candidate_length:.1f}</td></tr>"
        )
    assignment_by_edge = {
        assignment.edge_id: assignment for assignment in audit.edge_assignments
    }
    owned_edges = []
    edge_rows = []
    for edge in audit.topology.edges:
        assignment = assignment_by_edge[edge.id]
        colour = colours[assignment.leaf_key]
        points = " ".join(
            f"{(x + 0.5) * scale:.3f},{(y + 0.5) * scale:.3f}" for x, y in edge.path
        )
        owned_edges.append(
            f'<polyline class="owned-edge" style="--route:{colour}" points="{points}">'
            f"<title>chain {edge.id} → {_leaf_label(assignment.leaf_key)} · "
            f"cost {assignment.cost:.2f} · margin {assignment.margin:.2f}</title></polyline>"
        )
        edge_rows.append(
            f"<tr><td>{edge.id}</td>"
            f'<td><span style="color:{colour}">■</span> {_leaf_label(assignment.leaf_key)}</td>'
            f"<td>{assignment.cost:.2f}</td><td>{assignment.margin:.2f}</td></tr>"
        )
    cut_marks = "".join(
        f'<g class="cut"><circle cx="{cut.position[0]:.3f}" cy="{cut.position[1]:.3f}" r="1.45"/>'
        f'<text x="{cut.position[0] + 1.8:.3f}" y="{cut.position[1] - 1.8:.3f}">S{index}</text>'
        f"<title>{html.escape(', '.join(_leaf_label(key) for key in cut.leaf_keys))}</title></g>"
        for index, cut in enumerate(audit.cuts, 1)
    )
    cut_rows = "".join(
        f"<tr><td>S{index}</td><td>node {cut.node_id}</td>"
        f"<td>{html.escape(', '.join(_leaf_label(key) for key in cut.leaf_keys))}</td>"
        f"<td>{cut.stroke_indices}</td><td>{cut.shared_pixel_count}</td>"
        f"<td>否：等待全局路由覆盖一致性</td></tr>"
        for index, cut in enumerate(audit.cuts, 1)
    )
    summary = {
        "unicode": audit.request.unicode,
        "source": audit.request.source,
        "candidateGlyphId": audit.candidate.glyph_id,
        "candidateStrokeCount": len(audit.candidate.strokes),
        "matchedStrokeCount": len(audit.routes),
        "unmatchedStrokes": audit.unmatched_strokes,
        "expectedIDSLeafOccurrences": audit.expected_leaf_count,
        "rawSkeletonConnectedComponents": audit.raw_skeleton_components,
        "candidateCrossLeafContacts": audit.candidate_cross_leaf_contacts,
        "observedCrossLeafSharedNodes": audit.observed_cross_leaf_contacts,
        "semanticCutCount": len(audit.cuts),
        "independentRouteCutCountNegativeControl": len(audit.route_cuts),
        "exclusiveAssignedChainCount": len(audit.edge_assignments),
        "autoAcceptedCuts": 0,
        "truthUsed": False,
    }
    return f"""<!doctype html><meta charset="utf-8"><title>leaf route cover</title>
<style>body{{font:14px system-ui;margin:20px;background:#f3f5f8;color:#172033}}main{{display:grid;grid-template-columns:minmax(480px,.85fr) minmax(720px,1.4fr);gap:16px}}section{{background:#fff;border:1px solid #cbd5e1;border-radius:10px;padding:14px}}svg{{width:100%;aspect-ratio:1;background:#fff}}.skeleton{{fill:#d7dee8}}.owned-edge{{fill:none;stroke:var(--route);stroke-width:.85;stroke-linecap:round;stroke-linejoin:round}}.owned-edge:hover{{stroke-width:1.8}}.route{{fill:none;stroke:var(--route);stroke-width:.45;stroke-dasharray:1 1;stroke-linecap:round;stroke-linejoin:round;opacity:.25}}.cut circle{{fill:#fff;stroke:#ef4444;stroke-width:.65}}.cut text{{fill:#991b1b;font-size:3px;font-weight:700}}table{{width:100%;border-collapse:collapse}}th,td{{padding:6px;border-bottom:1px solid #e2e8f0;text-align:left}}pre{{background:#111827;color:#e5edf8;padding:12px;border-radius:8px}}.notice{{background:#fef3c7;color:#92400e;padding:9px;border-radius:7px}}</style>
<h1>{html.escape(audit.request.unicode)} · {html.escape(audit.request.source)} · IDS 叶部件路由覆盖</h1>
<p class="notice">实线是对全部最大无分叉骨架链的互斥、完整叶归属；淡虚线是旧的逐笔独立路由负面对照。红圈 S 是相邻链归属不同叶的物理节点，因此需要语义切口。物理墨迹无需断开，节点所有权需拆开。当前仍是 dry-run，未自动采用切口。</p>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<main><section><svg viewBox="0 0 100 100"><path class="skeleton" d="{skeleton_path}"/>{"".join(owned_edges)}{"".join(route_paths)}{cut_marks}</svg></section><section><h2>互斥全链归属</h2><table><thead><tr><th>链</th><th>IDS 叶实例</th><th>代价</th><th>次优差距</th></tr></thead><tbody>{"".join(edge_rows)}</tbody></table><h2>跨叶语义切口</h2><table><thead><tr><th>#</th><th>节点</th><th>叶部件</th><th>笔画</th><th>共享像素</th><th>自动采用</th></tr></thead><tbody>{cut_rows}</tbody></table><details><summary>逐笔独立路由（仅负面对照）</summary><table><thead><tr><th>#</th><th>笔画</th><th>IDS 叶实例</th><th>平均偏差</th><th>端点偏差</th><th>路由/候选长度</th></tr></thead><tbody>{"".join(rows)}</tbody></table></details></section></main>"""
