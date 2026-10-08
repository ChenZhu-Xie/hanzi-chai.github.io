"""Skeleton-only cut/no-cut hypotheses induced by recursive IDS constraints."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .candidate_graph import CandidateGraph, compile_candidate_graph
from .domain import DecompositionRequest
from .grammar import GlyphRepository
from .multiscale import MultiScaleCertificate, certify_multiscale
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .skeleton import generate_skeleton
from .topology import SkeletonTopology, compress_skeleton
from .topology_certificate import TopologyCertificate, certify_skeleton
from .topology_features import TopologyDescriptors, describe_topology


@dataclass(frozen=True)
class CutHypothesis:
    rank: int
    node_id: int
    operator: str
    axis: str
    boundary: float
    node_position: tuple[float, float]
    first_edges: tuple[int, ...]
    second_edges: tuple[int, ...]
    boundary_proximity: float
    side_separation: float
    separates_graph: bool
    downstream_side_purity: float
    boundary_span: float
    structural_gain_over_no_cut: float
    auto_accepted: bool = False


@dataclass(frozen=True)
class CutAudit:
    request: DecompositionRequest
    topology: SkeletonTopology
    descriptors: TopologyDescriptors
    topology_certificate: TopologyCertificate
    multiscale_certificate: MultiScaleCertificate
    candidate: CandidateGraph
    hypotheses: tuple[CutHypothesis, ...]
    expected_cross_child_contacts: int


def _far_position(edge, node_id: int, size: int) -> tuple[float, float]:
    if edge.start_node == node_id:
        point = edge.path[-1]
    elif edge.end_node == node_id:
        point = edge.path[0]
    else:
        raise ValueError(f"edge {edge.id} is not incident to node {node_id}")
    return point[0] / size, point[1] / size


def _cross_child_contacts(candidate: CandidateGraph) -> int:
    stroke_by_id = {stroke.index: stroke for stroke in candidate.strokes}
    count = 0
    for relation in candidate.relations:
        if not relation.exact_intersection:
            continue
        first = stroke_by_id[relation.first_stroke]
        second = stroke_by_id[relation.second_stroke]
        first_child = first.component_path[0] if first.component_path else None
        second_child = second.component_path[0] if second.component_path else None
        count += first_child != second_child
    return count


def _reachable_edges(
    topology: SkeletonTopology,
    seeds: tuple[int, ...],
    blocked_node: int,
) -> set[int]:
    edge_by_id = {edge.id: edge for edge in topology.edges}
    reachable = set(seeds)
    pending = list(seeds)
    while pending:
        edge = edge_by_id[pending.pop()]
        for node_id in {edge.start_node, edge.end_node} - {blocked_node}:
            for adjacent in topology.nodes[node_id].incident_edges:
                if adjacent not in reachable:
                    reachable.add(adjacent)
                    pending.append(adjacent)
    return reachable


def _downstream_purity(
    topology: SkeletonTopology,
    first_reachable: set[int],
    second_reachable: set[int],
    coordinate_index: int,
    boundary: float,
    size: int,
) -> float:
    edge_by_id = {edge.id: edge for edge in topology.edges}

    def coordinate(edge_id: int) -> float:
        edge = edge_by_id[edge_id]
        values = [point[coordinate_index] / size for point in edge.path]
        return sum(values) / len(values)

    correct = sum(coordinate(edge_id) <= boundary for edge_id in first_reachable)
    correct += sum(coordinate(edge_id) >= boundary for edge_id in second_reachable)
    total = len(first_reachable) + len(second_reachable)
    return correct / total if total else 0.0


def _boundary_span(
    topology: SkeletonTopology,
    edge_ids: set[int],
    coordinate_index: int,
    boundary: float,
    size: int,
) -> float:
    edge_by_id = {edge.id: edge for edge in topology.edges}
    coordinates = [
        point[coordinate_index] / size
        for edge_id in edge_ids
        for point in edge_by_id[edge_id].path
    ]
    if not coordinates:
        return 0.0
    left_extent = max(0.0, boundary - min(coordinates)) / max(boundary, 1e-9)
    right_extent = max(0.0, max(coordinates) - boundary) / max(1 - boundary, 1e-9)
    return min(1.0, 2 * min(left_extent, right_extent))


def generate_root_cut_hypotheses(
    topology: SkeletonTopology,
    descriptors: TopologyDescriptors,
    candidate: CandidateGraph,
    size: int,
    *,
    boundary_band: float = 0.5,
) -> tuple[CutHypothesis, ...]:
    """Rank root IDS cuts without modifying the skeleton or auto-accepting one."""
    operator = candidate.root.operator
    if operator not in {"⿰", "⿱"}:
        return ()
    axis = "x" if operator == "⿰" else "y"
    coordinate_index = 0 if axis == "x" else 1
    boundary = 0.5
    edge_by_id = {edge.id: edge for edge in topology.edges}
    proposals = []
    for node in topology.nodes:
        if node.degree < 3:
            continue
        position = (node.centre[0] / size, node.centre[1] / size)
        boundary_distance = abs(position[coordinate_index] - boundary)
        if boundary_distance > boundary_band:
            continue
        first_edges = []
        second_edges = []
        neutral_edges = []
        distances = []
        for edge_id in node.incident_edges:
            far = _far_position(edge_by_id[edge_id], node.id, size)
            signed = far[coordinate_index] - boundary
            distances.append(abs(signed))
            if signed < -0.015:
                first_edges.append(edge_id)
            elif signed > 0.015:
                second_edges.append(edge_id)
            else:
                neutral_edges.append(edge_id)
        proximity = math.exp(-boundary_distance / max(boundary_band / 2, 1e-9))
        separation = min(1.0, sum(distances) / len(distances) / 0.25)
        for assignments in product((0, 1), repeat=len(neutral_edges)):
            first = (
                *first_edges,
                *(edge for edge, side in zip(neutral_edges, assignments) if side == 0),
            )
            second = (
                *second_edges,
                *(edge for edge, side in zip(neutral_edges, assignments) if side == 1),
            )
            if not first or not second:
                continue
            first_reachable = _reachable_edges(topology, tuple(first), node.id)
            second_reachable = _reachable_edges(topology, tuple(second), node.id)
            separates = not bool(first_reachable & second_reachable)
            purity = (
                _downstream_purity(
                    topology,
                    first_reachable,
                    second_reachable,
                    coordinate_index,
                    boundary,
                    size,
                )
                if separates
                else 0.0
            )
            span = _boundary_span(
                topology,
                first_reachable | second_reachable,
                coordinate_index,
                boundary,
                size,
            )
            proposals.append(
                CutHypothesis(
                    rank=0,
                    node_id=node.id,
                    operator=operator,
                    axis=axis,
                    boundary=boundary,
                    node_position=position,
                    first_edges=tuple(sorted(first)),
                    second_edges=tuple(sorted(second)),
                    boundary_proximity=proximity,
                    side_separation=separation,
                    separates_graph=separates,
                    downstream_side_purity=purity,
                    boundary_span=span,
                    structural_gain_over_no_cut=(
                        proximity * separation * purity * span if separates else 0.0
                    ),
                )
            )
    proposals.sort(key=lambda item: (-item.structural_gain_over_no_cut, item.node_id))
    return tuple(
        CutHypothesis(**{**item.__dict__, "rank": rank})
        for rank, item in enumerate(proposals, 1)
    )


def build_cut_audit(request: DecompositionRequest, size: int = 256) -> CutAudit:
    """Use PDF only to freeze a certified skeleton, then reason on graphs only."""
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

    # From this boundary onward, neither PDF geometry nor filled ink is passed
    # to the reasoning functions.
    topology = compress_skeleton(skeleton)
    descriptors = describe_topology(topology, size)
    repository = GlyphRepository.load(Path(request.glyph_data_path))
    candidate = compile_candidate_graph(repository, request.candidate_glyph_id)
    hypotheses = generate_root_cut_hypotheses(topology, descriptors, candidate, size)
    return CutAudit(
        request=request,
        topology=topology,
        descriptors=descriptors,
        topology_certificate=certificate,
        multiscale_certificate=multiscale,
        candidate=candidate,
        hypotheses=hypotheses,
        expected_cross_child_contacts=_cross_child_contacts(candidate),
    )


def render_cut_audit(audit: CutAudit, size: int = 256) -> str:
    scale = 100 / size
    paths = []
    for edge in audit.topology.edges:
        points = " ".join(
            f"{(x + 0.5) * scale:.3f},{(y + 0.5) * scale:.3f}" for x, y in edge.path
        )
        paths.append(f'<polyline class="edge" points="{points}"/>')
    markers = []
    rows = []
    for hypothesis in audit.hypotheses:
        x, y = hypothesis.node_position
        markers.append(
            f'<g class="proposal"><circle cx="{x * 100:.3f}" cy="{y * 100:.3f}" r="1.4"/>'
            f'<text x="{x * 100 + 1.8:.3f}" y="{y * 100 - 1.8:.3f}">H{hypothesis.rank}</text></g>'
        )
        rows.append(
            f"<tr><td>H{hypothesis.rank}</td><td>node {hypothesis.node_id}</td>"
            f"<td>{hypothesis.first_edges}</td><td>{hypothesis.second_edges}</td>"
            f"<td>{hypothesis.boundary_proximity:.3f}</td>"
            f"<td>{hypothesis.side_separation:.3f}</td>"
            f"<td>{'是' if hypothesis.separates_graph else '否'}</td>"
            f"<td>{hypothesis.downstream_side_purity:.3f}</td>"
            f"<td>{hypothesis.boundary_span:.3f}</td>"
            f"<td>{hypothesis.structural_gain_over_no_cut:.3f}</td>"
            "<td>否：等待逐笔全局推演</td></tr>"
        )
    boundary = (
        '<line class="boundary" x1="50" y1="0" x2="50" y2="100"/>'
        if audit.candidate.root.operator == "⿰"
        else '<line class="boundary" x1="0" y1="50" x2="100" y2="50"/>'
    )
    return f"""<!doctype html><meta charset="utf-8"><title>cut hypotheses</title>
<style>body{{font:14px system-ui;margin:20px;background:#f3f5f8;color:#172033}}main{{display:grid;grid-template-columns:minmax(480px,.9fr) minmax(700px,1.4fr);gap:16px}}section{{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px}}svg{{width:100%;aspect-ratio:1}}.edge{{fill:none;stroke:#334155;stroke-width:.55;stroke-linecap:round}}.boundary{{stroke:#f59e0b;stroke-width:.45;stroke-dasharray:2 1}}.proposal circle{{fill:#ef4444;stroke:white;stroke-width:.3}}.proposal text{{font-size:3px;font-weight:700;fill:#991b1b}}table{{width:100%;border-collapse:collapse}}th,td{{padding:6px;border-bottom:1px solid #e2e8f0;text-align:left}}.safe{{background:#dcfce7;color:#166534;padding:9px;border-radius:6px}}</style>
<h1>{html.escape(audit.request.unicode)} · {html.escape(audit.request.source)} · candidate {audit.candidate.glyph_id}</h1>
<p class="safe">PDF 仅用于生成骨架；拓扑证书与多尺度证书通过后，以下候选全部只由骨架图＋candidate/IDS 产生。H0＝所有路口保持不切。当前 H1…Hn 只是结构先验排序，自动采用全部为否。</p>
<p>根 IDS：{audit.candidate.root.operator}；candidate 跨根子树物理接触：{audit.expected_cross_child_contacts}；候选切位：{len(audit.hypotheses)}</p>
<main><section><svg viewBox="0 0 100 100">{boundary}{"".join(paths)}{"".join(markers)}</svg></section><section><table><thead><tr><th>假设</th><th>位置</th><th>子树 0 边</th><th>子树 1 边</th><th>边界接近</th><th>局部分离</th><th>真正断图</th><th>下游纯度</th><th>跨边界幅度</th><th>相对 H0 结构增益</th><th>自动切</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section></main>"""
