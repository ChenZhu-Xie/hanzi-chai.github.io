"""Self-contained vector review artifact for one blind decomposition."""

from __future__ import annotations

import html
import json
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon

from .domain import ComponentProgram, DecompositionRequest
from .pipeline import DecompositionArtifacts
from .recursive import terminal_nodes

COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#db2777", "#0891b2")


def _polygons(geometry):
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return geometry.geoms
    return tuple(part for part in geometry.geoms if isinstance(part, Polygon))


def geometry_path_data(geometry) -> str:
    def ring_data(ring) -> str:
        points = list(ring.coords)
        return "M " + " L ".join(f"{x:.4f} {y:.4f}" for x, y in points) + " Z"

    fragments = []
    for polygon in _polygons(geometry):
        fragments.append(ring_data(polygon.exterior))
        fragments.extend(ring_data(interior) for interior in polygon.interiors)
    return " ".join(fragments)


def _truth_overlays(path: str | None) -> str:
    if not path:
        return ""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output = []
    for annotation in payload.get("annotations", ()):
        if annotation.get("type") not in {"lasso", "polygon"}:
            continue
        points = annotation.get("fixedPoints") or annotation.get("points") or ()
        if len(points) < 3:
            continue
        coordinate_text = " ".join(f"{point[0]:.4f},{point[1]:.4f}" for point in points)
        label = html.escape(str(annotation.get("label", "?")))
        output.append(
            f'<polygon points="{coordinate_text}" fill="none" stroke="#dc2626" '
            f'stroke-width="0.8" stroke-dasharray="2 1"><title>{label}</title></polygon>'
        )
    return "".join(output)


def _program_tree(program: ComponentProgram) -> str:
    label = f"{program.glyph_id}"
    if program.operator:
        label += f" · {program.operator}"
    if program.stroke_features:
        label += " · " + "、".join(program.stroke_features)
    children = "".join(_program_tree(child) for child in program.children)
    return f"<li><code>{html.escape(label)}</code>{f'<ul>{children}</ul>' if children else ''}</li>"


def render_review_html(
    request: DecompositionRequest, artifacts: DecompositionArtifacts
) -> str:
    result = artifacts.result
    source_path = geometry_path_data(artifacts.source_geometry)
    terminals = terminal_nodes(artifacts.decomposition)
    colored = "".join(
        f'<path d="{geometry_path_data(node.geometry)}" fill="{COLORS[index % len(COLORS)]}" '
        f'fill-rule="evenodd"><title>{node.program.glyph_id} · {node.evidence.status}</title></path>'
        for index, node in enumerate(terminals)
    )
    legend = " · ".join(
        f'<span style="color:{COLORS[index % len(COLORS)]}">■</span> '
        f"{node.program.glyph_id} ({node.evidence.status})"
        for index, node in enumerate(terminals)
    )
    if result.partition.axis == "x":
        cut_line = f'<line x1="{result.partition.cut}" y1="0" x2="{result.partition.cut}" y2="100"/>'
    else:
        cut_line = f'<line x1="0" y1="{result.partition.cut}" x2="100" y2="{result.partition.cut}"/>'
    evaluation = result.evaluation
    metrics = "未提供人工真值"
    if evaluation:
        metrics = (
            f"归类墨迹覆盖 {evaluation.classified_ink_ratio:.2%}；"
            f"归属准确率 {evaluation.component_accuracy:.2%}；"
            f"两子部件 IoU {evaluation.child_ious[0]:.2%} / {evaluation.child_ious[1]:.2%}"
        )
    recursive_metrics = ""
    if result.recursive_evaluation:
        recursive_metrics = (
            f"<p><strong>递归终端：</strong>{result.recursive_evaluation.terminal_count} 个；"
            f"归属准确率 {result.recursive_evaluation.component_accuracy:.2%}；"
            f"平均 IoU {result.recursive_evaluation.mean_iou:.2%}；"
            f"最低 IoU {result.recursive_evaluation.minimum_iou:.2%}。</p>"
        )
    truth = _truth_overlays(request.annotation_path)
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>{result.unicode}-{result.source} 确定性拆分审阅</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:16px}}
.card{{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:14px}}
svg{{width:100%;aspect-ratio:1;border:1px solid #e2e8f0;background:white}}
.cut line{{stroke:#111827;stroke-width:.45;stroke-dasharray:2 1}}
code{{font-size:13px}} ul{{margin:.35rem 0;padding-left:1.4rem}} .note{{color:#475569}}
</style>
<h1>{result.unicode} · {result.source} 源 · candidate {result.candidate_glyph_id}</h1>
<p><strong>盲推证据：</strong>IDS 根算子 {result.program.operator}；{result.partition.axis}={result.partition.cut:g}；
跨切线墨迹 {result.partition.crossing_ratio:.3%}；重构误差 {result.partition.reconstruction_error:.3g}。</p>
<div class="panels">
 <section class="card"><h2>PDF 原生矢量</h2><svg viewBox="0 0 100 100"><path d="{source_path}" fill="#111827" fill-rule="evenodd"/></svg></section>
 <section class="card"><h2>盲推 IDS 递归分区</h2><svg viewBox="0 0 100 100">{colored}<g class="cut">{cut_line}</g></svg><p>{legend}</p></section>
 <section class="card"><h2>事后人工真值轮廓</h2><svg viewBox="0 0 100 100"><path d="{source_path}" fill="#cbd5e1" fill-rule="evenodd"/>{truth}</svg></section>
</div>
<section class="card"><h2>结果</h2><p><strong>根层：</strong>{metrics}</p>{recursive_metrics}<p class="note">人工标注只在盲推完成后用于评分和红色虚线叠加，不进入求解目标。</p></section>
<section class="card"><h2>递归候选程序</h2><ul>{_program_tree(result.program)}</ul></section>
</html>"""
