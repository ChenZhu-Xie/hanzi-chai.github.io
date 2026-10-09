"""Compile repository candidate strokes and recursive IDS into a compact graph."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np
from shapely.geometry import LineString
from shapely.ops import nearest_points

from .candidate import (
    IDS_INTERVALS,
    StrokeSeed,
    compile_stroke_seeds,
    normalize_stroke_seed_order,
    stroke_points,
)
from .grammar import GlyphRepository
from .topology_features import simplify_path


@dataclass(frozen=True)
class CandidateComponent:
    glyph_id: int
    path: tuple[int, ...]
    kind: str
    operator: str | None
    bounds: tuple[float, float, float, float]
    stroke_indices: tuple[int, ...]
    children: tuple[CandidateComponent, ...]


@dataclass(frozen=True)
class CandidateStroke:
    index: int
    leaf_id: int
    occurrence: int
    component_path: tuple[int, ...]
    feature: str
    points: tuple[tuple[float, float], ...]
    simplified_path: tuple[tuple[float, float], ...]
    length: float
    chord_ratio: float
    total_turn: float
    start_tangent: float
    end_tangent: float
    bounds: tuple[float, float, float, float]
    commands: tuple[str, ...]


@dataclass(frozen=True)
class CandidateStrokeRelation:
    first_stroke: int
    second_stroke: int
    distance: float
    first_position: float
    second_position: float
    exact_intersection: bool
    same_leaf_occurrence: bool


@dataclass(frozen=True)
class CandidateGraph:
    glyph_id: int
    root: CandidateComponent
    strokes: tuple[CandidateStroke, ...]
    relations: tuple[CandidateStrokeRelation, ...]


def _compose(transform, interval):
    sx, sy, tx, ty = transform
    x0, y0, x1, y1 = interval
    return sx * (x1 - x0) / 100, sy * (y1 - y0) / 100, tx + sx * x0, ty + sy * y0


def _bounds(transform) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = transform
    return tx, ty, tx + 100 * sx, ty + 100 * sy


def _angle(vector) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _angular_distance(first: float, second: float) -> float:
    difference = abs(first - second) % (2 * math.pi)
    return min(difference, 2 * math.pi - difference)


def _describe_stroke(index: int, seed: StrokeSeed) -> CandidateStroke:
    simplified = simplify_path(seed.points, tolerance=0.75)
    segments = [
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        for start, end in pairwise(simplified)
        if start != end
    ]
    length = float(sum(np.linalg.norm(segment) for segment in segments))
    chord = float(
        np.linalg.norm(np.asarray(seed.points[-1]) - np.asarray(seed.points[0]))
    )
    directions = [_angle(segment) for segment in segments]
    total_turn = sum(
        _angular_distance(first, second) for first, second in pairwise(directions)
    )
    coordinates = np.asarray(seed.points, dtype=float)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    return CandidateStroke(
        index=index,
        leaf_id=seed.leaf_id,
        occurrence=seed.occurrence,
        component_path=seed.component_path,
        feature=seed.feature,
        points=seed.points,
        simplified_path=simplified,
        length=length,
        chord_ratio=chord / length if length else 1.0,
        total_turn=total_turn,
        start_tangent=directions[0] if directions else 0.0,
        end_tangent=directions[-1] if directions else 0.0,
        bounds=(
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ),
        commands=seed.commands,
    )


def _compile_component(
    repository: GlyphRepository,
    identifier: int,
    path: tuple[int, ...],
    transform,
    strokes: tuple[CandidateStroke, ...],
    active: tuple[int, ...],
) -> CandidateComponent:
    if identifier in active:
        raise ValueError(f"cyclic glyph references through {identifier}")
    record = repository.record(identifier)
    kind = str(record["type"])
    indices = tuple(
        stroke.index for stroke in strokes if stroke.component_path[: len(path)] == path
    )
    if kind == "component":
        return CandidateComponent(
            glyph_id=identifier,
            path=path,
            kind=kind,
            operator=None,
            bounds=_bounds(transform),
            stroke_indices=indices,
            children=(),
        )
    operator = record.get("operator")
    intervals = IDS_INTERVALS.get(operator)
    references = record.get("references", ())
    if intervals is None or len(intervals) != len(references):
        raise ValueError(f"candidate graph compiler does not support {operator!r}")
    children = tuple(
        _compile_component(
            repository,
            int(reference["id"]),
            (*path, index),
            _compose(transform, interval),
            strokes,
            (*active, identifier),
        )
        for index, (reference, interval) in enumerate(zip(references, intervals))
    )
    return CandidateComponent(
        glyph_id=identifier,
        path=path,
        kind=kind,
        operator=operator,
        bounds=_bounds(transform),
        stroke_indices=indices,
        children=children,
    )


def _stroke_relation(
    first: CandidateStroke, second: CandidateStroke
) -> CandidateStrokeRelation:
    first_line = LineString(first.points)
    second_line = LineString(second.points)
    first_nearest, second_nearest = nearest_points(first_line, second_line)
    distance = float(first_nearest.distance(second_nearest))
    return CandidateStrokeRelation(
        first_stroke=first.index,
        second_stroke=second.index,
        distance=distance,
        first_position=float(first_line.project(first_nearest, normalized=True)),
        second_position=float(second_line.project(second_nearest, normalized=True)),
        exact_intersection=distance <= 1e-9,
        same_leaf_occurrence=(
            first.leaf_id == second.leaf_id
            and first.occurrence == second.occurrence
            and first.component_path == second.component_path
        ),
    )


def compile_candidate_graph(
    repository: GlyphRepository, glyph_id: int
) -> CandidateGraph:
    seeds = compile_stroke_seeds(repository, glyph_id)
    strokes = tuple(_describe_stroke(index, seed) for index, seed in enumerate(seeds))
    root = _compile_component(
        repository, glyph_id, (), (1.0, 1.0, 0.0, 0.0), strokes, ()
    )
    relations = tuple(
        _stroke_relation(first, second) for first, second in combinations(strokes, 2)
    )
    return CandidateGraph(glyph_id, root, strokes, relations)


def compile_catalog_candidate_graph(
    catalog: dict, codepoint: int, glyph_id: int
) -> CandidateGraph:
    """Compile a locally reviewed candidate whose synthetic root is not upstream.

    Catalog rows already contain absolute 0..100 candidate paths and explicit
    leaf ownership. No annotation coordinates are read here.
    """
    row = next(
        (
            item
            for item in catalog.get("rows", ())
            if int(item.get("unicode", -1)) == codepoint
            and str(glyph_id) in item.get("candidates", {})
        ),
        None,
    )
    if row is None:
        raise KeyError(
            f"candidate {glyph_id} for U+{codepoint:04X} absent from catalog"
        )
    raw_strokes = row["candidates"][str(glyph_id)]
    leaves = row["candidateLeafSvgs"][str(glyph_id)]
    node_metadata: dict[tuple[int, int], dict] = {}
    child_keys: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for leaf in leaves:
        occurrence = int(leaf.get("occurrence", 0))
        hierarchy = leaf.get("hierarchy") or (
            {"id": int(leaf["leafId"]), "type": "component", "label": "末级部件"},
            {"id": glyph_id, "type": "glyph", "label": ""},
        )
        keys = [
            (int(item["id"]), occurrence if index == 0 else 0)
            for index, item in enumerate(hierarchy)
        ]
        for key, item in zip(keys, hierarchy):
            node_metadata[key] = item
        for child_key, parent_key in pairwise(keys):
            children = child_keys.setdefault(parent_key, [])
            if child_key not in children:
                children.append(child_key)

    path_by_key: dict[tuple[int, int], tuple[int, ...]] = {}

    def assign_paths(key: tuple[int, int], path: tuple[int, ...]):
        path_by_key[key] = path
        for index, child_key in enumerate(child_keys.get(key, ())):
            assign_paths(child_key, (*path, index))

    root_key = glyph_id, 0
    assign_paths(root_key, ())
    ownership = {}
    for leaf in leaves:
        leaf_id = int(leaf["leafId"])
        occurrence = int(leaf.get("occurrence", 0))
        for stroke_index in leaf["strokeIndices"]:
            ownership[int(stroke_index)] = (
                path_by_key[(leaf_id, occurrence)],
                leaf_id,
                occurrence,
            )
    if sorted(ownership) != list(range(len(raw_strokes))):
        raise ValueError("catalog candidate has incomplete leaf stroke ownership")
    seeds = normalize_stroke_seed_order(
        StrokeSeed(
            leaf_id=ownership[index][1],
            occurrence=ownership[index][2],
            feature=stroke.get("feature", "unknown"),
            points=tuple(stroke_points(stroke)),
            component_path=ownership[index][0],
            commands=tuple(
                str(curve["command"]) for curve in stroke.get("curveList", ())
            ),
        )
        for index, stroke in enumerate(raw_strokes)
    )
    strokes = tuple(_describe_stroke(index, seed) for index, seed in enumerate(seeds))

    def compile_node(key: tuple[int, int], path: tuple[int, ...]) -> CandidateComponent:
        identifier, _occurrence = key
        child_node_keys = child_keys.get(key, ())
        children = tuple(
            compile_node(child_key, (*path, index))
            for index, child_key in enumerate(child_node_keys)
        )
        indices = tuple(
            stroke.index
            for stroke in strokes
            if stroke.component_path[: len(path)] == path
        )
        coordinates = np.asarray(
            [point for index in indices for point in strokes[index].points], dtype=float
        )
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        metadata = node_metadata.get(key, {})
        label = str(metadata.get("label", ""))
        operator = (
            label
            if label.startswith(("⿰", "⿱", "⿲", "⿳", "⿸", "⿹"))
            else None
        )
        return CandidateComponent(
            glyph_id=identifier,
            path=path,
            kind="compound" if children else "component",
            operator=operator,
            bounds=(
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            ),
            stroke_indices=indices,
            children=children,
        )

    root = compile_node(root_key, ())
    relations = tuple(
        _stroke_relation(first, second) for first, second in combinations(strokes, 2)
    )
    return CandidateGraph(glyph_id, root, strokes, relations)


_COLOURS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#dc2626",
)


def _component_rectangles(node: CandidateComponent, depth: int = 0) -> str:
    x0, y0, x1, y1 = node.bounds
    label = f"path {node.path or ('root',)} · glyph {node.glyph_id}"
    if node.operator:
        label += f" · {node.operator}"
    rectangle = (
        f'<rect class="component depth-{depth % 4}" x="{x0}" y="{y0}" '
        f'width="{x1 - x0}" height="{y1 - y0}"><title>{html.escape(label)}</title></rect>'
    )
    return rectangle + "".join(
        _component_rectangles(child, depth + 1) for child in node.children
    )


def render_candidate_graph(graph: CandidateGraph) -> str:
    leaf_keys = sorted(
        {(stroke.component_path, stroke.leaf_id) for stroke in graph.strokes}
    )
    colour_by_leaf = {
        key: _COLOURS[index % len(_COLOURS)] for index, key in enumerate(leaf_keys)
    }
    paths = []
    rows = []
    relation_by_stroke = {
        stroke.index: [
            relation
            for relation in graph.relations
            if stroke.index in (relation.first_stroke, relation.second_stroke)
        ]
        for stroke in graph.strokes
    }
    for stroke in graph.strokes:
        colour = colour_by_leaf[(stroke.component_path, stroke.leaf_id)]
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in stroke.simplified_path)
        contacts = sum(
            relation.exact_intersection for relation in relation_by_stroke[stroke.index]
        )
        paths.append(
            f'<polyline points="{points}" style="--stroke:{colour}"><title>'
            f"{stroke.index}: {html.escape(stroke.feature)} · leaf {stroke.leaf_id} · "
            f"path {stroke.component_path} · {contacts} exact contacts</title></polyline>"
        )
        rows.append(
            f"<tr><td>{stroke.index}</td><td>{html.escape(stroke.feature)}</td>"
            f'<td><span style="color:{colour}">■</span> {stroke.leaf_id}</td>'
            f"<td>{stroke.component_path}</td><td>{stroke.length:.1f}</td>"
            f"<td>{stroke.chord_ratio:.2f}</td><td>{contacts}</td></tr>"
        )
    exact = [relation for relation in graph.relations if relation.exact_intersection]
    summary = {
        "candidateGlyphId": graph.glyph_id,
        "strokeCount": len(graph.strokes),
        "leafOccurrenceCount": len(leaf_keys),
        "pairwiseRelations": len(graph.relations),
        "exactStrokeContacts": len(exact),
        "truthUsed": False,
    }
    return f"""<!doctype html><meta charset="utf-8"><title>candidate graph {graph.glyph_id}</title>
<style>body{{font:14px system-ui;margin:20px;color:#172033;background:#f3f5f8}}main{{display:grid;grid-template-columns:minmax(420px,1fr) minmax(620px,1.3fr);gap:16px}}section{{background:#fff;border:1px solid #cbd5e1;border-radius:10px;padding:14px}}svg{{width:100%;aspect-ratio:1;background:#fff}}polyline{{fill:none;stroke:var(--stroke);stroke-width:1.05;stroke-linecap:round;stroke-linejoin:round}}polyline:hover{{stroke-width:2.2}}.component{{fill:none;stroke-width:.32;stroke-dasharray:1.5 1}}.depth-0{{stroke:#0f172a}}.depth-1{{stroke:#ef4444}}.depth-2{{stroke:#f59e0b}}.depth-3{{stroke:#8b5cf6}}table{{width:100%;border-collapse:collapse}}th,td{{padding:6px;border-bottom:1px solid #e2e8f0;text-align:left}}pre{{background:#111827;color:#e5edf8;padding:12px;border-radius:8px}}</style>
<h1>candidate {graph.glyph_id} 的递归 IDS / 笔画图</h1><p>矩形表示递归 IDS 作用域；同一叶部件实例使用同色；路径保留方向和书写顺序。该图未读取 PDF 或人工答案。</p><pre>{html.escape(str(summary))}</pre>
<main><section><svg viewBox="0 0 100 100">{_component_rectangles(graph.root)}{"".join(paths)}</svg></section><section><table><thead><tr><th>#</th><th>笔画</th><th>叶部件</th><th>IDS path</th><th>长度</th><th>弦比</th><th>交点</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section></main>"""
