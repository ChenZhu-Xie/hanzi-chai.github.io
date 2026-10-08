"""Skeleton-only directed-route continuation for historic ink audits."""

from __future__ import annotations

import heapq
import html
import json
import math
from pathlib import Path


def load_historic_review(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    marker = "const D="
    start = source.index(marker) + len(marker)
    end = source.index("const canvas", start)
    payload = source[start:end].strip().removesuffix(";").strip()
    return json.loads(payload)


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1e-9:
        return 0.0, 0.0
    return vector[0] / length, vector[1] / length


def _direction(points: list[tuple[int, int]], window: int = 7) -> tuple[float, float]:
    start = points[max(0, len(points) - window)]
    end = points[-1]
    return _unit((end[0] - start[0], end[1] - start[1]))


def _neighbours(point: tuple[int, int], skeleton: set[tuple[int, int]]):
    x, y = point
    return [
        (x + dx, y + dy)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx or dy) and (x + dx, y + dy) in skeleton
    ]


def _advance(
    route: list[tuple[int, int]],
    skeleton: set[tuple[int, int]],
    *,
    preserve_axis: bool = False,
    maximum_steps: int = 128,
    junction_margin: float = 0.16,
) -> tuple[list[tuple[int, int]], int]:
    output = list(route)
    used = set(output)
    extension_start = output[-1]
    route_direction = _direction(output)
    added = 0
    for _ in range(maximum_steps):
        direction = _direction(output)
        candidates = [
            point for point in _neighbours(output[-1], skeleton) if point not in used
        ]
        if not candidates:
            break
        ranked = sorted(
            (
                direction[0]
                * _unit((point[0] - output[-1][0], point[1] - output[-1][1]))[0]
                + direction[1]
                * _unit((point[0] - output[-1][0], point[1] - output[-1][1]))[1],
                point,
            )
            for point in candidates
        )
        if preserve_axis and added >= 2:
            ranked = [
                item
                for item in ranked
                if (
                    route_direction[0]
                    * _unit(
                        (
                            item[1][0] - extension_start[0],
                            item[1][1] - extension_start[1],
                        )
                    )[0]
                    + route_direction[1]
                    * _unit(
                        (
                            item[1][0] - extension_start[0],
                            item[1][1] - extension_start[1],
                        )
                    )[1]
                    >= 0.9
                )
            ]
            if not ranked:
                break
        best_score, best = ranked[-1]
        if len(ranked) > 1:
            runner_up = ranked[-2][0]
            if best_score < 0.2 or best_score - runner_up < junction_margin:
                break
        output.append(best)
        used.add(best)
        added += 1
    return output, added


def extend_directed_route(
    route: list[list[float]],
    skeleton: set[tuple[int, int]],
    feature: str | None = None,
) -> tuple[list[list[int]], dict]:
    points = [(round(point[0]), round(point[1])) for point in route]
    preserve_axis = feature in {"横", "竖", "提"}
    forward, end_added = _advance(points, skeleton, preserve_axis=preserve_axis)
    reversed_route, start_added = _advance(
        list(reversed(forward)), skeleton, preserve_axis=preserve_axis
    )
    completed = list(reversed(reversed_route))
    return (
        [[point[0], point[1]] for point in completed],
        {"startAdded": start_added, "endAdded": end_added},
    )


def _endpoint_error(route, truth, size: int) -> tuple[float, float]:
    scale = size / 100
    truth_start = (truth[0][0] * scale, truth[0][1] * scale)
    truth_end = (truth[-1][0] * scale, truth[-1][1] * scale)
    diagonal = size * math.sqrt(2)
    return (
        math.dist(route[0], truth_start) / diagonal,
        math.dist(route[-1], truth_end) / diagonal,
    )


def complete_historic_routes(payload: dict) -> tuple[list[list[list[int]]], list[dict]]:
    skeleton = {(round(point[0]), round(point[1])) for point in payload["skeleton"]}
    completed = []
    evidence = []
    for route, stroke in zip(payload["routes"], payload["strokes"]):
        result, item = extend_directed_route(route, skeleton, stroke["feature"])
        completed.append(result)
        evidence.append(item)
    return completed, evidence


def _nearest_skeleton_point(point, skeleton):
    return min(
        skeleton,
        key=lambda candidate: (
            (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2
        ),
    )


def _shortest_skeleton_path(start, end, skeleton):
    if start == end:
        return [start]
    queue = [(0.0, start)]
    costs = {start: 0.0}
    previous = {}
    while queue:
        cost, point = heapq.heappop(queue)
        if cost != costs.get(point):
            continue
        if point == end:
            break
        for neighbour in _neighbours(point, skeleton):
            candidate = cost + math.dist(point, neighbour)
            if candidate + 1e-9 < costs.get(neighbour, math.inf):
                costs[neighbour] = candidate
                previous[neighbour] = point
                heapq.heappush(queue, (candidate, neighbour))
    if end not in costs:
        raise ValueError(f"no skeleton path between {start} and {end}")
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return list(reversed(path))


def teacher_forced_routes(payload: dict) -> list[list[list[int]]]:
    """Normalize human trajectories onto the PDF skeleton.

    This intentionally reads truth and is never an inference method. It turns
    hand-drawn paths into an exact continuous target for later blind solvers.
    """
    scale = int(payload["size"]) / 100
    skeleton = {(round(point[0]), round(point[1])) for point in payload["skeleton"]}
    output = []
    for truth in payload["truth"]:
        snapped = []
        for point in truth:
            target = point[0] * scale, point[1] * scale
            nearest = _nearest_skeleton_point(target, skeleton)
            if not snapped or snapped[-1] != nearest:
                snapped.append(nearest)
        route = [snapped[0]]
        for end in snapped[1:]:
            segment = _shortest_skeleton_path(route[-1], end, skeleton)
            route.extend(segment[1:])
        output.append([[point[0], point[1]] for point in route])
    return output


def _polyline(points, css_class: str, scale: float = 1.0) -> str:
    value = " ".join(
        f"{point[0] * scale:.2f},{point[1] * scale:.2f}" for point in points
    )
    return f'<polyline class="{css_class}" points="{value}"/>'


def _markers(points, prefix: str, scale: float = 1.0) -> str:
    start = points[0][0] * scale, points[0][1] * scale
    end = points[-1][0] * scale, points[-1][1] * scale
    return (
        f'<circle class="{prefix}-start" cx="{start[0]:.2f}" cy="{start[1]:.2f}" r="2.2"/>'
        f'<rect class="{prefix}-end" x="{end[0] - 2.0:.2f}" y="{end[1] - 2.0:.2f}" width="4" height="4"/>'
    )


def render_trajectory_review(payload: dict, completed, evidence) -> str:
    size = int(payload["size"])
    scale = size / 100
    skeleton = " ".join(f"M{point[0]},{point[1]}h.1" for point in payload["skeleton"])
    old = "".join(
        _polyline(route, "old") + _markers(route, "old") for route in payload["routes"]
    )
    fitted = "".join(
        _polyline(route, "fitted") + _markers(route, "fitted") for route in completed
    )
    truth = "".join(
        _polyline(route, "truth", scale) + _markers(route, "truth", scale)
        for route in payload["truth"]
    )
    teacher_routes = teacher_forced_routes(payload)
    teacher = "".join(
        _polyline(route, "teacher") + _markers(route, "teacher")
        for route in teacher_routes
    )
    old_errors = [
        _endpoint_error(route, target, size)
        for route, target in zip(payload["routes"], payload["truth"])
    ]
    fitted_errors = [
        _endpoint_error(route, target, size)
        for route, target in zip(completed, payload["truth"])
    ]
    teacher_errors = [
        _endpoint_error(route, target, size)
        for route, target in zip(teacher_routes, payload["truth"])
    ]
    rows = "".join(
        "<tr>"
        f"<td>{index + 1}</td><td>{html.escape(stroke['feature'])}</td>"
        f"<td>{old_error[0]:.2%}</td><td>{new_error[0]:.2%}</td>"
        f"<td>{old_error[1]:.2%}</td><td>{new_error[1]:.2%}</td>"
        f"<td>{teacher_error[0]:.2%}</td><td>{teacher_error[1]:.2%}</td>"
        f"<td>+{proof['startAdded']} / +{proof['endAdded']}</td>"
        "</tr>"
        for index, (stroke, old_error, new_error, teacher_error, proof) in enumerate(
            zip(
                payload["strokes"],
                old_errors,
                fitted_errors,
                teacher_errors,
                evidence,
            )
        )
    )
    cards = "".join(
        f'<section class="stroke"><h3>第 {index + 1} 笔 · {html.escape(stroke["feature"])}</h3>'
        f'<svg viewBox="0 0 {size} {size}"><path class="skeleton" d="{skeleton}"/>'
        f"{_polyline(teacher_routes[index], 'teacher')}{_markers(teacher_routes[index], 'teacher')}"
        f"{_polyline(payload['truth'][index], 'truth', scale)}"
        f"{_markers(payload['truth'][index], 'truth', scale)}</svg></section>"
        for index, stroke in enumerate(payload["strokes"])
    )
    return f"""<!doctype html><meta charset="utf-8"><title>U+6418 骨架惯性</title>
<style>
body{{font:15px system-ui;margin:20px;background:#f3f5f8;color:#172033}}h1{{margin:0 0 8px}}
.note{{max-width:1200px;line-height:1.6}}.overview{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.strokes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
section{{background:white;border:1px solid #ccd5e2;border-radius:10px;padding:12px}}svg{{width:100%;aspect-ratio:1}}
.skeleton{{stroke:#d4d9e2;stroke-width:1.4;fill:none;stroke-linecap:round}}
.old{{stroke:#777;stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.fitted{{stroke:#111827;stroke-width:2.4;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.teacher{{stroke:#111827;stroke-width:2.6;fill:none;stroke-linecap:round;stroke-linejoin:round}}
.truth{{stroke:#06b6d4;stroke-width:1.8;fill:none;stroke-dasharray:5 3;stroke-linecap:round;stroke-linejoin:round}}
.old-start,.fitted-start{{fill:#ef4444}}.old-end,.fitted-end{{fill:#2563eb}}.truth-start{{fill:#22d3ee}}.truth-end{{fill:#0891b2}}
.teacher-start{{fill:#ef4444}}.teacher-end{{fill:#2563eb}}
table{{border-collapse:collapse;background:white;margin:16px 0;width:min(100%,1000px)}}th,td{{border:1px solid #d6deea;padding:7px;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}
.stroke h3{{margin:0 0 5px;font-size:14px}}
</style><h1>U+6418-G：有向骨架惯性（暂不上色）</h1>
<p class="note">红圆＝起笔，蓝方＝收笔；青圆/青方与青色虚线＝原始人工有向笔画。灰色为 PDF 骨架。第二栏是不读取人工线的盲推惯性实验；第三栏明确读取人工路标，只把标注规范化成机器可学习的连续 PDF 骨架真值，不能冒充推断结果。</p>
<div class="overview">
<section><h2>旧模型骨架</h2><svg viewBox="0 0 {size} {size}"><path class="skeleton" d="{skeleton}"/>{old}</svg></section>
<section><h2>端点惯性续行</h2><svg viewBox="0 0 {size} {size}"><path class="skeleton" d="{skeleton}"/>{fitted}</svg></section>
<section><h2>教师强制 PDF 骨架</h2><svg viewBox="0 0 {size} {size}"><path class="skeleton" d="{skeleton}"/>{teacher}</svg></section>
<section><h2>人工骨架</h2><svg viewBox="0 0 {size} {size}"><path class="skeleton" d="{skeleton}"/>{truth}</svg></section>
</div><table><thead><tr><th>笔</th><th>笔形</th><th>旧起误差</th><th>盲推起误差</th><th>旧收误差</th><th>盲推收误差</th><th>教师起误差</th><th>教师收误差</th><th>向前补点</th></tr></thead><tbody>{rows}</tbody></table>
<div class="strokes">{cards}</div>"""
