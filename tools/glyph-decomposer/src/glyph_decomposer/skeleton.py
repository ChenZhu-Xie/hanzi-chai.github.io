"""Deterministic PDF-ink skeleton generation and truth-isolated evaluation."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from shapely import contains_xy

from .geometry import normalize_geometry
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells


@dataclass(frozen=True)
class SkeletonCase:
    unicode: str
    source: str
    character: str
    skeleton: np.ndarray
    truth_paths: tuple[np.ndarray, ...]
    metrics: dict


def rasterize_geometry(geometry, size: int = 256) -> np.ndarray:
    """Sample normalized vector ink at pixel centres without annotation data."""
    coordinates = (np.arange(size, dtype=float) + 0.5) * (100.0 / size)
    x, y = np.meshgrid(coordinates, coordinates)
    return np.asarray(contains_xy(geometry, x, y), dtype=bool)


def zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Connectivity-preserving deterministic Zhang-Suen thinning."""
    image = np.asarray(binary, dtype=bool).copy()
    while True:
        changed = False
        for first_pass in (True, False):
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(neighbours)
            transitions = sum(
                (~left) & right
                for left, right in zip(neighbours, neighbours[1:] + neighbours[:1])
            )
            if first_pass:
                preserve1 = ~(p2 & p4 & p6)
                preserve2 = ~(p4 & p6 & p8)
            else:
                preserve1 = ~(p2 & p4 & p8)
                preserve2 = ~(p2 & p6 & p8)
            remove = (
                image
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & preserve1
                & preserve2
            )
            if remove.any():
                image[remove] = False
                changed = True
        if not changed:
            return image


def generate_skeleton(
    geometry, size: int = 256, *, padding: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    # This is the repository's historic annotation-canvas convention, not a
    # value fitted from truth. Keeping one public coordinate system lets old
    # and future reviews compare like with like.
    canonical = normalize_geometry(geometry, margin=padding * 100)
    ink = rasterize_geometry(canonical, size)
    if not ink.any():
        raise ValueError("PDF vector geometry rasterized to an empty mask")
    return ink, zhang_suen(ink)


def _densify(points: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    output = [points[0]]
    for start, end in pairwise(points):
        length = float(np.linalg.norm(end - start))
        steps = max(1, math.ceil(length / spacing))
        output.extend(
            start + (end - start) * (index / steps) for index in range(1, steps + 1)
        )
    return np.asarray(output)


def load_truth_paths(path: Path) -> tuple[dict, tuple[np.ndarray, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    paths = []
    for annotation in payload.get("annotations", ()):  # lasso is component truth
        if annotation.get("type") not in {"line", "polyline", "bezier", "pen"}:
            continue
        points = annotation.get("fixedPoints") or annotation.get("points") or ()
        if len(points) >= 2:
            paths.append(_densify(np.asarray(points, dtype=float)))
    return metadata, tuple(paths)


def _nearest_distances(first: np.ndarray, second: np.ndarray, chunk: int = 512):
    output = []
    for start in range(0, len(first), chunk):
        delta = first[start : start + chunk, None, :] - second[None, :, :]
        output.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return np.concatenate(output)


def evaluate_skeleton(
    skeleton: np.ndarray, truth_paths: tuple[np.ndarray, ...], *, tolerance: float = 1.5
) -> dict:
    if not truth_paths:
        raise ValueError("annotation contains no directed stroke paths")
    size = skeleton.shape[0]
    skeleton_yx = np.argwhere(skeleton)
    skeleton_xy = skeleton_yx[:, ::-1] * (100.0 / size)
    truth = np.concatenate([_densify(path) for path in truth_paths])
    truth_to_skeleton = _nearest_distances(truth, skeleton_xy)
    skeleton_to_truth = _nearest_distances(skeleton_xy, truth)
    endpoint_distances = _nearest_distances(
        np.asarray([point for path in truth_paths for point in (path[0], path[-1])]),
        skeleton_xy,
    )
    return {
        "strokeCount": len(truth_paths),
        "skeletonPixels": int(skeleton.sum()),
        "truthMeanDistance": float(truth_to_skeleton.mean()),
        "truthP95Distance": float(np.percentile(truth_to_skeleton, 95)),
        "truthCoverage": float(np.mean(truth_to_skeleton <= tolerance)),
        "skeletonMeanDistance": float(skeleton_to_truth.mean()),
        "skeletonP95Distance": float(np.percentile(skeleton_to_truth, 95)),
        "skeletonCoverage": float(np.mean(skeleton_to_truth <= tolerance)),
        "endpointMeanDistance": float(endpoint_distances.mean()),
        "endpointP95Distance": float(np.percentile(endpoint_distances, 95)),
        "truthReadAfterGeneration": True,
    }


def audit_annotations(
    pdf_path: Path,
    bbox_path: Path,
    annotation_paths: list[Path],
    *,
    size: int = 256,
) -> list[SkeletonCase]:
    cells = parse_cells(bbox_path)
    pages: dict[int, str] = {}
    output = []
    for annotation_path in annotation_paths:
        metadata, truth_paths = load_truth_paths(annotation_path)
        if not metadata.get("unicode") or not metadata.get("source"):
            continue
        codepoint = int(metadata["unicode"].removeprefix("U+"), 16)
        source = str(metadata["source"])
        cell = find_cell(cells, codepoint, source)
        if cell.page not in pages:
            pages[cell.page] = export_page_svg(pdf_path, cell.page)
        geometry = extract_cell_geometry(pages[cell.page], cell)
        _ink, skeleton = generate_skeleton(geometry, size)
        metrics = evaluate_skeleton(skeleton, truth_paths)
        output.append(
            SkeletonCase(
                unicode=f"U+{codepoint:04X}",
                source=source,
                character=chr(codepoint),
                skeleton=skeleton,
                truth_paths=truth_paths,
                metrics=metrics,
            )
        )
    return output


def _skeleton_path(skeleton: np.ndarray) -> str:
    scale = 100 / skeleton.shape[0]
    return "".join(
        f"M{x * scale:.3f},{y * scale:.3f}h{scale:.3f}v{scale:.3f}h-{scale:.3f}z"
        for y, x in np.argwhere(skeleton)
    )


def _truth_paths(paths: tuple[np.ndarray, ...]) -> str:
    return "".join(
        '<polyline points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in path) + '"/>'
        for path in paths
    )


def render_skeleton_audit(cases: list[SkeletonCase]) -> str:
    cards = "".join(
        f"""<section><h2>{case.unicode} {html.escape(case.character)} · {case.source}</h2>
<svg viewBox="0 0 100 100"><path class="skeleton" d="{_skeleton_path(case.skeleton)}"/>
<g class="truth">{_truth_paths(case.truth_paths)}</g></svg>
<p>人工路径→骨架：mean {case.metrics["truthMeanDistance"]:.2f} / p95 {case.metrics["truthP95Distance"]:.2f}；覆盖 {case.metrics["truthCoverage"]:.2%}</p>
<p>骨架→人工路径：mean {case.metrics["skeletonMeanDistance"]:.2f} / p95 {case.metrics["skeletonP95Distance"]:.2f}；覆盖 {case.metrics["skeletonCoverage"]:.2%}</p>
<p>起收笔→骨架：mean {case.metrics["endpointMeanDistance"]:.2f} / p95 {case.metrics["endpointP95Distance"]:.2f}</p></section>"""
        for case in cases
    )
    summary = {
        "caseCount": len(cases),
        "strokeCount": sum(case.metrics["strokeCount"] for case in cases),
        "meanTruthCoverage": float(
            np.mean([case.metrics["truthCoverage"] for case in cases])
        ),
        "meanSkeletonCoverage": float(
            np.mean([case.metrics["skeletonCoverage"] for case in cases])
        ),
        "meanEndpointDistance": float(
            np.mean([case.metrics["endpointMeanDistance"] for case in cases])
        ),
        "truthIsolation": "annotations loaded only after each PDF skeleton is frozen",
    }
    return f"""<!doctype html><meta charset="utf-8"><title>确定性 PDF 骨架回归</title>
<style>body{{font:15px system-ui;margin:20px;background:#f3f5f8;color:#172033}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}section{{background:#fff;border:1px solid #ccd5e2;border-radius:10px;padding:12px}}svg{{width:100%;aspect-ratio:1;background:white}}.skeleton{{fill:#111827}}.truth{{fill:none;stroke:#06b6d4;stroke-width:.55;stroke-dasharray:1.5 1;stroke-linecap:round;stroke-linejoin:round}}pre{{background:#111827;color:#e5edf8;padding:12px;border-radius:8px}}</style>
<h1>确定性 PDF 无向骨架生成器</h1><p>黑色＝仅由 PDF 矢量墨迹生成的 Zhang–Suen 骨架；青色虚线＝骨架冻结后才载入的人工有向笔画。人工标注不参与栅格化、细化或参数选择。</p>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre><div class="grid">{cards}</div>"""
