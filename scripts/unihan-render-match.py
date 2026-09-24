"""Evaluate repository glyph candidates against rasterized Unicode chart cells."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_pdf_module():
    path = Path(__file__).with_name("unihan-pdf-evidence.py")
    spec = importlib.util.spec_from_file_location("unihan_pdf_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PDF = load_pdf_module()


def load_pdf_page_svg(pdf_path: Path, page_number: int) -> str:
    """Export one PDF page as vector outlines without reusing its font assets."""
    executable = shutil.which("pdftocairo")
    if executable is None:
        raise RuntimeError("vector PDF evidence requires pdftocairo on PATH")
    completed = subprocess.run(
        [
            executable,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-svg",
            str(pdf_path),
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8")


def crop_page_svg(page_svg: str, bbox: list[float], margin=1.5) -> str:
    """Crop a page SVG by viewBox while retaining its vector glyph outlines."""
    left, top, right, bottom = bbox
    left -= margin
    top -= margin
    right += margin
    bottom += margin
    width = right - left
    height = bottom - top
    side = max(width, height)
    left -= (side - width) / 2
    top -= (side - height) / 2

    match = re.search(r"<svg\b[^>]*>", page_svg)
    if match is None:
        raise ValueError("page SVG has no root element")
    root = match.group(0)

    def attribute(markup: str, name: str, value: str) -> str:
        pattern = rf'\s{name}="[^"]*"'
        replacement = f' {name}="{value}"'
        if re.search(pattern, markup):
            return re.sub(pattern, replacement, markup, count=1)
        return markup[:-1] + replacement + ">"

    root = attribute(root, "width", f"{side:.4f}pt")
    root = attribute(root, "height", f"{side:.4f}pt")
    root = attribute(root, "viewBox", f"{left:.4f} {top:.4f} {side:.4f} {side:.4f}")
    return page_svg[: match.start()] + root + page_svg[match.end() :]


def chart_glyph_bbox(bbox: list[float], glyph_fraction=0.78) -> list[float]:
    """Remove the small IRG source-reference text printed below each glyph."""
    left, top, right, bottom = bbox
    return [left, top, right, top + (bottom - top) * glyph_fraction]


def render_pdf_vector_cell(
    page_svg: str, bbox: list[float], size=1024
) -> np.ndarray:
    """Render only one chart cell, preserving vector detail until the last step."""
    cropped = crop_page_svg(page_svg, bbox)
    match = re.search(r"<svg\b[^>]*>", cropped)
    if match is None:
        raise ValueError("cropped page SVG has no root element")
    root = match.group(0)
    root = re.sub(r'\swidth="[^"]*"', f' width="{size}px"', root, count=1)
    root = re.sub(r'\sheight="[^"]*"', f' height="{size}px"', root, count=1)
    cropped = cropped[: match.start()] + root + cropped[match.end() :]
    return render_svg_image(cropped, size=size)


def pdf_vector_skeleton(
    page_svg: str, bbox: list[float], output_size=128
) -> np.ndarray:
    evidence_size = max(512, output_size * 8)
    return normalize_skeleton(
        render_pdf_vector_cell(
            page_svg, chart_glyph_bbox(bbox), evidence_size
        ) < 224,
        output_size,
    )


def sample_cubic(start, parameters, steps=24):
    x0, y0 = start
    x1, y1, x2, y2, x3, y3 = parameters
    control1 = np.array([x0 + x1, y0 + y1], dtype=float)
    control2 = np.array([x0 + x2, y0 + y2], dtype=float)
    end = np.array([x0 + x3, y0 + y3], dtype=float)
    begin = np.array(start, dtype=float)
    values = []
    for value in np.linspace(0, 1, steps):
        point = (
            (1 - value) ** 3 * begin
            + 3 * (1 - value) ** 2 * value * control1
            + 3 * (1 - value) * value**2 * control2
            + value**3 * end
        )
        values.append(point)
    return values, end


def render_raw_strokes(strokes: list[dict], size=128) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = (size - 20) / 100
    offset = 10
    for stroke in strokes:
        current = np.array(stroke["start"], dtype=float)
        points = [current.copy()]
        for curve in stroke["curveList"]:
            command = curve["command"]
            parameters = curve["parameterList"]
            if command == "h":
                current = current + np.array([parameters[0], 0])
                points.append(current.copy())
            elif command == "v":
                current = current + np.array([0, parameters[0]])
                points.append(current.copy())
            elif command in ("c", "z"):
                sampled, current = sample_cubic(current, parameters)
                points.extend(sampled[1:])
            elif command == "a":
                angles = np.linspace(0, 2 * math.pi, 48)
                points.extend(
                    current + np.column_stack((np.cos(angles), np.sin(angles))) * 50
                )
        pixel_points = np.array(
            [
                [round(point[0] * scale + offset), round(point[1] * scale + offset)]
                for point in points
            ],
            dtype=np.int32,
        )
        if len(pixel_points) > 1:
            cv2.polylines(
                canvas,
                [pixel_points],
                False,
                255,
                thickness=5,
                lineType=cv2.LINE_AA,
            )
    return canvas


def render_strokes(strokes: list[dict], size=128, output_size=64) -> np.ndarray:
    return normalize_skeleton(render_raw_strokes(strokes, size) > 24, output_size)


def render_stroke_layers(
    strokes: list[dict], size=128, output_size=64
) -> list[np.ndarray]:
    layers = [render_raw_strokes([stroke], size) > 24 for stroke in strokes]
    if not layers:
        return []
    union = np.logical_or.reduce(layers)
    points = np.argwhere(union)
    if points.size == 0:
        return [np.zeros((output_size, output_size), dtype=bool) for _ in layers]
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    target = output_size - max(10, output_size // 8)
    scale = min(target / (right - left), target / (bottom - top))
    width = max(1, round((right - left) * scale))
    height = max(1, round((bottom - top) * scale))
    normalized = []
    for layer in layers:
        ink = layer[top:bottom, left:right].astype(np.uint8) * 255
        resized = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((output_size, output_size), dtype=bool)
        y = (output_size - height) // 2
        x = (output_size - width) // 2
        canvas[y : y + height, x : x + width] = resized > 24
        normalized.append(skeletonize(canvas))
    return normalized


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning that preserves connectivity of filled outlines."""
    image = binary.astype(bool).copy()
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


def graph_topology_signature(binary: np.ndarray) -> dict[str, int]:
    """Count robust graph invariants: components, ends, and branch/cross nodes."""
    skeleton = skeletonize(binary)
    component_count, _labels = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    padded = np.pad(skeleton, 1)
    ring = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    transitions = sum(
        (~ring[index]) & ring[(index + 1) % len(ring)]
        for index in range(len(ring))
    )

    def cluster_count(mask: np.ndarray, dilate=False) -> int:
        if dilate:
            mask = cv2.dilate(
                mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return max(0, int(count) - 1)

    return {
        "components": max(0, int(component_count) - 1),
        "endpoints": cluster_count(skeleton & (transitions == 1)),
        "junctions": cluster_count(skeleton & (transitions >= 3), dilate=True),
    }


def topology_signature_choice(
    target: dict[str, int], candidates: list[dict[str, int]]
) -> dict | None:
    """Choose only when every independent graph invariant names one candidate."""
    if len(candidates) < 2:
        return None
    keys = ("components", "endpoints", "junctions")
    winners = []
    per_metric = {}
    for key in keys:
        distances = [abs(candidate[key] - target[key]) for candidate in candidates]
        ordered = sorted((value, index) for index, value in enumerate(distances))
        if len(ordered) < 2 or ordered[0][0] == ordered[1][0]:
            return None
        winners.append(ordered[0][1])
        per_metric[key] = distances
    if len(set(winners)) != 1:
        return None
    weights = {"components": 2.0, "endpoints": 0.5, "junctions": 1.0}
    scores = [
        sum(abs(candidate[key] - target[key]) * weights[key] for key in keys)
        for candidate in candidates
    ]
    ordered_scores = sorted((value, index) for index, value in enumerate(scores))
    return {
        "candidateIndex": winners[0],
        "scores": [round(value, 3) for value in scores],
        "margin": round(ordered_scores[1][0] - ordered_scores[0][0], 3),
        "perMetricDistances": per_metric,
    }


def normalize_skeleton(binary: np.ndarray, output_size=64) -> np.ndarray:
    points = np.argwhere(binary)
    if points.size == 0:
        return np.zeros((output_size, output_size), dtype=bool)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    ink = binary[top:bottom, left:right].astype(np.uint8) * 255
    target = output_size - max(10, output_size // 8)
    scale = min(target / ink.shape[1], target / ink.shape[0])
    resized = cv2.resize(
        ink,
        (
            max(1, round(ink.shape[1] * scale)),
            max(1, round(ink.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    normalized = np.zeros((output_size, output_size), dtype=bool)
    y = (output_size - resized.shape[0]) // 2
    x = (output_size - resized.shape[1]) // 2
    normalized[y : y + resized.shape[0], x : x + resized.shape[1]] = resized > 24
    return skeletonize(normalized)


def render_svg_image(svg: str, size=256) -> np.ndarray:
    executable = shutil.which("magick")
    if executable is None:
        raise RuntimeError(
            "SVG evidence requires ImageMagick's `magick` executable on PATH"
        )
    completed = subprocess.run(
        [
            executable,
            "-background",
            "white",
            "svg:-",
            "-resize",
            f"{size}x{size}!",
            "-colorspace",
            "Gray",
            "png:-",
        ],
        input=svg.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    image = cv2.imdecode(
        np.frombuffer(completed.stdout, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise RuntimeError("ImageMagick returned an unreadable SVG raster")
    return image


def render_svg(svg: str, size=256, output_size=64) -> np.ndarray:
    return normalize_skeleton(render_svg_image(svg, size) < 224, output_size)


def discriminative_distance(
    pdf: np.ndarray, candidates: list[np.ndarray]
) -> list[float]:
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = [
        cv2.dilate(candidate.astype(np.uint8), kernel) > 0 for candidate in candidates
    ]
    common = np.logical_and.reduce(dilated)
    union = np.logical_or.reduce(candidates)
    difference = union & ~common
    roi = cv2.dilate(difference.astype(np.uint8), np.ones((11, 11), dtype=np.uint8)) > 0
    if not roi.any():
        return [PDF.chamfer_distance(pdf, candidate) for candidate in candidates]
    pdf_roi = pdf & roi
    return [PDF.chamfer_distance(pdf_roi, candidate & roi) for candidate in candidates]


def _residual_components(mask: np.ndarray, minimum_pixels=2) -> list[dict]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    return [
        {
            "pixels": int(stats[label, cv2.CC_STAT_AREA]),
            "bbox": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ],
            "center": [round(float(value), 2) for value in centroids[label]],
        }
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= minimum_pixels
    ]


def _strong_corners(mask: np.ndarray) -> list[tuple[int, int]]:
    detected = cv2.goodFeaturesToTrack(
        mask.astype(np.uint8) * 255,
        maxCorners=64,
        qualityLevel=0.08,
        minDistance=5,
        blockSize=5,
        useHarrisDetector=False,
    )
    if detected is None:
        return []
    return [
        (int(round(float(point[0]))), int(round(float(point[1]))))
        for point in detected.reshape(-1, 2)
    ]


def _cardinal_branches(mask: np.ndarray, x: int, y: int, radius=6) -> list[str]:
    height, width = mask.shape

    def count(left, top, right, bottom):
        return int(
            mask[
                max(0, top) : min(height, bottom),
                max(0, left) : min(width, right),
            ].sum()
        )

    bands = {
        "left": count(x - radius, y - 2, x - 1, y + 3),
        "right": count(x + 2, y - 2, x + radius + 1, y + 3),
        "up": count(x - 2, y - radius, x + 3, y - 1),
        "down": count(x - 2, y + 2, x + 3, y + radius + 1),
    }
    return sorted(direction for direction, pixels in bands.items() if pixels >= 3)


def _directional_corners(mask: np.ndarray) -> list[dict]:
    opposite = ({"left", "right"}, {"up", "down"})
    by_branches: dict[tuple[str, ...], np.ndarray] = {}
    for y, x in np.argwhere(mask):
        branches = tuple(_cardinal_branches(mask, int(x), int(y)))
        if len(branches) != 2 or set(branches) in opposite:
            continue
        by_branches.setdefault(branches, np.zeros_like(mask, dtype=np.uint8))[y, x] = 1
    corners = []
    for branches, pixels in by_branches.items():
        # Consecutive skeleton pixels describe one geometric turn.  Clustering
        # them avoids making detector sampling density part of the evidence.
        clustered = cv2.dilate(pixels, np.ones((3, 3), np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            clustered, 8
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            x, y = centroids[label]
            corners.append(
                {
                    "point": [int(round(float(x))), int(round(float(y)))],
                    "branches": list(branches),
                }
            )
    return corners


def directional_corner_mismatches(pdf: np.ndarray, candidate: np.ndarray) -> list[dict]:
    """Find a nearby sharp turn whose outgoing side is reversed in the PDF."""
    pdf_corners = _directional_corners(pdf)
    mismatches = []
    for candidate_corner in _directional_corners(candidate):
        candidate_x, candidate_y = candidate_corner["point"]
        candidate_branches = candidate_corner["branches"]
        nearby = sorted(
            (
                math.hypot(candidate_x - pdf_x, candidate_y - pdf_y),
                pdf_x,
                pdf_y,
                pdf_corner["branches"],
            )
            for pdf_corner in pdf_corners
            for pdf_x, pdf_y in [pdf_corner["point"]]
            if math.hypot(candidate_x - pdf_x, candidate_y - pdf_y) <= 5
        )
        if not nearby:
            continue
        separation, pdf_x, pdf_y, pdf_branches = nearby[0]
        if pdf_branches == candidate_branches:
            continue
        shared = set(pdf_branches) & set(candidate_branches)
        changed = set(pdf_branches) ^ set(candidate_branches)
        # Only promote the topology invariant established by the 骨 review:
        # the vertical continuation stays the same while a horizontal branch
        # moves from one side to the other.  Treating every nearby turn-angle
        # change as structural produces many font-style false positives.
        if shared not in ({"up"}, {"down"}) or changed != {"left", "right"}:
            continue
        mismatches.append(
            {
                "candidatePoint": [candidate_x, candidate_y],
                "candidateBranches": candidate_branches,
                "pdfPoint": [pdf_x, pdf_y],
                "pdfBranches": pdf_branches,
                "separation": round(separation, 3),
            }
        )
    return mismatches


def candidate_consensus(candidates: list[np.ndarray], tolerance=2.0) -> np.ndarray:
    """Keep base-candidate ink that every candidate has within raster tolerance."""
    consensus = candidates[0].copy()
    for candidate in candidates[1:]:
        distance = PDF.distance_transform_edt(~candidate)
        consensus &= distance <= tolerance
    return consensus


def residual_supports_side_reversal(
    candidate_only: np.ndarray, pdf_only: np.ndarray, mismatch: dict
) -> dict | None:
    """Require unmatched ink on both claimed sides of a reversed branch."""

    def side_pixels(mask: np.ndarray, point: list[int], side: str) -> int:
        x, y = point
        height, width = mask.shape
        if side == "left":
            left, right = x - 13, x - 2
        else:
            left, right = x + 2, x + 13
        return int(
            mask[
                max(0, y - 2) : min(height, y + 3),
                max(0, left) : min(width, right),
            ].sum()
        )

    candidate_side = next(
        side for side in ("left", "right") if side in mismatch["candidateBranches"]
    )
    pdf_side = next(
        side for side in ("left", "right") if side in mismatch["pdfBranches"]
    )
    candidate_pixels = side_pixels(
        candidate_only, mismatch["candidatePoint"], candidate_side
    )
    pdf_pixels = side_pixels(pdf_only, mismatch["pdfPoint"], pdf_side)
    if candidate_pixels < 4 or pdf_pixels < 4:
        return None
    return {
        **mismatch,
        "candidateOnlyBranchPixels": candidate_pixels,
        "pdfOnlyBranchPixels": pdf_pixels,
    }


def shared_structure_residual(
    pdf: np.ndarray,
    candidates: list[np.ndarray],
    distance_tolerance=2.0,
) -> dict:
    """Detect a structured mismatch shared by every supplied candidate.

    A common error cancels from candidate ranking. Nearby candidate-only and
    PDF-only ink instead signals a relocated branch and a possibly incomplete
    sibling set. This evidence is review-only and never enables a write.
    """
    if not candidates:
        return {"suspected": False, "reason": "no-candidates"}
    # Exact pixel intersection loses genuinely shared parts after tiny layout or
    # rasterization shifts.  A tolerant consensus preserves the shared subtree
    # whose wrong geometry candidate ranking would otherwise never question.
    common = candidate_consensus(candidates)
    union = np.logical_or.reduce(candidates)
    if not common.any():
        return {"suspected": False, "reason": "no-common-ink"}
    varying = union & ~common
    candidate_distance = PDF.distance_transform_edt(~common)
    pdf_distance = PDF.distance_transform_edt(~pdf)
    shared_zone = cv2.dilate(
        common.astype(np.uint8), np.ones((13, 13), np.uint8)
    ).astype(bool)
    if varying.any():
        shared_zone &= ~cv2.dilate(
            varying.astype(np.uint8), np.ones((7, 7), np.uint8)
        ).astype(bool)
    candidate_only = common & (pdf_distance > distance_tolerance)
    pdf_only = pdf & shared_zone & (candidate_distance > distance_tolerance)
    candidate_parts = _residual_components(candidate_only)
    pdf_parts = _residual_components(pdf_only)
    corner_mismatches = [
        supported
        for mismatch in directional_corner_mismatches(pdf, common)
        if (
            supported := residual_supports_side_reversal(
                candidate_only, pdf_only, mismatch
            )
        )
        is not None
    ]
    pairs = []
    for candidate_part in candidate_parts:
        for pdf_part in pdf_parts:
            separation = math.dist(candidate_part["center"], pdf_part["center"])
            if separation > 14:
                continue
            combined_pixels = candidate_part["pixels"] + pdf_part["pixels"]
            if combined_pixels < 6:
                continue
            pairs.append(
                {
                    "candidateOnly": candidate_part,
                    "pdfOnly": pdf_part,
                    "separation": round(separation, 3),
                    "combinedPixels": combined_pixels,
                }
            )
    pairs.sort(key=lambda item: (-item["combinedPixels"], item["separation"]))
    common_pixels = int(common.sum())
    residual_ratio = (int(candidate_only.sum()) + int(pdf_only.sum())) / common_pixels
    suspected = bool(corner_mismatches)
    return {
        "suspected": suspected,
        "reason": (
            "directional-corner-mismatch" if suspected else "no-stable-paired-residual"
        ),
        "commonPixels": common_pixels,
        "candidateOnlyPixels": int(candidate_only.sum()),
        "pdfOnlyPixels": int(pdf_only.sum()),
        "residualRatio": round(residual_ratio, 4),
        "candidateOnlyComponents": candidate_parts[:8],
        "pdfOnlyComponents": pdf_parts[:8],
        "pairs": pairs[:8],
        "candidateDirectionalCorners": _directional_corners(common),
        "pdfDirectionalCorners": _directional_corners(pdf),
        "directionalCornerMismatches": corner_mismatches,
    }


def warp_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    return (
        cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        > 0
    )


def align_candidates_to_pdf(
    pdf: np.ndarray, candidates: list[np.ndarray]
) -> tuple[list[np.ndarray], dict]:
    """Align candidate-common ink first so sibling differences cannot steer registration."""
    common = np.logical_and.reduce(candidates)
    points = np.argwhere(common)
    if points.size == 0:
        return candidates, {"scaleX": 1.0, "scaleY": 1.0, "dx": 0, "dy": 0}
    target_distance = PDF.distance_transform_edt(~pdf)
    center_y = (common.shape[0] - 1) / 2
    center_x = (common.shape[1] - 1) / 2

    def score(scale_x, scale_y, dx, dy):
        ys = np.rint((points[:, 0] - center_y) * scale_y + center_y + dy).astype(int)
        xs = np.rint((points[:, 1] - center_x) * scale_x + center_x + dx).astype(int)
        inside = (ys >= 0) & (ys < pdf.shape[0]) & (xs >= 0) & (xs < pdf.shape[1])
        if inside.mean() < 0.98:
            return math.inf
        return float(target_distance[ys[inside], xs[inside]].mean())

    best = (math.inf, 1.0, 1.0, 0, 0)
    translation = max(6, round(pdf.shape[1] * 0.095))
    coarse_step = max(2, round(pdf.shape[1] / 32))
    for scale_x in (0.9, 1.0, 1.1):
        for scale_y in (0.9, 1.0, 1.1):
            for dx in range(-translation, translation + 1, coarse_step):
                for dy in range(-translation, translation + 1, coarse_step):
                    candidate = (
                        score(scale_x, scale_y, dx, dy),
                        scale_x,
                        scale_y,
                        dx,
                        dy,
                    )
                    if candidate < best:
                        best = candidate
    _, coarse_x, coarse_y, coarse_dx, coarse_dy = best
    fine_translation = max(2, round(pdf.shape[1] / 32))
    for scale_x in np.arange(coarse_x - 0.04, coarse_x + 0.041, 0.02):
        for scale_y in np.arange(coarse_y - 0.04, coarse_y + 0.041, 0.02):
            for dx in range(
                coarse_dx - fine_translation,
                coarse_dx + fine_translation + 1,
            ):
                for dy in range(
                    coarse_dy - fine_translation,
                    coarse_dy + fine_translation + 1,
                ):
                    candidate = (
                        score(scale_x, scale_y, dx, dy),
                        scale_x,
                        scale_y,
                        dx,
                        dy,
                    )
                    if candidate < best:
                        best = candidate
    alignment_score, scale_x, scale_y, dx, dy = best
    matrix = np.array(
        [
            [scale_x, 0, (1 - scale_x) * center_x + dx],
            [0, scale_y, (1 - scale_y) * center_y + dy],
        ],
        dtype=np.float32,
    )
    return [warp_mask(item, matrix) for item in candidates], {
        "scaleX": round(float(scale_x), 4),
        "scaleY": round(float(scale_y), 4),
        "dx": int(dx),
        "dy": int(dy),
        "commonInkDistance": round(float(alignment_score), 4),
    }


def alignment_matrix(alignment: dict | None, shape: tuple[int, int]) -> np.ndarray:
    if alignment is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    center_y = (shape[0] - 1) / 2
    center_x = (shape[1] - 1) / 2
    scale_x = alignment["scaleX"]
    scale_y = alignment["scaleY"]
    return np.array(
        [
            [scale_x, 0, (1 - scale_x) * center_x + alignment["dx"]],
            [0, scale_y, (1 - scale_y) * center_y + alignment["dy"]],
        ],
        dtype=np.float32,
    )


def stroke_kind(stroke: dict) -> tuple:
    return (
        stroke.get("feature"),
        tuple(curve.get("command") for curve in stroke.get("curveList", [])),
    )


def unmatched_stroke_indices(
    strokes: list[dict], other: list[dict], maximum_start_distance=10.0
) -> set[int]:
    """Greedily match same-topology strokes by location and return leftovers."""
    unmatched = set(range(len(strokes)))
    used_other: set[int] = set()
    pairs = []
    for left_index, left in enumerate(strokes):
        for right_index, right in enumerate(other):
            if stroke_kind(left) != stroke_kind(right):
                continue
            distance = math.dist(left["start"], right["start"])
            # Matching a same-named stroke on the opposite side of a component
            # destroys exactly the topology we need to inspect (for example
            # 倒八 点+撇 versus 正八 撇+捺).  Ten repository units tolerates
            # layout jitter while preserving left/right and upper/lower roles.
            if distance > maximum_start_distance:
                continue
            pairs.append((distance, left_index, right_index))
    for _, left_index, right_index in sorted(pairs):
        if left_index not in unmatched or right_index in used_other:
            continue
        unmatched.remove(left_index)
        used_other.add(right_index)
    return unmatched


def orientation_maps(mask: np.ndarray) -> list[np.ndarray]:
    kernels = [
        np.ones((1, 5), np.float32),
        np.ones((5, 1), np.float32),
        np.eye(5, dtype=np.float32),
        np.fliplr(np.eye(5, dtype=np.float32)),
    ]
    responses = np.stack(
        [cv2.filter2D(mask.astype(np.float32), -1, kernel) for kernel in kernels]
    )
    winners = responses.argmax(axis=0)
    return [
        mask & (winners == index) & (responses[index] >= 3)
        for index in range(len(kernels))
    ]


def closest_component(mask: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Select residual ink nearest the candidate disagreement, not largest noise."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return mask
    anchor_distance = PDF.distance_transform_edt(~anchor)
    viable = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= 2
    ]
    if not viable:
        return mask
    label = min(
        viable,
        key=lambda value: (
            float(anchor_distance[labels == value].mean()),
            -int(stats[value, cv2.CC_STAT_AREA]),
        ),
    )
    return labels == label


def principal_axis_angle(mask: np.ndarray) -> float | None:
    """Return the dominant centerline angle in [0, 180) degrees."""
    points = np.argwhere(skeletonize(mask))
    if len(points) < 3:
        return None
    xy = points[:, [1, 0]].astype(float)
    covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    return float(math.degrees(math.atan2(direction[1], direction[0])) % 180)


def axis_angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180
    return min(difference, 180 - difference)


def topology_distances(
    pdf: np.ndarray,
    candidate_strokes: list[list[dict]],
    candidate_layers: list[list[np.ndarray]],
    alignment: dict | None,
) -> list[float]:
    target_orientations = orientation_maps(pdf)
    matrix = alignment_matrix(alignment, pdf.shape)
    distances = []
    for candidate_index, (strokes, layers) in enumerate(
        zip(candidate_strokes, candidate_layers)
    ):
        unique_indices: set[int] = set()
        for other_index, other in enumerate(candidate_strokes):
            if candidate_index != other_index:
                unique_indices.update(unmatched_stroke_indices(strokes, other))
        if not unique_indices:
            distances.append(0.0)
            continue
        unique_mask = np.logical_or.reduce(
            [layers[index] for index in sorted(unique_indices)]
        )
        unique_mask = warp_mask(unique_mask, matrix)
        values = []
        for candidate_orientation, target_orientation in zip(
            orientation_maps(unique_mask), target_orientations
        ):
            if not candidate_orientation.any():
                continue
            if not target_orientation.any():
                values.extend([16.0] * int(candidate_orientation.sum()))
                continue
            target_distance = PDF.distance_transform_edt(~target_orientation)
            values.extend(target_distance[candidate_orientation].tolist())
        if not values:
            target_distance = PDF.distance_transform_edt(~pdf)
            values = target_distance[unique_mask].tolist()
        distances.append(float(np.mean(values)) if values else 0.0)
    return distances


def _directed_orientation_distance(
    source: np.ndarray, target: np.ndarray, missing_penalty=16.0
) -> float:
    """Measure stroke direction before contour proximity.

    Each source pixel is compared only with target pixels having the same local
    orientation.  This prevents a nearby falling stroke from satisfying a
    horizontal candidate merely because the two contours happen to touch.
    """
    values = []
    for source_orientation, target_orientation in zip(
        orientation_maps(source), orientation_maps(target)
    ):
        if not source_orientation.any():
            continue
        if not target_orientation.any():
            values.extend([missing_penalty] * int(source_orientation.sum()))
            continue
        distance = PDF.distance_transform_edt(~target_orientation)
        values.extend(
            np.minimum(distance[source_orientation], missing_penalty).tolist()
        )
    if values:
        return float(np.mean(values))
    if not source.any():
        return 0.0
    if not target.any():
        return missing_penalty
    distance = PDF.distance_transform_edt(~target)
    return float(np.minimum(distance[source], missing_penalty).mean())


def _leaf_groups(leaf_ids: list[int]) -> list[dict]:
    """Group consecutive strokes belonging to one terminal component use."""
    groups = []
    for stroke_index, leaf_id in enumerate(leaf_ids):
        if groups and groups[-1]["id"] == leaf_id:
            groups[-1]["strokeIndices"].append(stroke_index)
        else:
            groups.append({"id": leaf_id, "strokeIndices": [stroke_index]})
    return groups


def _lcs_left_indices(left: list[int], right: list[int]) -> set[int]:
    """Return indices from one deterministic longest common subsequence."""
    lengths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index in range(len(left) - 1, -1, -1):
        for right_index in range(len(right) - 1, -1, -1):
            if left[left_index] == right[right_index]:
                lengths[left_index][right_index] = (
                    1 + lengths[left_index + 1][right_index + 1]
                )
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index + 1][right_index],
                    lengths[left_index][right_index + 1],
                )
    matched = set()
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        if (
            left[left_index] == right[right_index]
            and lengths[left_index][right_index]
            == 1 + lengths[left_index + 1][right_index + 1]
        ):
            matched.add(left_index)
            left_index += 1
            right_index += 1
        elif lengths[left_index + 1][right_index] >= lengths[left_index][right_index + 1]:
            left_index += 1
        else:
            right_index += 1
    return matched


def recursive_unique_stroke_indices(
    candidate_strokes: list[list[dict]],
    candidate_leaf_ids: list[list[int]] | None,
) -> tuple[list[set[int]], list[list[dict]] | None, list[set[int]]]:
    """Find the lowest-level disagreement between candidate component trees.

    Terminal component identity is stronger than a same-looking stroke elsewhere
    in the character.  LCS matching preserves repeated components and their
    parent order, so a 撇 on the left cannot cancel a 撇 on the right.  Older
    evidence without leaf provenance falls back to stroke topology + position.
    """
    if candidate_leaf_ids and all(
        len(ids) == len(strokes)
        for ids, strokes in zip(candidate_leaf_ids, candidate_strokes)
    ):
        groups = [_leaf_groups(ids) for ids in candidate_leaf_ids]
        leaf_unique = []
        for candidate_index, candidate_groups in enumerate(groups):
            common = set(range(len(candidate_groups)))
            left = [group["id"] for group in candidate_groups]
            for other_index, other_groups in enumerate(groups):
                if candidate_index == other_index:
                    continue
                common &= _lcs_left_indices(
                    left, [group["id"] for group in other_groups]
                )
            leaf_unique.append(
                {
                    stroke_index
                    for group_index, group in enumerate(candidate_groups)
                    if group_index not in common
                    for stroke_index in group["strokeIndices"]
                }
            )
        # Descend one level further: within the differing terminal components,
        # remove strokes whose topology and local role still agree.  For 匕
        # siblings this keeps only 横↔撇 while discarding their common 竖弯钩.
        refined: list[set[int]] = []
        for candidate_index, (strokes, candidate_indices) in enumerate(
            zip(candidate_strokes, leaf_unique)
        ):
            ordered = sorted(candidate_indices)
            local_strokes = [strokes[index] for index in ordered]
            # Within an already isolated terminal component, compare stroke
            # topology in sequence and ignore parent-layout translation/scale.
            # This removes the common hook from 匕 siblings even when one parent
            # compresses it enough to exceed the global 10-unit position gate.
            common = set(range(len(local_strokes)))
            for other_index, (other_strokes, other_indices) in enumerate(
                zip(candidate_strokes, leaf_unique)
            ):
                if candidate_index == other_index:
                    continue
                other_ordered = sorted(other_indices)
                other_local = [other_strokes[index] for index in other_ordered]
                if max(len(local_strokes), len(other_local)) <= 3:
                    common &= _lcs_left_indices(
                        [stroke_kind(stroke) for stroke in local_strokes],
                        [stroke_kind(stroke) for stroke in other_local],
                    )
                else:
                    common -= unmatched_stroke_indices(
                        local_strokes,
                        other_local,
                    )
            refined.append(
                {
                    ordered[index]
                    for index in range(len(local_strokes))
                    if index not in common
                }
            )
        # An empty side represents an absent stroke rather than a directly
        # observable stroke class. Preserve the complete differing subtree so
        # absence is judged from the full local topology, not from an infinity
        # placeholder.
        return (
            refined if all(indices for indices in refined) else leaf_unique,
            groups,
            leaf_unique,
        )

    unique = []
    for candidate_index, strokes in enumerate(candidate_strokes):
        indices: set[int] = set()
        for other_index, other in enumerate(candidate_strokes):
            if candidate_index != other_index:
                indices.update(unmatched_stroke_indices(strokes, other))
        unique.append(indices)
    return unique, None, unique


def recursive_stroke_distances(
    pdf: np.ndarray,
    candidate_images: list[np.ndarray],
    candidate_strokes: list[list[dict]],
    candidate_layers: list[list[np.ndarray]],
    alignment: dict | None,
    candidate_leaf_ids: list[list[int]] | None = None,
) -> tuple[list[float], dict]:
    """Compare only the terminal strokes that differ between sibling trees.

    The candidates are first recursively flattened to their terminal strokes by
    the repository renderer.  Strokes with the same feature, curve commands and
    position are paired across candidates.  The unpaired strokes are the
    smallest observable disagreement between the sibling trees.  PDF ink
    already explained by candidate-common geometry is removed before scoring,
    so a wrong diagonal cannot borrow evidence from another diagonal elsewhere
    in the parent character.

    This is deliberately a topology-first score.  Whole-glyph optical distance
    remains available only as a later tie-breaker.
    """
    matrix = alignment_matrix(alignment, pdf.shape)
    aligned_layers = [
        [warp_mask(layer, matrix) for layer in layers]
        for layers in candidate_layers
    ]
    unique_indices, leaf_groups, subtree_indices = recursive_unique_stroke_indices(
        candidate_strokes, candidate_leaf_ids
    )
    unique_masks = []
    for indices, layers in zip(unique_indices, aligned_layers):
        unique_masks.append(
            np.logical_or.reduce([layers[index] for index in sorted(indices)])
            if indices
            else np.zeros_like(pdf)
        )
    subtree_masks = [
        np.logical_or.reduce([layers[index] for index in sorted(indices)])
        if indices
        else np.zeros_like(pdf)
        for indices, layers in zip(subtree_indices, aligned_layers)
    ]
    if not any(mask.any() for mask in unique_masks):
        return [0.0 for _ in candidate_images], {
            "decision": "abstain",
            "reason": "no-terminal-stroke-difference",
            "uniqueStrokeIndices": [sorted(value) for value in unique_indices],
            "terminalLeafGroups": leaf_groups,
        }

    scale = max(1.0, pdf.shape[1] / 64)
    unique_union = np.logical_or.reduce(unique_masks)
    roi_width = max(11, round(11 * scale))
    if roi_width % 2 == 0:
        roi_width += 1
    roi = cv2.dilate(
        unique_union.astype(np.uint8), np.ones((roi_width, roi_width), np.uint8)
    ).astype(bool)
    common = candidate_consensus(candidate_images, tolerance=2.0 * scale)
    common_distance = PDF.distance_transform_edt(~common)
    # Keep only PDF ink that candidate-common terminal strokes cannot explain.
    # The small tolerance absorbs antialiasing and registration noise without
    # erasing a genuinely different horizontal/diagonal stroke.
    pdf_difference = pdf & roi & (common_distance > 2.0 * scale)

    def component_count(mask: np.ndarray) -> int:
        reconnect = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((max(1, round(scale)), max(1, round(scale))), np.uint8),
        ).astype(bool)
        skeleton = skeletonize(reconnect)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            skeleton.astype(np.uint8), 8
        )
        minimum_pixels = max(2, round(scale * 2))
        return sum(
            int(stats[label, cv2.CC_STAT_AREA]) >= minimum_pixels
            for label in range(1, count)
        )

    # Use a complete local window for component-count topology. Unlike the
    # residual score below, this intentionally retains shared strokes such as
    # 白 above 匕. It fixes the failure mode where connected-component
    # attribution dropped 白 before topology analysis.
    topology_window_width = max(9, round(9 * scale))
    if topology_window_width % 2 == 0:
        topology_window_width += 1
    subtree_union = np.logical_or.reduce(subtree_masks)
    topology_window = cv2.dilate(
        subtree_union.astype(np.uint8),
        np.ones((topology_window_width, topology_window_width), np.uint8),
    ).astype(bool)
    pdf_component_count = component_count(pdf & topology_window)
    candidate_component_counts = [component_count(mask) for mask in subtree_masks]
    pdf_topology_signature = graph_topology_signature(pdf & topology_window)
    candidate_topology_signatures = [
        graph_topology_signature(mask) for mask in subtree_masks
    ]
    topology_signature_decision = topology_signature_choice(
        pdf_topology_signature, candidate_topology_signatures
    )
    exact_topology_indices = [
        index
        for index, signature in enumerate(candidate_topology_signatures)
        if signature == pdf_topology_signature
    ]
    exact_component_indices = [
        index
        for index, count in enumerate(candidate_component_counts)
        if count == pdf_component_count
    ]
    if not pdf_difference.any():
        return [math.inf for _ in candidate_images], {
            "decision": "abstain",
            "reason": "no-unexplained-pdf-ink-in-difference-window",
            "uniqueStrokeIndices": [sorted(value) for value in unique_indices],
            "terminalLeafGroups": leaf_groups,
        }

    principal_axis_report = None
    if (
        len(unique_indices) == 2
        and all(len(indices) == 1 for indices in unique_indices)
    ):
        principal_target = closest_component(pdf_difference, unique_union)
        pdf_axis = principal_axis_angle(principal_target)
        candidate_axes = [principal_axis_angle(mask) for mask in unique_masks]
        if pdf_axis is not None and all(axis is not None for axis in candidate_axes):
            axis_distances = [
                axis_angle_distance(pdf_axis, axis) for axis in candidate_axes
            ]
            ordered_axes = sorted(
                (distance, index) for index, distance in enumerate(axis_distances)
            )
            principal_axis_report = {
                "pdfAngle": round(pdf_axis, 2),
                "candidateAngles": [round(axis, 2) for axis in candidate_axes],
                "distances": [round(value, 2) for value in axis_distances],
                "candidateIndex": ordered_axes[0][1],
                "margin": round(ordered_axes[1][0] - ordered_axes[0][0], 2),
                "targetPixels": int(principal_target.sum()),
            }

    scores = []
    details = []
    for mask in unique_masks:
        if not mask.any():
            scores.append(math.inf)
            details.append({"forward": math.inf, "reverse": math.inf})
            continue
        forward = _directed_orientation_distance(mask, pdf_difference)
        # Reverse evidence is intentionally weaker: neighbouring parent strokes
        # can enter the window, but the expected PDF stroke must still be
        # explained by the candidate.
        reverse = _directed_orientation_distance(pdf_difference, mask)
        score = forward + 0.35 * reverse
        scores.append(score)
        details.append(
            {
                "forward": round(forward, 4),
                "reverse": round(reverse, 4),
                "score": round(score, 4),
            }
        )
    finite = sorted((score, index) for index, score in enumerate(scores))
    margin = (
        finite[1][0] - finite[0][0]
        if len(finite) > 1 and math.isfinite(finite[1][0])
        else 0.0
    )
    return scores, {
        "decision": "candidate" if margin > 0 else "abstain",
        "candidateIndex": finite[0][1] if finite and margin > 0 else None,
        "margin": round(margin, 4),
        "uniqueStrokeIndices": [sorted(value) for value in unique_indices],
        "differingSubtreeStrokeIndices": [
            sorted(value) for value in subtree_indices
        ],
        "terminalLeafGroups": leaf_groups,
        "pdfDifferencePixels": int(pdf_difference.sum()),
        "candidateDetails": details,
        "pdfComponentCount": pdf_component_count,
        "candidateComponentCounts": candidate_component_counts,
        "pdfTopologySignature": pdf_topology_signature,
        "candidateTopologySignatures": candidate_topology_signatures,
        "topologySignatureDecision": topology_signature_decision,
        "principalAxis": principal_axis_report,
        "exactTopologyCandidateIndex": (
            exact_topology_indices[0] if len(exact_topology_indices) == 1 else None
        ),
        "exactComponentCandidateIndex": (
            exact_component_indices[0] if len(exact_component_indices) == 1 else None
        ),
    }


def confidence_threshold(results: list[dict], precision_target=0.995):
    ordered = sorted(results, key=lambda item: item["margin"], reverse=True)
    correct = 0
    selected = None
    index = 0
    while index < len(ordered):
        margin = ordered[index]["margin"]
        group_end = index
        while group_end < len(ordered) and ordered[group_end]["margin"] == margin:
            correct += ordered[group_end]["correct"]
            group_end += 1
        precision = correct / group_end
        if precision >= precision_target:
            selected = {
                "minimumMargin": margin,
                "accepted": group_end,
                "correct": correct,
                "precision": precision,
                "coverage": group_end / len(results),
            }
        index = group_end
    if selected is None:
        return None
    return selected


def summarize_results(results: list[dict]):
    correct = sum(item["correct"] for item in results)
    return {
        "evaluated": len(results),
        "top1Correct": correct,
        "top1Accuracy": correct / len(results) if results else None,
    }


def acceptance_summary(all_results: list[dict], accepted: list[dict]):
    """Report accepted accuracy without losing the held-out denominator."""
    correct = sum(item["correct"] for item in accepted)
    return {
        "evaluated": len(all_results),
        "accepted": len(accepted),
        "correct": correct,
        "accuracyWhenAccepted": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(all_results) if all_results else None,
    }


def topology_gate_choice(result: dict, minimum_topology_margin, maximum_visual_margin):
    visual = sorted(
        (candidate["visualDistance"], candidate["id"])
        for candidate in result["candidates"]
    )
    topology = sorted(
        (candidate["topologyDistance"], candidate["id"])
        for candidate in result["candidates"]
    )
    choice = visual[0][1]
    override = (
        topology[0][1] != choice
        and topology[1][0] - topology[0][0] >= minimum_topology_margin
        and visual[1][0] - visual[0][0] <= maximum_visual_margin
    )
    return (topology[0][1] if override else choice), override


def summarize_topology_gate(results, minimum_topology_margin, maximum_visual_margin):
    correct = 0
    overrides = 0
    for result in results:
        choice, override = topology_gate_choice(
            result, minimum_topology_margin, maximum_visual_margin
        )
        correct += choice == result["expectedGlyphId"]
        overrides += override
    return {
        "evaluated": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else None,
        "overrides": overrides,
    }


def calibrate_topology_gate(train_results):
    if not train_results:
        return None
    candidates = []
    for topology_step in range(5, 61):
        minimum_topology_margin = topology_step / 10
        for visual_step in range(1, 101):
            maximum_visual_margin = visual_step / 100
            summary = summarize_topology_gate(
                train_results, minimum_topology_margin, maximum_visual_margin
            )
            candidates.append(
                (
                    summary["correct"],
                    -summary["overrides"],
                    minimum_topology_margin,
                    -maximum_visual_margin,
                    summary,
                )
            )
    _, _, minimum_topology_margin, negative_visual_margin, summary = max(candidates)
    return {
        "minimumTopologyMargin": minimum_topology_margin,
        "maximumVisualMargin": -negative_visual_margin,
        "train": summary,
    }


def candidate_absence_probe(train_results):
    positive = []
    withheld = []
    for result in train_results:
        expected = next(
            candidate
            for candidate in result["candidates"]
            if candidate["id"] == result["expectedGlyphId"]
        )
        alternatives = [
            candidate
            for candidate in result["candidates"]
            if candidate["id"] != result["expectedGlyphId"]
        ]
        if not alternatives:
            continue
        positive.append(expected["visualDistance"])
        withheld.append(min(item["visualDistance"] for item in alternatives))
    return {
        "enabled": False,
        "reason": "absolute residuals overlap; keep no-match cases for review",
        "evaluated": len(positive),
        "expectedCloser": sum(
            expected < alternative for expected, alternative in zip(positive, withheld)
        ),
        "expectedDistance": {
            "minimum": min(positive) if positive else None,
            "mean": sum(positive) / len(positive) if positive else None,
            "maximum": max(positive) if positive else None,
        },
        "withheldExpectedDistance": {
            "minimum": min(withheld) if withheld else None,
            "mean": sum(withheld) / len(withheld) if withheld else None,
            "maximum": max(withheld) if withheld else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path)
    parser.add_argument(
        "--pdf",
        type=Path,
        help="use direct vector page/cell evidence instead of pre-rendered PBM pages",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer", choices=("centerline", "svg"), default="centerline"
    )
    parser.add_argument("--alignment", choices=("none", "common"), default="none")
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--evidence-size", type=int, default=128)
    parser.add_argument(
        "--unicode",
        action="append",
        type=lambda value: int(value, 0),
        help="limit an experiment to one or more decimal/0x-prefixed codepoints",
    )
    args = parser.parse_args()
    if args.pdf is None and args.pages_dir is None:
        parser.error("one of --pdf or --pages-dir is required")
    evidence_size = args.evidence_size

    candidate_rows = json.loads(args.candidates.read_text(encoding="utf-8"))["rows"]
    if args.unicode:
        selected_codepoints = set(args.unicode)
        candidate_rows = [
            row for row in candidate_rows if row["unicode"] in selected_codepoints
        ]
    candidates_by_unicode = {row["unicode"]: row for row in candidate_rows}
    records, page_sizes = PDF.parse_pdf_cells(args.bbox_cache)
    records_by_page = defaultdict(list)
    for record in records:
        if record["unicode"] in candidates_by_unicode:
            records_by_page[record["page"]].append(record)
    rendered = {}
    stroke_layers = {}
    for row in candidate_rows:
        for glyph_id, strokes in row["candidates"].items():
            numeric_id = int(glyph_id)
            if numeric_id in rendered:
                continue
            stroke_layers[numeric_id] = render_stroke_layers(
                strokes,
                size=max(128, evidence_size * 2),
                output_size=evidence_size,
            )
            if args.renderer == "svg":
                svg = row.get("candidateSvgs", {}).get(glyph_id)
                if svg is None:
                    raise RuntimeError(
                        f"candidate {glyph_id} has no SVG; regenerate candidate evidence"
                    )
                rendered[numeric_id] = render_svg(
                    svg,
                    size=max(256, evidence_size * 3),
                    output_size=evidence_size,
                )
            else:
                rendered[numeric_id] = render_strokes(
                    strokes,
                    size=max(128, evidence_size * 2),
                    output_size=evidence_size,
                )

    results = []
    for page_number, page_records in sorted(records_by_page.items()):
        page_svg = (
            load_pdf_page_svg(args.pdf, page_number) if args.pdf is not None else None
        )
        page_context = (
            contextlib.nullcontext(None)
            if page_svg is not None
            else Image.open(args.pages_dir / f"page-{page_number:03d}.pbm")
        )
        with page_context as page:
            for record in page_records:
                row = candidates_by_unicode[record["unicode"]]
                expected = row["sourceGlyphs"].get(record["source"])
                if expected is None:
                    continue
                pdf_skeleton = (
                    pdf_vector_skeleton(
                        page_svg,
                        record["bbox"],
                        output_size=evidence_size,
                    )
                    if page_svg is not None
                    else PDF.normalized_skeleton(
                        page,
                        record["bbox"],
                        page_sizes[page_number],
                        size=evidence_size,
                    )
                )
                candidate_ids = [int(glyph_id) for glyph_id in row["candidates"]]
                candidate_images = [rendered[glyph_id] for glyph_id in candidate_ids]
                alignment = None
                if args.alignment == "common":
                    candidate_images, alignment = align_candidates_to_pdf(
                        pdf_skeleton, candidate_images
                    )
                # Audit the registered common geometry with a tolerant
                # consensus. Exact pixel intersection used to erase a shared
                # misplaced branch after tiny per-candidate layout shifts.
                shared_residual = shared_structure_residual(
                    pdf_skeleton, candidate_images
                )
                visual_scores = discriminative_distance(pdf_skeleton, candidate_images)
                candidate_strokes = [
                    row["candidates"][str(glyph_id)] for glyph_id in candidate_ids
                ]
                candidate_layers = [
                    stroke_layers[glyph_id] for glyph_id in candidate_ids
                ]
                topology_scores = topology_distances(
                    pdf_skeleton,
                    candidate_strokes,
                    candidate_layers,
                    alignment,
                )
                recursive_scores, recursive_report = recursive_stroke_distances(
                    pdf_skeleton,
                    candidate_images,
                    candidate_strokes,
                    candidate_layers,
                    alignment,
                    [
                        (
                            row.get("candidateLeafStrokeIds")
                            or row.get("candidateLeafIds")
                            or {}
                        ).get(str(glyph_id), [])
                        for glyph_id in candidate_ids
                    ]
                    if row.get("candidateLeafStrokeIds")
                    or row.get("candidateLeafIds")
                    else None,
                )
                # All indices inside recursiveTopology refer to this stable
                # repository order. The human-readable candidate list below is
                # sorted by optical distance and must never be used to decode
                # those indices.
                recursive_report["candidateGlyphIds"] = candidate_ids
                scores = [
                    visual + args.topology_weight * topology
                    for visual, topology in zip(visual_scores, topology_scores)
                ]
                distances = sorted(zip(scores, candidate_ids))
                optical_best_distance, optical_best_id = distances[0]
                recursive_ranking = sorted(zip(recursive_scores, candidate_ids))
                recursive_best_distance, recursive_best_id = recursive_ranking[0]
                exact_component_index = recursive_report.get(
                    "exactComponentCandidateIndex"
                )
                if exact_component_index is not None:
                    recursive_best_id = candidate_ids[exact_component_index]
                recursive_decisive = (
                    recursive_report["decision"] == "candidate"
                    and math.isfinite(recursive_best_distance)
                )
                if recursive_decisive and exact_component_index is not None:
                    best_id = recursive_best_id
                    best_distance = recursive_scores[exact_component_index]
                    second_distance = min(
                        score
                        for index, score in enumerate(recursive_scores)
                        if index != exact_component_index
                    )
                else:
                    chosen_ranking = (
                        recursive_ranking if recursive_decisive else distances
                    )
                    best_distance, best_id = chosen_ranking[0]
                    second_distance = chosen_ranking[1][0]
                results.append(
                    {
                        "unicode": record["unicode"],
                        "source": record["source"],
                        "split": row.get("split"),
                        "familyKeys": row.get("familyKeys", []),
                        "expectedGlyphId": expected,
                        "predictedGlyphId": best_id,
                        "correct": best_id == expected,
                        "predictionMethod": (
                            "recursive-component-count-topology"
                            if recursive_decisive
                            and exact_component_index is not None
                            else "recursive-terminal-stroke-topology"
                            if recursive_decisive
                            else "whole-glyph-optical-fallback"
                        ),
                        "opticalPredictionGlyphId": optical_best_id,
                        "recursiveTopologyPredictionGlyphId": (
                            recursive_best_id if recursive_decisive else None
                        ),
                        "recursiveTopology": recursive_report,
                        "bestDistance": round(best_distance, 4),
                        "secondDistance": round(second_distance, 4),
                        "margin": round(second_distance - best_distance, 4),
                        "alignment": alignment,
                        "candidateSetIncomplete": shared_residual["suspected"],
                        "sharedStructureResidual": shared_residual,
                        "candidates": [
                            {
                                "id": glyph_id,
                                "distance": round(distance, 4),
                                "visualDistance": round(
                                    visual_scores[candidate_ids.index(glyph_id)], 4
                                ),
                                "topologyDistance": round(
                                    topology_scores[candidate_ids.index(glyph_id)], 4
                                ),
                                "recursiveTopologyDistance": round(
                                    recursive_scores[candidate_ids.index(glyph_id)], 4
                                ),
                            }
                            for distance, glyph_id in distances
                        ],
                    }
                )
    summary = summarize_results(results)
    train_results = [item for item in results if item.get("split") == "train"]
    test_results = [item for item in results if item.get("split") == "test"]
    calibration = confidence_threshold(train_results) if train_results else None
    held_out_acceptance = None
    if calibration is not None and test_results:
        accepted = [
            item
            for item in test_results
            if item["margin"] >= calibration["minimumMargin"]
        ]
        held_out_acceptance = acceptance_summary(test_results, accepted)
    topology_gate = calibrate_topology_gate(train_results)
    if topology_gate is not None:
        topology_gate["test"] = summarize_topology_gate(
            test_results,
            topology_gate["minimumTopologyMargin"],
            topology_gate["maximumVisualMargin"],
        )
        for result in results:
            prediction, override = topology_gate_choice(
                result,
                topology_gate["minimumTopologyMargin"],
                topology_gate["maximumVisualMargin"],
            )
            result["topologyGatePrediction"] = prediction
            result["topologyGateOverride"] = override
            result["topologyGateCorrect"] = prediction == result["expectedGlyphId"]
    payload = {
        "metadata": {
            "method": (
                "recursive terminal-stroke topology before whole-glyph optical "
                f"fallback; repository {args.renderer} render with "
                f"{args.alignment} alignment at {evidence_size}px"
            ),
            **summary,
            # This descriptive value sees both splits and must never be used as
            # an apply threshold. Only heldOutAcceptance tests train calibration.
            "inSampleHighConfidence": confidence_threshold(results),
            "automaticWriteEnabled": False,
            "candidateSetIncompleteReview": sum(
                item["candidateSetIncomplete"] for item in results
            ),
            "splits": {
                "train": summarize_results(train_results),
                "test": summarize_results(test_results),
                "calibration": calibration,
                "heldOutAcceptance": held_out_acceptance,
                "topologyGate": topology_gate,
                "candidateAbsence": candidate_absence_probe(train_results),
            },
            "pdfVectorEvidence": args.pdf is not None,
            "fontOutlinesReusedAsProjectData": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
