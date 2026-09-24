"""Create deterministic PDF-vs-SVG panels for human or multimodal review."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_matcher():
    path = Path(__file__).with_name("unihan-render-match.py")
    spec = importlib.util.spec_from_file_location("unihan_render_match", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_matcher()
PDF = MATCHER.PDF
cv2 = MATCHER.cv2

TOPOLOGY_COLORS = {
    "endpoint": (220, 38, 38),
    "junction": (2, 132, 199),
    "corner": (22, 163, 74),
}


def pdf_cell(page, bbox, page_size, size=256):
    x_scale = page.width / page_size[0]
    y_scale = page.height / page_size[1]
    margin = 1.5
    box = (
        max(0, round((bbox[0] - margin) * x_scale)),
        max(0, round((bbox[1] - margin) * y_scale)),
        min(page.width, round((bbox[2] + margin) * x_scale)),
        min(page.height, round((bbox[3] + margin) * y_scale)),
    )
    image = page.crop(box).convert("L")
    array = np.asarray(image) < 224
    points = np.argwhere(array)
    if points.size:
        top, left = points.min(axis=0)
        bottom, right = points.max(axis=0) + 1
        image = image.crop((left, top, right, bottom))
    target = size - 24
    scale = min(target / image.width, target / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas.convert("RGB")


def render_svg_review_image(svg: str, size=256) -> Image.Image:
    """Rasterize review SVGs in color; scoring remains grayscale elsewhere."""
    executable = shutil.which("magick")
    if executable is None:
        raise RuntimeError(
            "SVG review panels require ImageMagick's `magick` executable on PATH"
        )
    completed = subprocess.run(
        [
            executable,
            "-background",
            "white",
            "svg:-",
            "-resize",
            f"{size}x{size}!",
            "png:-",
        ],
        input=svg.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    with Image.open(io.BytesIO(completed.stdout)) as image:
        return image.convert("RGB")


def _point_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    return [
        (round(float(centroids[label][0])), round(float(centroids[label][1])))
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] > 0
    ]


def thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, kept local so review tooling needs no new package."""
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


def topology_points(binary: np.ndarray):
    """Return endpoint/junction centers and the one-pixel raster skeleton."""
    skeleton = thin(binary)
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
        (~ring[index]) & ring[(index + 1) % len(ring)] for index in range(len(ring))
    )
    # The circular transition count is invariant to diagonal adjacency. A
    # right-angle turn has two branches, not the three raw 8-neighbours that
    # would otherwise make it look like a junction.
    endpoints = _point_centers(skeleton & (transitions == 1))
    junction_mask = skeleton & (transitions >= 3)
    junction_mask = (
        cv2.dilate(junction_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    )
    junctions = _point_centers(junction_mask)
    return endpoints, junctions, skeleton


def corner_points(
    skeleton: np.ndarray,
    endpoints: list[tuple[int, int]],
    junctions: list[tuple[int, int]],
):
    """Find strong direction changes that are neither endpoints nor junctions.

    Graph degree alone deliberately treats an L-shaped turn as an ordinary
    degree-two pixel. Shi-Tomasi corners add that missing geometric information
    while nearby endpoint/junction detections are suppressed to keep the legend
    categories disjoint.
    """
    detected = cv2.goodFeaturesToTrack(
        skeleton.astype(np.uint8) * 255,
        maxCorners=64,
        qualityLevel=0.08,
        minDistance=8,
        blockSize=7,
        useHarrisDetector=False,
    )
    if detected is None:
        return []
    graph_nodes = [*endpoints, *junctions]
    corners = []
    for point in detected.reshape(-1, 2):
        x, y = (int(round(float(value))) for value in point)
        if any(
            math.hypot(x - node_x, y - node_y) <= 7 for node_x, node_y in graph_nodes
        ):
            continue
        corners.append((x, y))
    return corners


def _cluster_axis(values: list[float], maximum_gap: float) -> list[float]:
    """Collapse nearby raster detections into one geometric location."""
    if not values:
        return []
    groups = [[value] for value in sorted(values)]
    merged = [groups[0]]
    for group in groups[1:]:
        if group[0] - merged[-1][-1] <= maximum_gap:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return [float(np.mean(group)) for group in merged]


def _branch_angle_at(
    skeleton: np.ndarray, x: float, y: float, radius_x=10, radius_y=18
) -> float | None:
    """Estimate a crossing branch angle after excluding the horizontal itself."""
    points = np.argwhere(skeleton)
    if not points.size:
        return None
    dy = points[:, 0] - y
    dx = points[:, 1] - x
    local = points[
        (np.abs(dx) <= radius_x) & (np.abs(dy) <= radius_y) & (np.abs(dy) >= 4)
    ]
    if len(local) < 6:
        return None
    coordinates = np.column_stack((local[:, 1], local[:, 0])).astype(float)
    coordinates -= coordinates.mean(axis=0)
    covariance = np.cov(coordinates, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    angle = abs(float(np.degrees(np.arctan2(direction[1], direction[0]))))
    return min(angle, 180 - angle)


def horizontal_crossing_evidence(binary: np.ndarray) -> dict:
    """Find offset stroke crossings on a long horizontal skeleton segment.

    This intentionally measures local topology only. Overall contour curvature is
    not returned: a direct falling stroke and a vertical-plus-falling pair both
    produce a curved outer silhouette, so that feature cannot distinguish them.
    """
    _endpoints, junctions, skeleton = topology_points(binary)
    height, width = skeleton.shape
    # A horizontal opening recovers the complete ink band even when crossings
    # split a one-pixel Hough segment. This matters for the two intersections in
    # U+7740 J, where treating each fragment separately hides the left crossing.
    horizontal_mask = cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((1, max(9, round(width * 0.08))), dtype=np.uint8),
    )
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        horizontal_mask, 8
    )
    horizontal_lines = [
        {
            "left": float(stats[label, cv2.CC_STAT_LEFT]),
            "right": float(
                stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] - 1
            ),
            "y": float(centroids[label][1]),
        }
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_WIDTH] >= width * 0.18
        and stats[label, cv2.CC_STAT_WIDTH] >= stats[label, cv2.CC_STAT_HEIGHT] * 3
    ]
    if not horizontal_lines:
        return {"decision": "abstain", "reason": "no-long-horizontal"}

    rows: list[dict] = []
    for line in sorted(horizontal_lines, key=lambda item: item["y"]):
        target = next(
            (
                row
                for row in rows
                if abs(row["y"] - line["y"]) <= 5
                and line["left"] <= row["right"] + 18
                and line["right"] >= row["left"] - 18
            ),
            None,
        )
        if target is None:
            rows.append(dict(line))
        else:
            target["left"] = min(target["left"], line["left"])
            target["right"] = max(target["right"], line["right"])
            target["y"] = (target["y"] + line["y"]) / 2

    candidates = []
    for row in rows:
        span = row["right"] - row["left"]
        if span < width * 0.18 or not height * 0.3 <= row["y"] <= height * 0.72:
            continue
        # Junction dilation also marks decorated line ends. They are not stroke
        # crossings, so only accept points safely inside the horizontal span.
        interior_margin = max(8, span * 0.12)
        crossing_xs = _cluster_axis(
            [
                float(x)
                for x, y in junctions
                if row["left"] + interior_margin <= x <= row["right"] - interior_margin
                and abs(y - row["y"]) <= 10
            ],
            max(7, width * 0.045),
        )
        angles = [_branch_angle_at(skeleton, x, row["y"]) for x in crossing_xs]
        candidates.append(
            {
                "horizontalY": round(row["y"], 1),
                "horizontalSpan": [round(row["left"], 1), round(row["right"], 1)],
                "crossingXs": [round(value, 1) for value in crossing_xs],
                "crossingAngles": [
                    None if angle is None else round(angle, 1) for angle in angles
                ],
            }
        )

    qualified = [
        row
        for row in candidates
        if len(row["crossingXs"]) >= 2
        and row["crossingXs"][-1] - row["crossingXs"][0] >= width * 0.08
        and any(angle is not None and angle >= 72 for angle in row["crossingAngles"])
    ]
    if not qualified:
        return {
            "decision": "abstain",
            "reason": "no-stable-offset-double-crossing",
            "rows": candidates,
        }
    best = max(
        qualified,
        key=lambda row: (
            len(row["crossingXs"]),
            row["crossingXs"][-1] - row["crossingXs"][0],
            row["horizontalY"],
        ),
    )
    return {
        "decision": "split-vertical-and-falling-strokes",
        "reason": "offset-crossings-with-near-orthogonal-branch",
        **best,
    }


def horizontal_crossing_consensus(image: Image.Image) -> dict:
    """Require the local-topology conclusion to survive threshold changes."""
    gray = np.asarray(image.convert("L"))
    thresholds = (128, 160, 192)
    trials = [horizontal_crossing_evidence(gray < value) for value in thresholds]
    positive = [
        trial
        for trial in trials
        if trial["decision"] == "split-vertical-and-falling-strokes"
    ]
    groups: list[list[dict]] = []
    for trial in positive:
        matching = next(
            (
                group
                for group in groups
                if abs(group[0]["horizontalY"] - trial["horizontalY"])
                <= image.height * 0.06
                and abs(group[0]["crossingXs"][0] - trial["crossingXs"][0])
                <= image.width * 0.08
                and abs(group[0]["crossingXs"][-1] - trial["crossingXs"][-1])
                <= image.width * 0.08
            ),
            None,
        )
        if matching is None:
            groups.append([trial])
        else:
            matching.append(trial)
    stable = max(groups, key=len, default=[])
    decision = "split-vertical-and-falling-strokes" if len(stable) >= 2 else "abstain"
    return {
        "decision": decision,
        "consensus": f"{len(stable)}/{len(thresholds)}",
        "stableEvidence": stable[0] if stable else None,
        "trials": [
            {"threshold": threshold, **trial}
            for threshold, trial in zip(thresholds, trials)
        ],
    }


def choose_split_stroke_candidate(pdf_evidence: dict, candidate_topology: dict) -> dict:
    """Match PDF local topology to repository stroke identity, or abstain."""
    if pdf_evidence.get("decision") != "split-vertical-and-falling-strokes":
        return {"decision": "abstain", "reason": "pdf-topology-not-stable"}
    matches = [
        int(candidate_id)
        for candidate_id, structures in candidate_topology.items()
        if any(
            structure.get("hasSeparateVerticalAndFallingLeaves")
            for structure in structures
        )
    ]
    if len(matches) != 1:
        return {
            "decision": "abstain",
            "reason": "candidate-structure-not-unique",
            "matchingCandidateIds": matches,
        }
    return {
        "decision": "candidate",
        "candidateId": matches[0],
        "reason": "stable-offset-crossings-match-separate-vertical-and-falling-leaves",
    }


def draw_topology_markers(
    image: Image.Image,
    endpoints: list[tuple[int, int]],
    junctions: list[tuple[int, int]],
    corners: list[tuple[int, int]],
    radius=5,
) -> Image.Image:
    """Draw large, haloed topology nodes without confusing them with editor handles."""
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for points, color in (
        (endpoints, TOPOLOGY_COLORS["endpoint"]),
        (junctions, TOPOLOGY_COLORS["junction"]),
        (corners, TOPOLOGY_COLORS["corner"]),
    ):
        for x, y in points:
            draw.ellipse(
                (
                    x - radius - 2,
                    y - radius - 2,
                    x + radius + 2,
                    y + radius + 2,
                ),
                fill="white",
            )
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def topology_overlay(image: Image.Image) -> Image.Image:
    """Put the lower-row topology node vocabulary onto a candidate SVG."""
    gray = np.asarray(image.convert("L"))
    endpoints, junctions, skeleton = topology_points(gray < 224)
    corners = corner_points(skeleton, endpoints, junctions)
    return draw_topology_markers(image, endpoints, junctions, corners, radius=6)


def topology_panel(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    endpoints, junctions, skeleton = topology_points(gray < 224)
    corners = corner_points(skeleton, endpoints, junctions)
    canvas = Image.new("RGB", image.size, "white")
    array = np.asarray(canvas).copy()
    array[skeleton] = (55, 65, 81)
    canvas = Image.fromarray(array)
    return draw_topology_markers(canvas, endpoints, junctions, corners)


def topology_signature(image: Image.Image):
    endpoints, junctions, skeleton = topology_points(
        np.asarray(image.convert("L")) < 224
    )
    corners = corner_points(skeleton, endpoints, junctions)
    components = cv2.connectedComponents(skeleton.astype(np.uint8), 8)[0] - 1
    ink_points = np.argwhere(skeleton)
    if ink_points.size:
        top, left = ink_points.min(axis=0)
        bottom, right = ink_points.max(axis=0)
        x_span = max(1, int(right - left))
        y_span = max(1, int(bottom - top))

        def normalize(points):
            return [
                [round((x - left) / x_span, 4), round((y - top) / y_span, 4)]
                for x, y in points
            ]

        sample_step = max(1, len(ink_points) // 128)
        sampled_skeleton = [
            (int(x), int(y)) for y, x in ink_points[::sample_step][:128]
        ]
    else:

        def normalize(_points):
            return []

        sampled_skeleton = []
    orientation_maps = MATCHER.orientation_maps(skeleton)
    orientation_counts = [int(value.sum()) for value in orientation_maps]
    orientation_total = sum(orientation_counts)
    return {
        "components": int(components),
        "endpoints": len(endpoints),
        "junctions": len(junctions),
        "corners": len(corners),
        "endpointPositions": normalize(endpoints),
        "junctionPositions": normalize(junctions),
        "cornerPositions": normalize(corners),
        "skeletonPositions": normalize(sampled_skeleton),
        "orientationHistogram": [
            round(count / orientation_total, 6) if orientation_total else 0.0
            for count in orientation_counts
        ],
        "horizontalCrossingEvidence": horizontal_crossing_consensus(image),
    }


def discriminative_box(images: list[Image.Image], margin=12):
    masks = [np.asarray(image.convert("L")) < 224 for image in images]
    varying = np.logical_or.reduce(masks) & ~np.logical_and.reduce(masks)
    points = np.argwhere(varying)
    if points.size == 0:
        return (0, 0, images[0].width, images[0].height)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    return (
        max(0, int(left) - margin),
        max(0, int(top) - margin),
        min(images[0].width, int(right) + margin),
        min(images[0].height, int(bottom) + margin),
    )


def color_box(images: list[Image.Image], color: str, margin=4):
    value = color.removeprefix("#")
    target = np.array(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=int
    )
    masks = []
    for image in images:
        rgb = np.asarray(image.convert("RGB"), dtype=int)
        masks.append(np.linalg.norm(rgb - target, axis=2) < 45)
    points = np.argwhere(np.logical_or.reduce(masks))
    if points.size == 0:
        return discriminative_box(images, margin)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    return (
        max(0, int(left) - margin),
        max(0, int(top) - margin),
        min(images[0].width, int(right) + margin),
        min(images[0].height, int(bottom) + margin),
    )


def isolate_color(image: Image.Image, color: str):
    value = color.removeprefix("#")
    target = np.array(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=int
    )
    rgb = np.asarray(image.convert("RGB"), dtype=int)
    distance = np.linalg.norm(rgb - target, axis=2)
    output = np.full(rgb.shape, 255, dtype=np.uint8)
    output[distance < 60] = (0, 0, 0)
    return Image.fromarray(output)


def color_mask(image: Image.Image, color: str):
    value = color.removeprefix("#")
    target = np.array(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=int
    )
    rgb = np.asarray(image.convert("RGB"), dtype=int)
    return np.linalg.norm(rgb - target, axis=2) < 60


def mask_panel(mask: np.ndarray, size=256):
    points = np.argwhere(mask)
    canvas = Image.new("L", (size, size), "white")
    if points.size == 0:
        return canvas.convert("RGB")
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    crop = Image.fromarray((~mask[top:bottom, left:right]).astype(np.uint8) * 255)
    crop.thumbnail((size - 24, size - 24), Image.Resampling.NEAREST)
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas.convert("RGB")


def structure_guided_focus(
    pdf_image: Image.Image,
    candidate_images: list[Image.Image],
    color: str,
    candidate_target_images: list[Image.Image] | None = None,
):
    """Attribute PDF ink to a leaf after aligning on all invariant leaves."""
    masks = structure_guided_focus_masks(
        pdf_image, candidate_images, color, candidate_target_images
    )
    return [mask_panel(mask) for mask in masks]


def structure_guided_focus_masks(
    pdf_image: Image.Image,
    candidate_images: list[Image.Image],
    color: str,
    candidate_target_images: list[Image.Image] | None = None,
):
    """Return attributed PDF ink and aligned target leaves in one coordinate space."""
    pdf = np.asarray(pdf_image.convert("L")) < 224
    visible_targets = [color_mask(image, color) for image in candidate_images]
    targets = [
        color_mask(image, color)
        for image in (candidate_target_images or candidate_images)
    ]
    full = [np.asarray(image.convert("L")) < 224 for image in candidate_images]
    kernel = np.ones((5, 5), dtype=np.uint8)
    non_targets = [
        ink & ~(cv2.dilate(target.astype(np.uint8), kernel, iterations=1).astype(bool))
        for ink, target in zip(full, visible_targets)
    ]
    _aligned_non_targets, alignment = MATCHER.align_candidates_to_pdf(pdf, non_targets)
    global_matrix = MATCHER.alignment_matrix(alignment, pdf.shape)
    aligned_targets = [MATCHER.warp_mask(target, global_matrix) for target in targets]
    aligned_others = [MATCHER.warp_mask(other, global_matrix) for other in non_targets]
    target_union = np.logical_or.reduce(aligned_targets)
    other_union = np.logical_or.reduce(aligned_others)
    target_distance = PDF.distance_transform_edt(~target_union)
    other_distance = PDF.distance_transform_edt(~other_union)
    pixel_attribution = (
        pdf & (target_distance <= other_distance) & (target_distance <= 10)
    )
    attributed = np.zeros_like(pdf)
    count, labels = cv2.connectedComponents(pdf.astype(np.uint8), 8)
    for label in range(1, count):
        component = labels == label
        target_values = target_distance[component]
        other_values = other_distance[component]
        target_mean = float(target_values.mean())
        other_mean = float(other_values.mean())
        target_near = float(np.quantile(target_values, 0.1))
        other_near = float(np.quantile(other_values, 0.1))
        if target_near <= 5 and target_mean + 1 < other_mean:
            attributed |= component
        elif other_near <= 5 and other_mean + 1 < target_mean:
            continue
        else:
            attributed |= component & pixel_attribution
    return [attributed, *aligned_targets]


def target_window_focus_masks(
    pdf_image: Image.Image,
    candidate_images: list[Image.Image],
    color: str,
    candidate_target_images: list[Image.Image] | None = None,
    margin: int = 12,
):
    """Keep chart ink in the aligned target-leaf window without classifying it.

    This deliberately tolerates overlapping/adjacent parent strokes. It avoids
    turning a difficult attribution decision into an empty observation.
    """
    pdf = np.asarray(pdf_image.convert("L")) < 224
    visible_targets = [color_mask(image, color) for image in candidate_images]
    targets = [
        color_mask(image, color)
        for image in (candidate_target_images or candidate_images)
    ]
    full = [np.asarray(image.convert("L")) < 224 for image in candidate_images]
    kernel = np.ones((5, 5), dtype=np.uint8)
    non_targets = [
        ink & ~cv2.dilate(target.astype(np.uint8), kernel, iterations=1).astype(bool)
        for ink, target in zip(full, visible_targets)
    ]
    _aligned_non_targets, alignment = MATCHER.align_candidates_to_pdf(pdf, non_targets)
    matrix = MATCHER.alignment_matrix(alignment, pdf.shape)
    aligned_targets = [MATCHER.warp_mask(target, matrix) for target in targets]
    target_union = np.logical_or.reduce(aligned_targets)
    window = cv2.dilate(
        target_union.astype(np.uint8),
        np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8),
    ).astype(bool)
    return [pdf & window, *aligned_targets]


def focus_panel(image: Image.Image, box, size=256):
    crop = image.crop(box)
    crop.thumbnail((size - 16, size - 16), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def without_review_points(svg: str) -> str:
    return re.sub(r"<circle\b[^>]*/>", "", svg)


def draw_feature_annotations(
    montage: Image.Image, config: dict | list[dict], topology_row: bool
) -> Image.Image:
    """Mark the same reviewed feature in the glyph and topology rows."""
    if isinstance(config, list):
        subject = None
        annotations = config
    else:
        subject = config.get("subject")
        annotations = config.get("annotations", [])
    if not annotations:
        return montage
    footer_lines = len(annotations) + (1 if subject else 0)
    footer_height = 8 + 22 * footer_lines
    annotated = Image.new(
        "RGB", (montage.width, montage.height + footer_height), "white"
    )
    annotated.paste(montage, (0, 0))
    draw = ImageDraw.Draw(annotated)
    orange = (217, 119, 6)
    footer_line = 0
    if subject:
        draw.text(
            (8, montage.height + 5),
            f"Review target: {subject}",
            fill=(30, 64, 175),
        )
        footer_line = 1
    for number, annotation in enumerate(annotations, start=1):
        candidate = str(annotation["candidate"]).upper()
        panel_index = ord(candidate) - ord("A") + 1
        points = [
            ("pdf", 0, 36),
            ("glyph", panel_index, 36),
        ]
        if topology_row:
            points.extend(
                [
                    ("pdfTopology", 0, 324),
                    ("topology", panel_index, 324),
                ]
            )
        for key, target_panel, row_y in points:
            if key not in annotation:
                continue
            x_fraction, y_fraction = annotation[key]
            x = target_panel * 256 + round(float(x_fraction) * 256)
            y = row_y + round(float(y_fraction) * 256)
            draw.ellipse(
                (x - 11, y - 11, x + 11, y + 11),
                outline=orange,
                width=4,
            )
            draw.ellipse((x + 7, y - 20, x + 23, y - 4), fill=orange)
            draw.text((x + 12, y - 19), str(number), fill="white", anchor="ma")
        draw.text(
            (8, montage.height + 5 + (footer_line + number - 1) * 22),
            f"{number}. {annotation['text']}",
            fill=orange,
        )
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--answer-key", type=Path)
    parser.add_argument("--review-template", type=Path)
    parser.add_argument("--topology-row", action="store_true")
    parser.add_argument("--focus-differences", action="store_true")
    parser.add_argument("--focus-color")
    parser.add_argument("--topology-report", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()

    candidate_rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    rows_by_unicode = {row["unicode"]: row for row in candidate_rows}
    result_rows = json.loads(args.results.read_text("utf-8"))["results"]
    selected = {}
    for result in result_rows:
        if args.split and result.get("split") != args.split:
            continue
        if args.only_errors and result["correct"]:
            continue
        selected[(result["unicode"], result["source"])] = result

    records, page_sizes = PDF.parse_pdf_cells(args.bbox_cache)
    records_by_page = defaultdict(list)
    for record in records:
        if (record["unicode"], record["source"]) in selected:
            records_by_page[record["page"]].append(record)

    svg_images = {}
    answer_key = {}
    review_template = {}
    topology_report = {}
    annotations = (
        json.loads(args.annotations.read_text("utf-8")) if args.annotations else {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_number, page_records in sorted(records_by_page.items()):
        with Image.open(args.pages_dir / f"page-{page_number:03d}.pbm") as page:
            for record in page_records:
                result = selected[(record["unicode"], record["source"])]
                row = rows_by_unicode[record["unicode"]]
                ids = [int(value) for value in row["candidates"]]
                panels = [pdf_cell(page, record["bbox"], page_sizes[page_number])]
                full_pdf_crossing_evidence = horizontal_crossing_consensus(panels[0])
                topology_inputs = [panels[0]]
                topology_target_inputs = []
                labels = [f"PDF {record['source']}"]
                candidate_set_warning = bool(result.get("candidateSetIncomplete"))
                for glyph_id in ids:
                    if glyph_id not in svg_images:
                        svg_markup = row["candidateSvgs"][str(glyph_id)]
                        candidate_image = render_svg_review_image(
                            without_review_points(svg_markup)
                            if args.topology_row
                            else svg_markup
                        )
                        svg_images[glyph_id] = (
                            topology_overlay(candidate_image)
                            if args.topology_row
                            else candidate_image
                        )
                    panels.append(ImageOps.contain(svg_images[glyph_id], (256, 256)))
                    topology_svg = (
                        row.get("candidateFocusSvgs", {}).get(str(glyph_id))
                        if args.focus_color
                        else None
                    ) or row["candidateSvgs"][str(glyph_id)]
                    topology_inputs.append(
                        render_svg_review_image(without_review_points(topology_svg))
                    )
                    target_svg = row.get("candidateTopologySvgs", {}).get(
                        str(glyph_id), topology_svg
                    )
                    topology_target_inputs.append(
                        render_svg_review_image(without_review_points(target_svg))
                    )
                    if args.blind:
                        candidate = chr(65 + len(labels) - 1)
                        focus_ids = row.get("candidateFocusIds", {}).get(
                            str(glyph_id), []
                        )
                        focus_label = (
                            f" [component {','.join(map(str, focus_ids))}]"
                            if focus_ids
                            else ""
                        )
                        labels.append(f"Candidate {candidate}{focus_label}")
                    else:
                        flags = []
                        if glyph_id == result["expectedGlyphId"]:
                            flags.append("expected")
                        if glyph_id == result["predictedGlyphId"]:
                            flags.append("predicted")
                        labels.append(f"{glyph_id} {'/'.join(flags)}".strip())
                width = 256 * len(panels)
                height = 580 if args.topology_row else 300
                montage = Image.new("RGB", (width, height), "white")
                draw = ImageDraw.Draw(montage)
                for index, (panel, label) in enumerate(zip(panels, labels)):
                    montage.paste(panel, (index * 256, 36))
                    draw.text((index * 256 + 8, 10), label, fill="black")
                if candidate_set_warning:
                    draw.text(
                        (8, 23),
                        "REVIEW: all candidates may share a wrong component",
                        fill=(194, 65, 12),
                    )
                if args.topology_row:
                    if args.focus_color or args.focus_differences:
                        if args.focus_color:
                            topology_inputs = structure_guided_focus(
                                topology_inputs[0],
                                topology_inputs[1:],
                                args.focus_color,
                                topology_target_inputs,
                            )
                        else:
                            focus_box = discriminative_box(topology_inputs[1:])
                            topology_inputs = [
                                focus_panel(panel, focus_box)
                                for panel in topology_inputs
                            ]
                    draw.text(
                        (8, 306),
                        "Topology markers (also on SVG): red=end, blue=branch/cross, green=degree-2 sharp turn"
                        if args.focus_color
                        else "Topology markers (also on SVG): red=end, blue=branch/cross, green=degree-2 sharp turn"
                        if args.focus_differences
                        else "Topology markers (also on SVG): red=end, blue=branch/cross, green=degree-2 sharp turn",
                        fill="black",
                    )
                    for index, panel in enumerate(topology_inputs):
                        montage.paste(topology_panel(panel), (index * 256, 324))
                codepoint = f"U+{record['unicode']:04X}"
                filename = f"{codepoint}-{record['source']}.png"
                if args.topology_row:
                    topology_report[filename] = {
                        label: topology_signature(panel)
                        for label, panel in zip(labels, topology_inputs)
                    }
                    topology_report[filename]["candidateFocusTopology"] = row.get(
                        "candidateFocusTopology", {}
                    )
                    topology_report[filename]["fullPdfHorizontalCrossingEvidence"] = (
                        full_pdf_crossing_evidence
                    )
                    topology_report[filename]["localTopologyChoice"] = (
                        choose_split_stroke_candidate(
                            full_pdf_crossing_evidence,
                            row.get("candidateFocusTopology", {}),
                        )
                    )
                montage = draw_feature_annotations(
                    montage, annotations.get(filename, []), args.topology_row
                )
                montage.save(args.output_dir / filename)
                expected_index = ids.index(result["expectedGlyphId"])
                predicted_id = result.get("predictedGlyphId")
                prediction = (
                    chr(65 + ids.index(predicted_id))
                    if predicted_id in ids
                    else "abstain"
                )
                answer_key[filename] = {
                    "expected": chr(65 + expected_index),
                    "expectedGlyphId": result["expectedGlyphId"],
                    "algorithmPrediction": prediction,
                    "algorithmGlyphId": predicted_id,
                }
                review_template[filename] = {
                    "choice": "uncertain",
                    "reason": "",
                }
    if args.answer_key:
        args.answer_key.parent.mkdir(parents=True, exist_ok=True)
        args.answer_key.write_text(
            json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.review_template:
        args.review_template.parent.mkdir(parents=True, exist_ok=True)
        args.review_template.write_text(
            json.dumps(review_template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.topology_report:
        args.topology_report.parent.mkdir(parents=True, exist_ok=True)
        args.topology_report.write_text(
            json.dumps(topology_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
