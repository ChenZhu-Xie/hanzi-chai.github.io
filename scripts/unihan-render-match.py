"""Evaluate repository glyph candidates against rasterized Unicode chart cells."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


def render_strokes(strokes: list[dict], size=128) -> np.ndarray:
    return normalize_skeleton(render_raw_strokes(strokes, size) > 24)


def render_stroke_layers(strokes: list[dict], size=128) -> list[np.ndarray]:
    layers = [render_raw_strokes([stroke], size) > 24 for stroke in strokes]
    if not layers:
        return []
    union = np.logical_or.reduce(layers)
    points = np.argwhere(union)
    if points.size == 0:
        return [np.zeros((64, 64), dtype=bool) for _ in layers]
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    target = 54
    scale = min(target / (right - left), target / (bottom - top))
    width = max(1, round((right - left) * scale))
    height = max(1, round((bottom - top) * scale))
    normalized = []
    for layer in layers:
        ink = layer[top:bottom, left:right].astype(np.uint8) * 255
        resized = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((64, 64), dtype=bool)
        y = (64 - height) // 2
        x = (64 - width) // 2
        canvas[y : y + height, x : x + width] = resized > 24
        normalized.append(skeletonize(canvas))
    return normalized


def skeletonize(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton > 0


def normalize_skeleton(binary: np.ndarray) -> np.ndarray:
    points = np.argwhere(binary)
    if points.size == 0:
        return np.zeros((64, 64), dtype=bool)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    ink = binary[top:bottom, left:right].astype(np.uint8) * 255
    target = 54
    scale = min(target / ink.shape[1], target / ink.shape[0])
    resized = cv2.resize(
        ink,
        (
            max(1, round(ink.shape[1] * scale)),
            max(1, round(ink.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    normalized = np.zeros((64, 64), dtype=bool)
    y = (64 - resized.shape[0]) // 2
    x = (64 - resized.shape[1]) // 2
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


def render_svg(svg: str, size=256) -> np.ndarray:
    return normalize_skeleton(render_svg_image(svg, size) < 224)


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
    for scale_x in (0.9, 1.0, 1.1):
        for scale_y in (0.9, 1.0, 1.1):
            for dx in range(-6, 7, 2):
                for dy in range(-6, 7, 2):
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
    for scale_x in np.arange(coarse_x - 0.04, coarse_x + 0.041, 0.02):
        for scale_y in np.arange(coarse_y - 0.04, coarse_y + 0.041, 0.02):
            for dx in range(coarse_dx - 2, coarse_dx + 3):
                for dy in range(coarse_dy - 2, coarse_dy + 3):
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


def unmatched_stroke_indices(strokes: list[dict], other: list[dict]) -> set[int]:
    """Greedily match same-topology strokes by location and return leftovers."""
    unmatched = set(range(len(strokes)))
    used_other: set[int] = set()
    pairs = []
    for left_index, left in enumerate(strokes):
        for right_index, right in enumerate(other):
            if stroke_kind(left) != stroke_kind(right):
                continue
            distance = math.dist(left["start"], right["start"])
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


def confidence_threshold(results: list[dict], precision_target=0.995):
    ordered = sorted(results, key=lambda item: item["margin"], reverse=True)
    correct = 0
    selected = None
    for index, item in enumerate(ordered, start=1):
        correct += item["correct"]
        precision = correct / index
        if precision >= precision_target:
            selected = {
                "minimumMargin": item["margin"],
                "accepted": index,
                "correct": correct,
                "precision": precision,
                "coverage": index / len(results),
            }
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
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer", choices=("centerline", "svg"), default="centerline"
    )
    parser.add_argument("--alignment", choices=("none", "common"), default="none")
    parser.add_argument("--topology-weight", type=float, default=0.0)
    args = parser.parse_args()

    candidate_rows = json.loads(args.candidates.read_text(encoding="utf-8"))["rows"]
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
            stroke_layers[numeric_id] = render_stroke_layers(strokes)
            if args.renderer == "svg":
                svg = row.get("candidateSvgs", {}).get(glyph_id)
                if svg is None:
                    raise RuntimeError(
                        f"candidate {glyph_id} has no SVG; regenerate candidate evidence"
                    )
                rendered[numeric_id] = render_svg(svg)
            else:
                rendered[numeric_id] = render_strokes(strokes)

    results = []
    for page_number, page_records in sorted(records_by_page.items()):
        with Image.open(args.pages_dir / f"page-{page_number:03d}.pbm") as page:
            for record in page_records:
                row = candidates_by_unicode[record["unicode"]]
                expected = row["sourceGlyphs"].get(record["source"])
                if expected is None:
                    continue
                pdf_skeleton = PDF.normalized_skeleton(
                    page, record["bbox"], page_sizes[page_number]
                )
                candidate_ids = [int(glyph_id) for glyph_id in row["candidates"]]
                candidate_images = [rendered[glyph_id] for glyph_id in candidate_ids]
                alignment = None
                if args.alignment == "common":
                    candidate_images, alignment = align_candidates_to_pdf(
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
                scores = [
                    visual + args.topology_weight * topology
                    for visual, topology in zip(visual_scores, topology_scores)
                ]
                distances = sorted(zip(scores, candidate_ids))
                best_distance, best_id = distances[0]
                second_distance = distances[1][0]
                results.append(
                    {
                        "unicode": record["unicode"],
                        "source": record["source"],
                        "split": row.get("split"),
                        "familyKeys": row.get("familyKeys", []),
                        "expectedGlyphId": expected,
                        "predictedGlyphId": best_id,
                        "correct": best_id == expected,
                        "bestDistance": round(best_distance, 4),
                        "secondDistance": round(second_distance, 4),
                        "margin": round(second_distance - best_distance, 4),
                        "alignment": alignment,
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
            "method": f"candidate-difference-region repository {args.renderer} render with {args.alignment} alignment, topology weight {args.topology_weight}",
            **summary,
            # This descriptive value sees both splits and must never be used as
            # an apply threshold. Only heldOutAcceptance tests train calibration.
            "inSampleHighConfidence": confidence_threshold(results),
            "automaticWriteEnabled": False,
            "splits": {
                "train": summarize_results(train_results),
                "test": summarize_results(test_results),
                "calibration": calibration,
                "heldOutAcceptance": held_out_acceptance,
                "topologyGate": topology_gate,
                "candidateAbsence": candidate_absence_probe(train_results),
            },
            "fontOutlinesExtracted": False,
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
