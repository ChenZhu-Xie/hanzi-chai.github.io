"""Evaluate repository glyph candidates against rasterized Unicode chart cells."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


def render_strokes(strokes: list[dict], size=128) -> np.ndarray:
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
    points = np.argwhere(canvas > 24)
    if points.size == 0:
        return np.zeros((64, 64), dtype=bool)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    ink = canvas[top:bottom, left:right]
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
    image = normalized.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton > 0


def discriminative_distance(
    pdf: np.ndarray, candidates: list[np.ndarray]
) -> list[float]:
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = [
        cv2.dilate(candidate.astype(np.uint8), kernel) > 0
        for candidate in candidates
    ]
    common = np.logical_and.reduce(dilated)
    union = np.logical_or.reduce(candidates)
    difference = union & ~common
    roi = cv2.dilate(
        difference.astype(np.uint8), np.ones((11, 11), dtype=np.uint8)
    ) > 0
    if not roi.any():
        return [PDF.chamfer_distance(pdf, candidate) for candidate in candidates]
    pdf_roi = pdf & roi
    return [
        PDF.chamfer_distance(pdf_roi, candidate & roi)
        for candidate in candidates
    ]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = json.loads(
        args.candidates.read_text(encoding="utf-8")
    )["rows"]
    candidates_by_unicode = {row["unicode"]: row for row in candidate_rows}
    records, page_sizes = PDF.parse_pdf_cells(args.bbox_cache)
    records_by_page = defaultdict(list)
    for record in records:
        if record["unicode"] in candidates_by_unicode:
            records_by_page[record["page"]].append(record)
    rendered = {}
    for row in candidate_rows:
        for glyph_id, strokes in row["candidates"].items():
            rendered[int(glyph_id)] = render_strokes(strokes)

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
                scores = discriminative_distance(pdf_skeleton, candidate_images)
                distances = sorted(zip(scores, candidate_ids))
                best_distance, best_id = distances[0]
                second_distance = distances[1][0]
                results.append(
                    {
                        "unicode": record["unicode"],
                        "source": record["source"],
                        "expectedGlyphId": expected,
                        "predictedGlyphId": best_id,
                        "correct": best_id == expected,
                        "bestDistance": round(best_distance, 4),
                        "secondDistance": round(second_distance, 4),
                        "margin": round(second_distance - best_distance, 4),
                        "candidates": [
                            {"id": glyph_id, "distance": round(distance, 4)}
                            for distance, glyph_id in distances
                        ],
                    }
                )
    correct = sum(item["correct"] for item in results)
    threshold = confidence_threshold(results)
    payload = {
        "metadata": {
            "method": "candidate-difference-region repository render to PDF skeleton chamfer",
            "evaluated": len(results),
            "top1Correct": correct,
            "top1Accuracy": correct / len(results),
            "highConfidence": threshold,
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
