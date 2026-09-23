"""Build conservative source-form similarity evidence from a Unicode chart.

The script uses the PDF text layer only to locate codepoint/source cells and a
rasterized page only to compare visible glyphs. It never extracts or reuses an
embedded font or font outline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


REVIEWED_RANGES = (
    (0x4E00, 0x6400),
    (0x7A70, 0x7ACA),
    (0x7CF8, 0x7F35),
    (0x8278, 0x827F),
    (0x8FB6, 0x9090),
    (0x96E8, 0x9761),
)
SOURCE_RE = re.compile(r"^(KP|UK|[GHJKMSTUV])(?=[A-Z0-9_-])")
SOURCE_MAP = {"KP": "N", "UK": "B"}


def reviewed(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in REVIEWED_RANGES)


def source_from_reference(value: str) -> str | None:
    match = SOURCE_RE.match(value)
    if not match:
        return None
    raw = match.group(1)
    return SOURCE_MAP.get(raw, raw)


def ensure_pdf_cache(pdf: Path, bbox: Path, pages: Path, dpi: int) -> None:
    if not bbox.exists():
        subprocess.run(
            ["pdftotext", "-bbox", str(pdf), str(bbox)], check=True
        )
    pages.mkdir(parents=True, exist_ok=True)
    if not next(pages.glob("page-*.pbm"), None):
        subprocess.run(
            [
                "pdftoppm",
                "-mono",
                "-r",
                str(dpi),
                str(pdf),
                str(pages / "page"),
            ],
            check=True,
        )


def parse_pdf_cells(bbox: Path) -> tuple[list[dict], dict[int, tuple[float, float]]]:
    records: list[dict] = []
    page_sizes: dict[int, tuple[float, float]] = {}
    page_number = 0
    for event, element in ET.iterparse(bbox, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "page":
            continue
        page_number += 1
        width = float(element.attrib["width"])
        height = float(element.attrib["height"])
        page_sizes[page_number] = (width, height)
        words = []
        for child in element:
            if child.tag.rsplit("}", 1)[-1] != "word" or child.text is None:
                continue
            words.append(
                {
                    "text": child.text,
                    "x0": float(child.attrib["xMin"]),
                    "y0": float(child.attrib["yMin"]),
                    "x1": float(child.attrib["xMax"]),
                    "y1": float(child.attrib["yMax"]),
                }
            )
        codepoints = []
        for word in words:
            if not re.fullmatch(r"[0-9A-F]{4}", word["text"]):
                continue
            value = int(word["text"], 16)
            if not 0x4E00 <= value <= 0x9FFF or word["y0"] < 80:
                continue
            if not (80 <= word["x0"] <= 115 or 305 <= word["x0"] <= 340):
                continue
            word["codepoint"] = value
            word["column"] = 0 if word["x0"] < width / 2 else 1
            codepoints.append(word)
        codepoints.sort(key=lambda word: (word["column"], word["y0"]))
        for index, codepoint_word in enumerate(codepoints):
            same_column = [
                word
                for word in codepoints
                if word["column"] == codepoint_word["column"]
                and word["y0"] > codepoint_word["y0"]
            ]
            row_end = (
                min(word["y0"] for word in same_column)
                if same_column
                else codepoint_word["y0"] + 34
            )
            row_end = min(row_end, codepoint_word["y0"] + 34)
            half_start = 0 if codepoint_word["column"] == 0 else width / 2
            half_end = width / 2 if codepoint_word["column"] == 0 else width
            character = chr(codepoint_word["codepoint"])
            glyph_words = [
                word
                for word in words
                if word["text"] == character
                and half_start <= (word["x0"] + word["x1"]) / 2 < half_end
                and codepoint_word["y0"] - 3 <= word["y0"] <= row_end
            ]
            source_words = []
            for word in words:
                center = (word["x0"] + word["x1"]) / 2
                source = source_from_reference(word["text"])
                if (
                    source
                    and half_start <= center < half_end
                    and codepoint_word["y0"] + 13 <= word["y0"] <= row_end + 1
                ):
                    source_words.append((source, word))
            seen_sources: set[str] = set()
            for source, source_word in source_words:
                if source in seen_sources:
                    continue
                source_x = (source_word["x0"] + source_word["x1"]) / 2
                candidates = sorted(
                    glyph_words,
                    key=lambda word: abs(
                        (word["x0"] + word["x1"]) / 2 - source_x
                    ),
                )
                if not candidates:
                    continue
                glyph_word = candidates[0]
                glyph_x = (glyph_word["x0"] + glyph_word["x1"]) / 2
                if abs(glyph_x - source_x) > 8:
                    continue
                seen_sources.add(source)
                records.append(
                    {
                        "unicode": codepoint_word["codepoint"],
                        "source": source,
                        "page": page_number,
                        "bbox": [
                            glyph_word["x0"],
                            glyph_word["y0"],
                            glyph_word["x1"],
                            glyph_word["y1"],
                        ],
                    }
                )
        element.clear()
    return records, page_sizes


def normalized_skeleton(
    page: Image.Image,
    bbox: list[float],
    page_size: tuple[float, float],
    size: int = 64,
) -> np.ndarray:
    x_scale = page.width / page_size[0]
    y_scale = page.height / page_size[1]
    margin = 1.5
    x0 = max(0, round((bbox[0] - margin) * x_scale))
    y0 = max(0, round((bbox[1] - margin) * y_scale))
    x1 = min(page.width, round((bbox[2] + margin) * x_scale))
    y1 = min(page.height, round((bbox[3] + margin) * y_scale))
    array = np.asarray(page.crop((x0, y0, x1, y1)).convert("L")) < 128
    points = np.argwhere(array)
    if points.size == 0:
        return np.zeros((size, size), dtype=bool)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    ink = array[top:bottom, left:right].astype(np.uint8) * 255
    target = size - 10
    scale = min(target / ink.shape[1], target / ink.shape[0])
    resized = cv2.resize(
        ink,
        (
            max(1, round(ink.shape[1] * scale)),
            max(1, round(ink.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), dtype=bool)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized > 0
    image = canvas.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton > 0


def chamfer_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not left.any() or not right.any():
        return math.inf
    left_distance = distance_transform_edt(~left)
    right_distance = distance_transform_edt(~right)
    return float(
        (right_distance[left].mean() + left_distance[right].mean()) / 2
    )


def choose_threshold(
    labels: list[tuple[float, bool]],
    precision_target: float = 0.995,
    minimum_true_positive: int = 1,
) -> tuple[float, dict]:
    if not labels:
        raise RuntimeError("No reviewed PDF pairs were available for calibration")
    ordered = sorted(labels, key=lambda item: item[0])
    total_same = sum(same for _, same in ordered)
    total_different = len(ordered) - total_same
    selected = ordered[0][0]
    selected_metrics: dict = {}
    best_recall = -1.0
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        false_negative = total_same - true_positive
        true_negative = total_different - false_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if (
            true_positive >= minimum_true_positive
            and precision >= precision_target
            and recall > best_recall
        ):
            selected = threshold
            best_recall = recall
            selected_metrics = {
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "trueNegative": true_negative,
                "falseNegative": false_negative,
                "precision": precision,
                "recall": recall,
            }
    if not selected_metrics:
        raise RuntimeError(
            "Unable to calibrate a threshold with "
            f"{precision_target:.1%} precision and "
            f"{minimum_true_positive} positive pairs"
        )
    return selected, selected_metrics


def choose_different_threshold(
    labels: list[tuple[float, bool]],
) -> tuple[float, dict]:
    """Choose distance >= threshold as a high-precision different-form edge."""
    if not labels:
        raise RuntimeError("No reviewed PDF pairs were available for calibration")
    ordered = sorted(labels, key=lambda item: item[0], reverse=True)
    total_different = sum(not same for _, same in ordered)
    total_same = len(ordered) - total_different
    selected = ordered[0][0]
    selected_metrics: dict = {}
    best_recall = -1.0
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                false_positive += 1
            else:
                true_positive += 1
            index += 1
        false_negative = total_different - true_positive
        true_negative = total_same - false_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if true_positive > 0 and precision >= 0.995 and recall > best_recall:
            selected = threshold
            best_recall = recall
            selected_metrics = {
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "trueNegative": true_negative,
                "falseNegative": false_negative,
                "precision": precision,
                "recall": recall,
            }
    if not selected_metrics:
        raise RuntimeError(
            "Unable to calibrate a different threshold with 99.5% precision"
        )
    return selected, selected_metrics


def complete_link_clusters(
    sources: list[str],
    distances: dict[tuple[str, str], float],
    thresholds: dict[tuple[str, str], float],
) -> list[list[str]]:
    clusters = [[source] for source in sources]
    while True:
        merge: tuple[int, int] | None = None
        merge_score = math.inf
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                cross = [
                    (
                        distances[tuple(sorted((left, right)))],
                        thresholds.get(tuple(sorted((left, right))))
                    )
                    for left in clusters[left_index]
                    for right in clusters[right_index]
                ]
                if any(threshold is None for _, threshold in cross):
                    continue
                score = max(
                    distance / threshold
                    for distance, threshold in cross
                    if threshold is not None
                )
                if score <= 1 and score < merge_score:
                    merge = (left_index, right_index)
                    merge_score = score
        if merge is None:
            return [sorted(cluster) for cluster in clusters]
        left_index, right_index = merge
        clusters[left_index] = clusters[left_index] + clusters[right_index]
        del clusters[right_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--characters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    ensure_pdf_cache(
        args.pdf, args.bbox_cache, args.pages_dir, args.dpi
    )
    records, page_sizes = parse_pdf_cells(args.bbox_cache)
    characters = json.loads(args.characters.read_text(encoding="utf-8"))
    source_glyph = {
        (character["unicode"], source): entry["id"]
        for character in characters
        for entry in character["glyphs"]
        for source in entry["sources"]
    }
    records_by_page: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        records_by_page[record["page"]].append(record)
    pair_rows: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    calibration: list[tuple[float, bool]] = []
    calibration_by_pair: dict[tuple[str, str], list[tuple[float, bool]]] = (
        defaultdict(list)
    )
    extracted = 0
    for page_number, page_records in sorted(records_by_page.items()):
        path = args.pages_dir / f"page-{page_number:03d}.pbm"
        with Image.open(path) as page:
            skeletons: dict[tuple[int, str], np.ndarray] = {}
            for record in page_records:
                skeletons[(record["unicode"], record["source"])] = (
                    normalized_skeleton(
                        page,
                        record["bbox"],
                        page_sizes[page_number],
                    )
                )
                extracted += 1
        by_unicode: dict[int, list[str]] = defaultdict(list)
        for record in page_records:
            by_unicode[record["unicode"]].append(record["source"])
        for unicode_value, row_sources in by_unicode.items():
            unique_sources = sorted(set(row_sources))
            for left_index, left in enumerate(unique_sources):
                for right in unique_sources[left_index + 1 :]:
                    distance = chamfer_distance(
                        skeletons[(unicode_value, left)],
                        skeletons[(unicode_value, right)],
                    )
                    pair_rows[unicode_value][tuple(sorted((left, right)))] = distance
                    if reviewed(unicode_value):
                        left_id = source_glyph.get((unicode_value, left))
                        right_id = source_glyph.get((unicode_value, right))
                        if left_id is not None and right_id is not None:
                            label = (distance, left_id == right_id)
                            calibration.append(label)
                            calibration_by_pair[tuple(sorted((left, right)))].append(
                                label
                            )
        if page_number % 50 == 0:
            print(
                f"processed page {page_number}/{max(records_by_page)}; "
                f"source glyphs {extracted}",
                flush=True,
            )

    threshold, metrics = choose_threshold(calibration, 0.995)
    likely_threshold, likely_metrics = choose_threshold(calibration, 0.95)
    try:
        different_threshold, different_metrics = choose_different_threshold(
            calibration
        )
    except RuntimeError:
        # Different IRG fonts create pair-specific distance distributions; a
        # global cutoff is deliberately optional and is never substituted for
        # missing per-pair calibration.
        different_threshold, different_metrics = None, None
    pair_thresholds: dict[tuple[str, str], float] = {}
    pair_likely_thresholds: dict[tuple[str, str], float] = {}
    pair_different_thresholds: dict[tuple[str, str], float] = {}
    pair_metrics = {}
    for pair, labels in sorted(calibration_by_pair.items()):
        if sum(not same for _, same in labels) < 10:
            continue
        pair_key = "/".join(pair)
        pair_metrics[pair_key] = {"pairs": len(labels)}
        try:
            pair_threshold, pair_metric = choose_threshold(labels, 0.995, 10)
            pair_thresholds[pair] = pair_threshold
            pair_metrics[pair_key].update(
                {
                    "sameThreshold": pair_threshold,
                    "sameCalibration": pair_metric,
                }
            )
        except RuntimeError:
            pass
        try:
            pair_likely_threshold, pair_likely_metric = choose_threshold(
                labels, 0.95, 10
            )
            pair_likely_thresholds[pair] = pair_likely_threshold
            pair_metrics[pair_key].update(
                {
                    "likelySameThreshold": pair_likely_threshold,
                    "likelySameCalibration": pair_likely_metric,
                }
            )
        except RuntimeError:
            pass
        try:
            pair_different_threshold, pair_different_metric = (
                choose_different_threshold(labels)
            )
        except RuntimeError:
            continue
        pair_different_thresholds[pair] = pair_different_threshold
        pair_metrics[pair_key].update(
            {
                "differentThreshold": pair_different_threshold,
                "differentCalibration": pair_different_metric,
            }
        )
    rows = {}
    for unicode_value, distances in sorted(pair_rows.items()):
        sources = sorted({source for pair in distances for source in pair})
        rows[f"U+{unicode_value:04X}"] = {
            "clusters": complete_link_clusters(
                sources, distances, pair_thresholds
            ),
            "sameEdges": [
                "/".join(pair)
                for pair, distance in sorted(distances.items())
                if pair in pair_thresholds
                and distance <= pair_thresholds[pair]
            ],
            "likelySameEdges": [
                "/".join(pair)
                for pair, distance in sorted(distances.items())
                if pair in pair_likely_thresholds
                and distance <= pair_likely_thresholds[pair]
            ],
            "differentEdges": [
                "/".join(pair)
                for pair, distance in sorted(distances.items())
                if pair in pair_different_thresholds
                and distance >= pair_different_thresholds[pair]
            ],
            "distances": {
                "/".join(pair): round(distance, 4)
                for pair, distance in sorted(distances.items())
            },
        }
    payload = {
        "metadata": {
            "method": "normalized skeleton symmetric chamfer, complete-link clustering",
            "dpi": args.dpi,
            "sameThreshold": threshold,
            "likelySameThreshold": likely_threshold,
            "differentThreshold": different_threshold,
            "calibrationPairs": len(calibration),
            "calibration": metrics,
            "likelySameCalibration": likely_metrics,
            "differentCalibration": different_metrics,
            "sourcePairCalibration": pair_metrics,
            "locatedSourceGlyphs": len(records),
            "extractedSourceGlyphs": extracted,
            "fontOutlinesExtracted": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
