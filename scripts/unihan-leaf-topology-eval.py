"""Blindly evaluate leaf-isolated PDF topology on reviewed sibling families."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_montage():
    path = Path(__file__).with_name("unihan-render-montage.py")
    spec = importlib.util.spec_from_file_location("unihan_render_montage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONTAGE = load_montage()


def point_set_distance(left: list[list[float]], right: list[list[float]]):
    """Symmetric Chamfer distance for normalized topology landmarks."""
    if not left and not right:
        return 0.0
    if not left or not right:
        return math.sqrt(2)

    def directed(source, target):
        return sum(
            min(math.dist(point, candidate) for candidate in target) for point in source
        ) / len(source)

    return (directed(left, right) + directed(right, left)) / 2


def raster_skeleton_distance(left: Image.Image, right: Image.Image) -> float:
    """Symmetric full-skeleton distance after scale/translation normalization."""
    left_mask = MONTAGE.MATCHER.normalize_skeleton(np.asarray(left.convert("L")) < 224)
    right_mask = MONTAGE.MATCHER.normalize_skeleton(
        np.asarray(right.convert("L")) < 224
    )
    if not left_mask.any() and not right_mask.any():
        return 0.0
    if not left_mask.any() or not right_mask.any():
        return 1.0
    left_distance = MONTAGE.PDF.distance_transform_edt(~left_mask)
    right_distance = MONTAGE.PDF.distance_transform_edt(~right_mask)
    diagonal = math.hypot(*left_mask.shape)
    return float(
        (left_distance[right_mask].mean() + right_distance[left_mask].mean())
        / (2 * diagonal)
    )


def local_difference_distances(
    pdf_mask: np.ndarray, candidate_masks: list[np.ndarray]
) -> list[float]:
    """Compare only sibling-specific strokes, in their shared parent coordinates.

    Candidate-to-PDF distance is dominant so an adjacent parent stroke accidentally
    attributed to the leaf cannot win merely by adding ink. A small reverse term
    still distinguishes an absent dot/short stroke from a present one.
    """
    pdf = MONTAGE.MATCHER.skeletonize(pdf_mask)
    candidates = [MONTAGE.MATCHER.skeletonize(mask) for mask in candidate_masks]
    if not pdf.any():
        return [1.0 for _ in candidates]
    pdf_distance = MONTAGE.PDF.distance_transform_edt(~pdf)
    candidate_union = np.logical_or.reduce(candidates)
    union_distance = MONTAGE.PDF.distance_transform_edt(~candidate_union)
    nearby_pdf = pdf & (union_distance <= 7)
    scale = math.hypot(*pdf.shape)
    scores = []
    for index, candidate in enumerate(candidates):
        others = [item for other, item in enumerate(candidates) if other != index]
        other_union = (
            np.logical_or.reduce(others) if others else np.zeros_like(candidate)
        )
        other_distance = MONTAGE.PDF.distance_transform_edt(~other_union)
        distinctive = candidate & (other_distance > 3)
        if not distinctive.any():
            distinctive = candidate
        region = MONTAGE.cv2.dilate(
            distinctive.astype(np.uint8), np.ones((13, 13), np.uint8)
        ).astype(bool)
        candidate_points = candidate & region
        forward = (
            float(pdf_distance[candidate_points].mean())
            if candidate_points.any()
            else scale
        )
        local_pdf = nearby_pdf & region
        if local_pdf.any() and candidate.any():
            candidate_distance = MONTAGE.PDF.distance_transform_edt(~candidate)
            reverse = float(np.minimum(candidate_distance[local_pdf], 8).mean())
        else:
            reverse = 8.0
        scores.append((forward + 0.25 * reverse) / scale)
    return scores


def distance_features(
    left: dict, right: dict, raster_distance: float, local_distance: float
) -> tuple[float, ...]:
    return (
        abs(left["components"] - right["components"]),
        abs(left["endpoints"] - right["endpoints"]),
        abs(left["junctions"] - right["junctions"]),
        (
            point_set_distance(left["endpointPositions"], right["endpointPositions"])
            + point_set_distance(left["junctionPositions"], right["junctionPositions"])
        ),
        point_set_distance(left["skeletonPositions"], right["skeletonPositions"]),
        raster_distance,
        local_distance,
        sum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left["orientationHistogram"], right["orientationHistogram"]
            )
        ),
    )


def weighted_distance(features: tuple[float, ...], weights: tuple[float, ...]):
    return sum(feature * weight for feature, weight in zip(features, weights))


def prepare_distances(rows: list[dict]):
    return [
        {
            **row,
            "candidateFeatureDistances": {
                glyph_id: distance_features(
                    row["pdfSignature"],
                    signature,
                    row["candidateRasterDistances"][glyph_id],
                    row["candidateLocalDistances"][glyph_id],
                )
                for glyph_id, signature in row["candidateSignatures"].items()
            },
        }
        for row in rows
    ]


def score_rows(rows: list[dict], weights: tuple[float, ...]):
    scored = []
    for row in rows:
        pdf = row["pdfSignature"]
        candidate_signatures = row["candidateSignatures"]
        distances = sorted(
            (
                weighted_distance(row["candidateFeatureDistances"][glyph_id], weights),
                glyph_id,
            )
            for glyph_id in candidate_signatures
        )
        best_distance, predicted = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else best_distance
        tie = len(distances) > 1 and second_distance == best_distance
        exact_component_matches = [
            int(glyph_id)
            for glyph_id, signature in candidate_signatures.items()
            if signature["components"] == pdf["components"]
        ]
        scored.append(
            {
                **row,
                "predictedGlyphId": None if tie else int(predicted),
                "correct": not tie and int(predicted) == row["expectedGlyphId"],
                "abstained": tie or pdf["components"] == 0,
                "bestDistance": best_distance,
                "margin": second_distance - best_distance,
                "candidateDistances": {
                    glyph_id: weighted_distance(
                        row["candidateFeatureDistances"][glyph_id], weights
                    )
                    for glyph_id in candidate_signatures
                },
                "exactComponentPredictionGlyphId": (
                    exact_component_matches[0]
                    if len(exact_component_matches) == 1
                    else None
                ),
                "exactComponentCandidateIds": exact_component_matches,
            }
        )
    return scored


def summary(rows: list[dict]):
    attempted = [row for row in rows if not row["abstained"]]
    correct = sum(row["correct"] for row in attempted)
    return {
        "evaluated": len(rows),
        "attempted": len(attempted),
        "correct": correct,
        "accuracyWhenAttempted": correct / len(attempted) if attempted else None,
        "coverage": len(attempted) / len(rows) if rows else None,
    }


def acceptance_summary(all_rows: list[dict], accepted: list[dict]):
    correct = sum(row["correct"] for row in accepted)
    return {
        "evaluated": len(all_rows),
        "attempted": len(accepted),
        "correct": correct,
        "accuracyWhenAttempted": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(all_rows) if all_rows else None,
    }


def calibrate_threshold(train: list[dict]):
    ordered = sorted(
        (row for row in train if not row["abstained"]),
        key=lambda row: row["margin"],
        reverse=True,
    )
    correct = 0
    selected = None
    for index, row in enumerate(ordered, start=1):
        correct += row["correct"]
        if correct == index:
            selected = {
                "minimumMargin": row["margin"],
                "accepted": index,
                "precision": 1.0,
            }
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-weights",
        action="store_true",
        help="skip calibration when only categorical subtree topology is needed",
    )
    args = parser.parse_args()
    if args.pdf is None and args.pages_dir is None:
        parser.error("one of --pdf or --pages-dir is required")

    payload = json.loads(args.candidates.read_text("utf-8"))
    rows_by_unicode = {row["unicode"]: row for row in payload["rows"]}
    records, page_sizes = MONTAGE.PDF.parse_pdf_cells(args.bbox_cache)
    records_by_page = defaultdict(list)
    for record in records:
        row = rows_by_unicode.get(record["unicode"])
        if row and record["source"] in row["sourceGlyphs"]:
            records_by_page[record["page"]].append(record)

    raw = []
    for page_number, page_records in sorted(records_by_page.items()):
        page_svg = (
            MONTAGE.MATCHER.load_pdf_page_svg(args.pdf, page_number)
            if args.pdf is not None
            else None
        )
        page_context = (
            contextlib.nullcontext(None)
            if page_svg is not None
            else Image.open(args.pages_dir / f"page-{page_number:03d}.pbm")
        )
        with page_context as page:
            for record in page_records:
                row = rows_by_unicode[record["unicode"]]
                candidate_ids = [int(value) for value in row["candidates"]]
                candidate_images = [
                    MONTAGE.render_svg_review_image(
                        row["candidateFocusSvgs"][str(glyph_id)]
                    )
                    for glyph_id in candidate_ids
                ]
                candidate_target_images = [
                    MONTAGE.render_svg_review_image(
                        row["candidateTopologySvgs"][str(glyph_id)]
                    )
                    for glyph_id in candidate_ids
                ]
                pdf = (
                    MONTAGE.pdf_vector_cell(
                        page_svg, record["bbox"], glyph_only=True
                    )
                    if page_svg is not None
                    else MONTAGE.pdf_cell(
                        page, record["bbox"], page_sizes[page_number]
                    )
                )
                focused_masks = MONTAGE.target_window_focus_masks(
                    pdf, candidate_images, "f59e0b", candidate_target_images
                )
                focused = [MONTAGE.mask_panel(mask) for mask in focused_masks]
                signatures = [MONTAGE.topology_signature(image) for image in focused]
                raster_distances = [
                    raster_skeleton_distance(focused[0], candidate)
                    for candidate in focused[1:]
                ]
                local_distances = local_difference_distances(
                    focused_masks[0], focused_masks[1:]
                )
                raw.append(
                    {
                        "unicode": record["unicode"],
                        "source": record["source"],
                        "split": row["split"],
                        "familyKeys": row["familyKeys"],
                        "focusLeafIds": row["focusLeafIds"],
                        "expectedGlyphId": int(row["sourceGlyphs"][record["source"]]),
                        "pdfSignature": signatures[0],
                        "candidateSignatures": {
                            str(glyph_id): signature
                            for glyph_id, signature in zip(
                                candidate_ids, signatures[1:]
                            )
                        },
                        "candidateRasterDistances": {
                            str(glyph_id): distance
                            for glyph_id, distance in zip(
                                candidate_ids, raster_distances
                            )
                        },
                        "candidateLocalDistances": {
                            str(glyph_id): distance
                            for glyph_id, distance in zip(
                                candidate_ids, local_distances
                            )
                        },
                    }
                )

    raw = prepare_distances(raw)
    train_raw = [row for row in raw if row["split"] == "train"]
    weight_candidates = [] if args.fixed_weights else [
        (
            component,
            endpoint,
            junction,
            landmark,
            sampled_skeleton,
            raster_skeleton,
            local_difference,
            orientation,
        )
        for component in (1.0, 2.0, 4.0)
        for endpoint in (0.5, 1.0, 2.0)
        for junction in (0.0, 0.25, 0.5, 1.0)
        for landmark in (0.0, 0.5, 1.0, 2.0)
        for sampled_skeleton in (0.0, 0.25, 0.5, 1.0)
        for raster_skeleton in (0.0, 0.5, 1.0, 2.0)
        for local_difference in (0.0, 2.0, 4.0, 8.0)
        for orientation in (0.0, 1.0, 2.0, 4.0)
    ]
    if args.fixed_weights:
        # The topology cascade consumes the categorical exact-component field,
        # which is independent of these descriptive ranking weights.
        weights = (2.0, 1.0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0)
    else:
        ranked = []
        for weights in weight_candidates:
            scored = score_rows(train_raw, weights)
            report = summary(scored)
            ranked.append(
                (
                    report["correct"],
                    report["accuracyWhenAttempted"] or 0,
                    report["coverage"] or 0,
                    -sum(weights),
                    weights,
                )
            )
        weights = max(ranked)[-1]
    scored = score_rows(raw, weights)
    train = [row for row in scored if row["split"] == "train"]
    test = [row for row in scored if row["split"] == "test"]
    threshold = calibrate_threshold(train)
    accepted_test = (
        [
            row
            for row in test
            if not row["abstained"] and row["margin"] >= threshold["minimumMargin"]
        ]
        if threshold
        else []
    )
    result = {
        "metadata": {
            "method": "structure-guided leaf attribution plus raster topology",
            "weights": {
                "components": weights[0],
                "endpoints": weights[1],
                "junctions": weights[2],
                "landmarkPositions": weights[3],
                "skeletonPositions": weights[4],
                "rasterSkeleton": weights[5],
                "localDifference": weights[6],
                "orientationHistogram": weights[7],
            },
            "train": summary(train),
            "test": summary(test),
            "calibration": threshold,
            "heldOutAcceptance": acceptance_summary(test, accepted_test),
            "automaticWriteEnabled": False,
        },
        "results": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
