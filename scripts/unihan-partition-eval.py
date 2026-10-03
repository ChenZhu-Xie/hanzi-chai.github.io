"""Evaluate candidate-driven PDF ink partitions against exported human truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


def load_transfer_module():
    path = Path(__file__).with_name("unihan-stroke-transfer.py")
    spec = importlib.util.spec_from_file_location("unihan_partition_eval_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = load_transfer_module()


def union_by_component(strokes: list[dict], masks: list[np.ndarray]) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for stroke, mask in zip(strokes, masks):
        component_id = int(stroke["componentId"])
        output.setdefault(component_id, np.zeros_like(mask))
        output[component_id] |= mask
    return output


def mask_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict:
    intersection = int(np.logical_and(predicted, expected).sum())
    union = int(np.logical_or(predicted, expected).sum())
    predicted_pixels = int(predicted.sum())
    expected_pixels = int(expected.sum())
    return {
        "intersection": intersection,
        "union": union,
        "iou": round(intersection / union, 6) if union else 1.0,
        "precision": round(intersection / predicted_pixels, 6) if predicted_pixels else 0.0,
        "recall": round(intersection / expected_pixels, 6) if expected_pixels else 0.0,
    }


def compare_partitions(
    target: np.ndarray,
    predicted_strokes: list[dict],
    predicted_masks: list[np.ndarray],
    truth_strokes: list[dict],
    truth_masks: list[np.ndarray],
) -> dict:
    predicted_components = union_by_component(predicted_strokes, predicted_masks)
    truth_components = union_by_component(truth_strokes, truth_masks)
    component_ids = sorted(truth_components)
    component_scores = {
        str(component_id): mask_metrics(
            predicted_components.get(component_id, np.zeros_like(target)),
            truth_components[component_id],
        )
        for component_id in component_ids
    }
    stroke_scores = [
        mask_metrics(predicted, expected)
        for predicted, expected in zip(predicted_masks, truth_masks)
    ]
    confusion = {
        str(predicted_id): {
            str(expected_id): int(
                np.logical_and(predicted_mask, expected_mask).sum()
            )
            for expected_id, expected_mask in truth_components.items()
        }
        for predicted_id, predicted_mask in predicted_components.items()
    }
    correct_pixels = sum(
        int(
            np.logical_and(
                predicted_components.get(component_id, np.zeros_like(target)),
                truth_components[component_id],
            ).sum()
        )
        for component_id in component_ids
    )
    return {
        "componentMacroIoU": round(
            sum(item["iou"] for item in component_scores.values()) / len(component_scores),
            6,
        ),
        "strokeMacroIoU": round(
            sum(item["iou"] for item in stroke_scores) / len(stroke_scores), 6
        ),
        "inkComponentAccuracy": round(correct_pixels / int(target.sum()), 6),
        "components": component_scores,
        "strokes": stroke_scores,
        "componentConfusionPixels": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reference-annotations", type=Path, nargs="*", default=[])
    parser.add_argument("--unicode", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--glyph-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=512)
    args = parser.parse_args()

    codepoint = int(args.unicode.removeprefix("U+").removeprefix("u+"), 16)
    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    reference_candidates = TRANSFER.candidate_catalog(rows)
    row = next(item for item in rows if int(item["unicode"]) == codepoint)
    records, _sizes = TRANSFER.PDF.parse_pdf_cells(args.bbox_cache)
    record = next(
        item for item in records
        if item["unicode"] == codepoint and item["source"] == args.source
    )
    page_svg = TRANSFER.MATCHER.load_pdf_page_svg(args.pdf, record["page"])
    source_mask = TRANSFER.MATCHER.render_pdf_vector_cell(
        page_svg,
        TRANSFER.MATCHER.chart_glyph_bbox(record["bbox"]),
        size=args.canvas,
    ) < 224
    target = TRANSFER.normalize_target(source_mask, args.canvas)
    candidate = TRANSFER.candidate_strokes(row, args.glyph_id)

    truth_strokes, truth_document, corrections = TRANSFER.load_human_annotations(
        args.annotations,
        codepoint=codepoint,
        source=args.source,
        glyph_id=args.glyph_id,
        candidate=candidate,
        canvas=args.canvas,
    )
    truth_centerlines, _truth_distances = TRANSFER.snap_centerlines(
        [stroke["points"] for stroke in truth_strokes],
        target,
        max_distance=args.canvas * 0.045,
    )
    truth_masks, _truth_ambiguous, truth_metrics = TRANSFER.partition_human_truth(
        target,
        truth_centerlines,
        truth_strokes,
        truth_document["annotations"],
        args.canvas,
    )

    adaptive_strokes, adaptive_choices = TRANSFER.select_component_templates_by_alignment(
        candidate,
        args.reference_annotations,
        target,
        args.canvas,
        target_unicode=f"U+{codepoint:04X}",
        target_source=args.source,
        reference_candidates=reference_candidates,
    )
    global_strokes, global_search_result = TRANSFER.search_component_templates_globally(
        candidate,
        args.reference_annotations,
        target,
        args.canvas,
        target_unicode=f"U+{codepoint:04X}",
        target_source=args.source,
        reference_candidates=reference_candidates,
    )

    modes = []
    for name, template_mode, snap_mode in (
        ("candidate-coherent", "candidate", "coherent"),
        ("candidate-graph", "candidate", "graph"),
        ("verified-transfer-coherent", "first-reference", "coherent"),
        ("verified-transfer-graph", "first-reference", "graph"),
        ("adaptive-transfer-coherent", "adaptive", "coherent"),
        ("adaptive-transfer-graph", "adaptive", "graph"),
        ("global-transfer-coherent", "global", "coherent"),
        ("global-transfer-graph", "global", "graph"),
        ("global-transfer-corridor", "global", "corridor"),
        ("global-transfer-sequential", "global", "sequential"),
        ("global-transfer-graph-hierarchical", "global", "graph"),
        ("global-transfer-sequential-hierarchical", "global", "sequential"),
    ):
        if template_mode == "global":
            strokes = global_strokes
            global_search = global_search_result
            transferred = []
            template_choices = []
        elif template_mode == "adaptive":
            strokes = adaptive_strokes
            template_choices = adaptive_choices
            transferred = []
            global_search = None
        else:
            strokes, transferred = TRANSFER.transfer_verified_component_strokes(
                candidate,
                args.reference_annotations if template_mode == "first-reference" else [],
                target_unicode=f"U+{codepoint:04X}",
                target_source=args.source,
                reference_candidates=reference_candidates,
            )
            template_choices = []
            global_search = None
        fitted = TRANSFER.fit_centerlines(strokes, args.canvas)
        sequential_search = None
        if snap_mode == "sequential":
            centerlines, _distances, sequential_search = (
                TRANSFER.snap_centerlines_sequentially(strokes, fitted, target)
            )
        elif snap_mode in {"graph", "corridor"}:
            centerlines, _distances = TRANSFER.snap_centerlines_on_skeleton_graph(
                fitted,
                target,
                corridor_ratio=0.05 if snap_mode == "corridor" else None,
            )
        else:
            centerlines, _distances = TRANSFER.snap_centerlines(
                fitted, target, max_distance=args.canvas * 0.08
            )
        if name.endswith("-hierarchical"):
            masks, ambiguous, partition_metrics = (
                TRANSFER.partition_components_then_strokes(
                    target, centerlines, strokes
                )
            )
        else:
            masks, ambiguous, partition_metrics = TRANSFER.partition_strokes(
                target, centerlines
            )
        reconstruction = TRANSFER.candidate_alignment_metrics(
            strokes, target, args.canvas
        )
        reconstruction.pop("snapped", None)
        component_count = []
        for mask in masks:
            count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            component_count.append(max(0, int(count) - 1))
        modes.append(
            {
                "name": name,
                "strokeSequenceProfile": TRANSFER.stroke_sequence_profile(strokes),
                "transferredComponents": transferred,
                "templateChoices": template_choices,
                "globalTemplateSearch": global_search,
                "sequentialStrokeSearch": sequential_search,
                "ambiguousRatio": partition_metrics["ambiguousRatio"],
                "totalInkFragments": sum(component_count),
                "reconstruction": reconstruction,
                **compare_partitions(
                    target, strokes, masks, truth_strokes, truth_masks
                ),
            }
        )

    payload = {
        "case": f"U+{codepoint:04X}-{args.source}",
        "glyphId": args.glyph_id,
        "annotationLabelCorrections": corrections,
        "truth": truth_metrics,
        "modes": modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
