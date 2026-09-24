"""Evaluate candidate-conditioned recursive leaf evidence without writing data.

Each candidate is recursively split into terminal component occurrences. The
Unicode chart ink is aligned once to that candidate and attributed to its
nearest leaf. Every leaf is then compared independently, so contacts between
different components cannot manufacture topology evidence.
"""

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


def mask_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    left_skeleton = MONTAGE.MATCHER.skeletonize(left)
    right_skeleton = MONTAGE.MATCHER.skeletonize(right)
    if not left_skeleton.any() and not right_skeleton.any():
        return 0.0
    if not left_skeleton.any() or not right_skeleton.any():
        return 1.0
    left_distance = MONTAGE.PDF.distance_transform_edt(~left_skeleton)
    right_distance = MONTAGE.PDF.distance_transform_edt(~right_skeleton)
    diagonal = math.hypot(*left.shape)
    return float(
        (
            left_distance[right_skeleton].mean()
            + right_distance[left_skeleton].mean()
        )
        / (2 * diagonal)
    )


def mask_signature(mask: np.ndarray) -> dict:
    image = Image.fromarray((~mask).astype(np.uint8) * 255, mode="L")
    return MONTAGE.topology_signature(image)


def leaf_features(source: np.ndarray, candidate: np.ndarray) -> dict:
    source_signature = mask_signature(source)
    candidate_signature = mask_signature(candidate)
    return {
        "chamfer": mask_chamfer(source, candidate),
        "componentDelta": abs(
            source_signature["components"] - candidate_signature["components"]
        ),
        "endpointDelta": abs(
            source_signature["endpoints"] - candidate_signature["endpoints"]
        ),
        "junctionDelta": abs(
            source_signature["junctions"] - candidate_signature["junctions"]
        ),
        "orientationDelta": sum(
            abs(left - right)
            for left, right in zip(
                source_signature["orientationHistogram"],
                candidate_signature["orientationHistogram"],
            )
        ),
    }


def aggregate_candidate_features(
    pdf_mask: np.ndarray,
    attributed: list[np.ndarray],
    candidate_leaves: list[np.ndarray],
) -> dict:
    leaves = [
        leaf_features(source, candidate)
        for source, candidate in zip(attributed, candidate_leaves)
    ]
    weights = np.array([max(1, int(mask.sum())) for mask in candidate_leaves])
    weights = weights / weights.sum()
    attributed_union = np.logical_or.reduce(attributed)
    return {
        "leafChamfer": float(
            sum(weight * leaf["chamfer"] for weight, leaf in zip(weights, leaves))
        ),
        "componentDelta": float(
            sum(
                weight * leaf["componentDelta"]
                for weight, leaf in zip(weights, leaves)
            )
        ),
        "endpointDelta": float(
            sum(
                weight * leaf["endpointDelta"]
                for weight, leaf in zip(weights, leaves)
            )
        ),
        "junctionDelta": float(
            sum(
                weight * leaf["junctionDelta"]
                for weight, leaf in zip(weights, leaves)
            )
        ),
        "orientationDelta": float(
            sum(
                weight * leaf["orientationDelta"]
                for weight, leaf in zip(weights, leaves)
            )
        ),
        "unassignedRatio": 1
        - float(attributed_union.sum()) / max(1, int(pdf_mask.sum())),
        "leaves": leaves,
    }


FEATURE_NAMES = (
    "leafChamfer",
    "componentDelta",
    "endpointDelta",
    "junctionDelta",
    "orientationDelta",
    "unassignedRatio",
)


def weighted_score(features: dict, weights: tuple[float, ...]) -> float:
    return sum(features[name] * weight for name, weight in zip(FEATURE_NAMES, weights))


def score_rows(rows: list[dict], weights: tuple[float, ...]) -> list[dict]:
    output = []
    for row in rows:
        ranked = sorted(
            (
                weighted_score(features, weights),
                int(glyph_id),
            )
            for glyph_id, features in row["candidateFeatures"].items()
        )
        best_score, predicted = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else best_score
        tied = len(ranked) > 1 and math.isclose(best_score, second_score)
        output.append(
            {
                **row,
                "predictedGlyphId": None if tied else predicted,
                "correct": not tied and predicted == row["expectedGlyphId"],
                "abstained": tied,
                "bestScore": best_score,
                "margin": second_score - best_score,
                "candidateScores": {
                    glyph_id: weighted_score(features, weights)
                    for glyph_id, features in row["candidateFeatures"].items()
                },
            }
        )
    return output


def report(rows: list[dict]) -> dict:
    attempted = [row for row in rows if not row["abstained"]]
    correct = sum(row["correct"] for row in attempted)
    return {
        "evaluated": len(rows),
        "attempted": len(attempted),
        "correct": correct,
        "accuracyWhenAttempted": correct / len(attempted) if attempted else None,
        "coverage": len(attempted) / len(rows) if rows else None,
    }


def acceptance_report(all_rows: list[dict], accepted: list[dict]) -> dict:
    correct = sum(row["correct"] for row in accepted)
    return {
        "evaluated": len(all_rows),
        "accepted": len(accepted),
        "correct": correct,
        "precision": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(all_rows) if all_rows else None,
    }


def compare_with_baseline(rows: list[dict], baseline_payload: dict) -> dict:
    baseline = {
        (row["unicode"], row["source"]): row
        for row in baseline_payload.get("results", [])
    }
    compared = [
        (row, baseline[(row["unicode"], row["source"])])
        for row in rows
        if (row["unicode"], row["source"]) in baseline
    ]
    return {
        split: {
            "evaluated": sum(row["split"] == split for row, _base in compared),
            "rescues": sum(
                row["split"] == split and row["correct"] and not base["correct"]
                for row, base in compared
            ),
            "breaks": sum(
                row["split"] == split and not row["correct"] and base["correct"]
                for row, base in compared
            ),
        }
        for split in ("train", "test")
    }


def calibrate_safe_margin(train: list[dict]) -> dict | None:
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
            selected = {"minimumMargin": row["margin"], "accepted": index}
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pages-dir", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-size", type=int, default=128)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="compare against the existing scorer; never changes predictions",
    )
    parser.add_argument(
        "--reuse-features",
        type=Path,
        help="reuse candidateFeatures from a prior run when only scoring changes",
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

    candidate_cache = {}
    raw = []
    if args.reuse_features:
        reused = json.loads(args.reuse_features.read_text("utf-8"))
        reused_size = reused.get("metadata", {}).get("evidenceSize")
        if reused_size != args.evidence_size:
            raise ValueError(
                f"feature evidence size is {reused_size}, expected {args.evidence_size}"
            )
        raw = [
            {
                "unicode": row["unicode"],
                "source": row["source"],
                "split": row["split"],
                "familyKeys": row["familyKeys"],
                "expectedGlyphId": row["expectedGlyphId"],
                "candidateFeatures": row["candidateFeatures"],
                "alignments": row["alignments"],
            }
            for row in reused["results"]
        ]
        records_by_page = {}
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
                pdf_image = (
                    MONTAGE.pdf_vector_cell(
                        page_svg,
                        record["bbox"],
                        size=args.evidence_size,
                        glyph_only=True,
                    )
                    if page_svg is not None
                    else MONTAGE.pdf_cell(
                        page,
                        record["bbox"],
                        page_sizes[page_number],
                        size=args.evidence_size,
                    )
                )
                pdf_mask = np.asarray(pdf_image.convert("L")) < 224
                features_by_candidate = {}
                alignments = {}
                for glyph_id in row["candidates"]:
                    key = (record["unicode"], glyph_id)
                    cached = candidate_cache.get(key)
                    if cached is None:
                        scoring_svg = row.get("candidateScoringSvgs", {}).get(
                            glyph_id,
                            MONTAGE.without_review_points(
                                row["candidateSvgs"][glyph_id]
                            ),
                        )
                        candidate_mask = np.asarray(
                            MONTAGE.render_svg_review_image(
                                scoring_svg, size=args.evidence_size
                            ).convert("L")
                        ) < 224
                        leaf_entries = row["candidateLeafSvgs"][glyph_id]
                        leaf_masks = [
                            np.asarray(
                                MONTAGE.render_svg_review_image(
                                    entry["svg"], size=args.evidence_size
                                ).convert("L")
                            )
                            < 224
                            for entry in leaf_entries
                        ]
                        cached = (candidate_mask, leaf_masks, leaf_entries)
                        candidate_cache[key] = cached
                    candidate_mask, leaf_masks, leaf_entries = cached
                    _aligned, alignment = MONTAGE.MATCHER.align_candidates_to_pdf(
                        pdf_mask, [candidate_mask]
                    )
                    matrix = MONTAGE.MATCHER.alignment_matrix(
                        alignment, pdf_mask.shape
                    )
                    aligned_leaves = [
                        MONTAGE.MATCHER.warp_mask(mask, matrix)
                        for mask in leaf_masks
                    ]
                    attributed = MONTAGE.attribute_pdf_to_leaf_masks(
                        pdf_mask, aligned_leaves
                    )
                    features = aggregate_candidate_features(
                        pdf_mask, attributed, aligned_leaves
                    )
                    features["leafIds"] = [
                        entry["leafId"] for entry in leaf_entries
                    ]
                    features_by_candidate[glyph_id] = features
                    alignments[glyph_id] = alignment
                raw.append(
                    {
                        "unicode": record["unicode"],
                        "source": record["source"],
                        "split": row["split"],
                        "familyKeys": row["familyKeys"],
                        "expectedGlyphId": int(row["sourceGlyphs"][record["source"]]),
                        "candidateFeatures": features_by_candidate,
                        "alignments": alignments,
                    }
                )

    train_raw = [row for row in raw if row["split"] == "train"]
    weight_candidates = [
        (1.0, component, endpoint, junction, orientation, unassigned)
        for component in (0.0, 0.25, 0.5, 1.0)
        for endpoint in (0.0, 0.1, 0.25, 0.5)
        for junction in (0.0, 0.1, 0.25, 0.5)
        for orientation in (0.0, 0.25, 0.5, 1.0)
        for unassigned in (0.0, 0.5, 1.0, 2.0)
    ]
    ranked_weights = []
    for weights in weight_candidates:
        scored = score_rows(train_raw, weights)
        summary = report(scored)
        ranked_weights.append(
            (
                summary["correct"],
                summary["accuracyWhenAttempted"] or 0,
                -sum(weights),
                weights,
            )
        )
    weights = max(ranked_weights)[-1]
    results = score_rows(raw, weights)
    train = [row for row in results if row["split"] == "train"]
    test = [row for row in results if row["split"] == "test"]
    threshold = calibrate_safe_margin(train)
    for row in results:
        row["safeAccepted"] = bool(
            threshold
            and not row["abstained"]
            and row["margin"] >= threshold["minimumMargin"]
        )
    safe_test = [row for row in test if row["safeAccepted"]]
    baseline_comparison = (
        compare_with_baseline(
            results, json.loads(args.baseline.read_text("utf-8"))
        )
        if args.baseline
        else None
    )
    adoption_decision = (
        "reject-as-automatic-evidence"
        if threshold is None
        or (
            baseline_comparison
            and baseline_comparison["test"]["breaks"]
            >= baseline_comparison["test"]["rescues"]
        )
        else "requires-maintainer-review"
    )
    output = {
        "metadata": {
            "method": "candidate-conditioned recursive terminal-leaf attribution",
            "weights": dict(zip(FEATURE_NAMES, weights)),
            "all": report(results),
            "train": report(train),
            "test": report(test),
            "safeThreshold": threshold,
            "safeHeldOut": acceptance_report(test, safe_test),
            "baselineComparison": baseline_comparison,
            "adoptionDecision": adoption_decision,
            "adoptionReason": (
                "candidate-conditioned attribution can make a wrong candidate "
                "explain its own nearby ink; retain it for human review only"
            ),
            "automaticWriteEnabled": False,
            "evidenceSize": args.evidence_size,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
