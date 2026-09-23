"""Blindly evaluate leaf-isolated PDF topology on reviewed sibling families."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def load_montage():
    path = Path(__file__).with_name("unihan-render-montage.py")
    spec = importlib.util.spec_from_file_location("unihan_render_montage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONTAGE = load_montage()


def distance(left: dict, right: dict, weights: tuple[float, float, float]):
    return (
        weights[0] * abs(left["components"] - right["components"])
        + weights[1] * abs(left["endpoints"] - right["endpoints"])
        + weights[2] * abs(left["junctions"] - right["junctions"])
    )


def score_rows(rows: list[dict], weights: tuple[float, float, float]):
    scored = []
    for row in rows:
        pdf = row["pdfSignature"]
        candidate_signatures = row["candidateSignatures"]
        exact_components = [
            glyph_id
            for glyph_id, signature in candidate_signatures.items()
            if signature["components"] == pdf["components"]
        ]
        eligible = exact_components or list(candidate_signatures)
        distances = sorted(
            (distance(pdf, candidate_signatures[glyph_id], weights), glyph_id)
            for glyph_id in eligible
        )
        best_distance, predicted = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else best_distance
        tie = len(distances) > 1 and second_distance == best_distance
        scored.append(
            {
                **row,
                "predictedGlyphId": None if tie else int(predicted),
                "correct": not tie and int(predicted) == row["expectedGlyphId"],
                "abstained": tie or pdf["components"] == 0,
                "bestDistance": best_distance,
                "margin": second_distance - best_distance,
                "candidateDistances": {
                    glyph_id: distance(pdf, signature, weights)
                    for glyph_id, signature in candidate_signatures.items()
                },
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
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
        with Image.open(args.pages_dir / f"page-{page_number:03d}.pbm") as page:
            for record in page_records:
                row = rows_by_unicode[record["unicode"]]
                candidate_ids = [int(value) for value in row["candidates"]]
                candidate_images = [
                    MONTAGE.render_svg_review_image(
                        row["candidateFocusSvgs"][str(glyph_id)]
                    )
                    for glyph_id in candidate_ids
                ]
                pdf = MONTAGE.pdf_cell(
                    page, record["bbox"], page_sizes[page_number]
                )
                focused = MONTAGE.structure_guided_focus(
                    pdf, candidate_images, "2563eb"
                )
                signatures = [MONTAGE.topology_signature(image) for image in focused]
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
                            for glyph_id, signature in zip(candidate_ids, signatures[1:])
                        },
                    }
                )

    train_raw = [row for row in raw if row["split"] == "train"]
    weight_candidates = [
        (component, endpoint, junction)
        for component in (1.0, 2.0, 4.0, 8.0)
        for endpoint in (0.5, 1.0, 2.0)
        for junction in (0.0, 0.25, 0.5, 1.0)
    ]
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
            },
            "train": summary(train),
            "test": summary(test),
            "calibration": threshold,
            "heldOutAcceptance": summary(accepted_test),
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
