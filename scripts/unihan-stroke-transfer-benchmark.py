"""Evaluate coherent stroke transfer against the human-reviewed answer key."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


def load_transfer_module():
    path = Path(__file__).with_name("unihan-stroke-transfer.py")
    spec = importlib.util.spec_from_file_location("unihan_stroke_transfer_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = load_transfer_module()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--mode", choices=("global", "focus"), default="global")
    args = parser.parse_args()

    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    rows_by_unicode = {row["unicode"]: row for row in rows}
    answers = json.loads(args.answers.read_text("utf-8"))
    answer_by_key = {}
    for filename, answer in answers.items():
        match = re.fullmatch(r"U\+([0-9A-F]+)-([A-Z]+)\.png", filename)
        if match is not None:
            answer_by_key[(int(match.group(1), 16), match.group(2))] = answer
    records, _sizes = TRANSFER.PDF.parse_pdf_cells(args.bbox_cache)
    records_by_key = {(record["unicode"], record["source"]): record for record in records}
    page_cache: dict[int, str] = {}
    results = []

    cases = [
        (row, source, int(expected))
        for row in rows
        for source, expected in row["sourceGlyphs"].items()
        if str(expected) in row["candidates"]
    ]
    for position, (row, source, expected) in enumerate(cases, start=1):
        codepoint = int(row["unicode"])
        filename = f"U+{codepoint:04X}-{source}.png"
        record = records_by_key[(codepoint, source)]
        if record["page"] not in page_cache:
            page_cache[record["page"]] = TRANSFER.MATCHER.load_pdf_page_svg(
                args.pdf, record["page"]
            )
        source_mask = TRANSFER.MATCHER.render_pdf_vector_cell(
            page_cache[record["page"]],
            TRANSFER.MATCHER.chart_glyph_bbox(record["bbox"]),
            size=args.canvas,
        ) < 224
        target = TRANSFER.normalize_target(source_mask, args.canvas)
        candidate_metrics = {}
        candidate_ids = [int(glyph_id_text) for glyph_id_text in row["candidates"]]
        for glyph_id_text in row["candidates"]:
            glyph_id = int(glyph_id_text)
            strokes = TRANSFER.candidate_strokes(row, glyph_id)
            metrics = TRANSFER.candidate_alignment_metrics(
                strokes,
                target,
                args.canvas,
                focus_component_ids=(
                    set(row["focusLeafIds"]) if args.mode == "focus" else None
                ),
            )
            metrics.pop("snapped")
            candidate_metrics[glyph_id_text] = metrics
        ranking = sorted(
            (metrics["score"], int(glyph_id))
            for glyph_id, metrics in candidate_metrics.items()
        )
        predicted = ranking[0][1]
        expected_index = candidate_ids.index(expected)
        expected_label = chr(ord("A") + expected_index)
        historical_answer = answer_by_key.get((codepoint, source))
        result = {
            "key": filename,
            "unicode": codepoint,
            "source": source,
            "split": row["split"],
            "expectedLabel": expected_label,
            "expectedGlyphId": expected,
            "historicalExpectedGlyphId": (
                int(historical_answer["expectedGlyphId"]) if historical_answer else None
            ),
            "predictedGlyphId": predicted,
            "correct": predicted == expected,
            "margin": round(ranking[1][0] - ranking[0][0], 6) if len(ranking) > 1 else None,
            "candidates": candidate_metrics,
        }
        results.append(result)
        verdict = "OK" if result["correct"] else "WRONG"
        print(
            f"[{position:03d}/{len(cases):03d}] {filename} {verdict} "
            f"expected={expected} predicted={predicted} margin={result['margin']}",
            flush=True,
        )

    summaries = {}
    for split in ("all", "train", "test"):
        selected = results if split == "all" else [row for row in results if row["split"] == split]
        correct = sum(row["correct"] for row in selected)
        summaries[split] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": round(correct / len(selected), 6) if selected else None,
        }
    payload = {
        "mode": args.mode,
        "summary": summaries,
        "errors": [row for row in results if not row["correct"]],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
