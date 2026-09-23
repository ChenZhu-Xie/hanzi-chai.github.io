"""Create deterministic PDF-vs-SVG panels for human or multimodal review."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    image.thumbnail((size - 24, size - 24), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas.convert("RGB")


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_number, page_records in sorted(records_by_page.items()):
        with Image.open(args.pages_dir / f"page-{page_number:03d}.pbm") as page:
            for record in page_records:
                result = selected[(record["unicode"], record["source"])]
                row = rows_by_unicode[record["unicode"]]
                ids = [int(value) for value in row["candidates"]]
                panels = [pdf_cell(page, record["bbox"], page_sizes[page_number])]
                labels = [f"PDF {record['source']}"]
                for glyph_id in ids:
                    if glyph_id not in svg_images:
                        raster = MATCHER.render_svg_image(
                            row["candidateSvgs"][str(glyph_id)]
                        )
                        svg_images[glyph_id] = Image.fromarray(raster).convert("RGB")
                    panels.append(ImageOps.contain(svg_images[glyph_id], (256, 256)))
                    if args.blind:
                        labels.append(f"Candidate {chr(65 + len(labels) - 1)}")
                    else:
                        flags = []
                        if glyph_id == result["expectedGlyphId"]:
                            flags.append("expected")
                        if glyph_id == result["predictedGlyphId"]:
                            flags.append("predicted")
                        labels.append(f"{glyph_id} {'/'.join(flags)}".strip())
                width = 256 * len(panels)
                montage = Image.new("RGB", (width, 300), "white")
                draw = ImageDraw.Draw(montage)
                for index, (panel, label) in enumerate(zip(panels, labels)):
                    montage.paste(panel, (index * 256, 36))
                    draw.text((index * 256 + 8, 10), label, fill="black")
                codepoint = f"U+{record['unicode']:04X}"
                filename = f"{codepoint}-{record['source']}.png"
                montage.save(args.output_dir / filename)
                expected_index = ids.index(result["expectedGlyphId"])
                answer_key[filename] = {
                    "expected": chr(65 + expected_index),
                    "expectedGlyphId": result["expectedGlyphId"],
                    "algorithmPrediction": chr(
                        65 + ids.index(result["predictedGlyphId"])
                    ),
                    "algorithmGlyphId": result["predictedGlyphId"],
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


if __name__ == "__main__":
    main()
