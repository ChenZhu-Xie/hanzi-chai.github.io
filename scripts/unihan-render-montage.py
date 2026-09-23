"""Create deterministic PDF-vs-SVG panels for human or multimodal review."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
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
    neighbours = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - skeleton.astype(np.int16)
    endpoints = _point_centers(skeleton & (neighbours == 1))
    junction_mask = skeleton & (neighbours >= 3)
    junction_mask = cv2.dilate(
        junction_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    junctions = _point_centers(junction_mask)
    return endpoints, junctions, skeleton


def topology_panel(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    endpoints, junctions, skeleton = topology_points(gray < 224)
    canvas = Image.new("RGB", image.size, "white")
    array = np.asarray(canvas).copy()
    array[skeleton] = (55, 65, 81)
    canvas = Image.fromarray(array)
    draw = ImageDraw.Draw(canvas)
    for x, y in endpoints:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(220, 38, 38))
    for x, y in junctions:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(2, 132, 199))
    return canvas


def topology_signature(image: Image.Image):
    endpoints, junctions, skeleton = topology_points(
        np.asarray(image.convert("L")) < 224
    )
    components = cv2.connectedComponents(skeleton.astype(np.uint8), 8)[0] - 1
    return {
        "components": int(components),
        "endpoints": len(endpoints),
        "junctions": len(junctions),
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
    pdf_image: Image.Image, candidate_images: list[Image.Image], color: str
):
    """Attribute PDF ink to a leaf after aligning on all invariant leaves."""
    pdf = np.asarray(pdf_image.convert("L")) < 224
    targets = [color_mask(image, color) for image in candidate_images]
    full = [np.asarray(image.convert("L")) < 224 for image in candidate_images]
    kernel = np.ones((5, 5), dtype=np.uint8)
    non_targets = [
        ink
        & ~(
            cv2.dilate(target.astype(np.uint8), kernel, iterations=1).astype(bool)
        )
        for ink, target in zip(full, targets)
    ]
    _aligned_non_targets, alignment = MATCHER.align_candidates_to_pdf(
        pdf, non_targets
    )
    global_matrix = MATCHER.alignment_matrix(alignment, pdf.shape)
    aligned_targets = [
        MATCHER.warp_mask(target, global_matrix) for target in targets
    ]
    aligned_others = [
        MATCHER.warp_mask(other, global_matrix) for other in non_targets
    ]
    target_union = np.logical_or.reduce(aligned_targets)
    other_union = np.logical_or.reduce(aligned_others)
    target_distance = PDF.distance_transform_edt(~target_union)
    other_distance = PDF.distance_transform_edt(~other_union)
    pixel_attribution = pdf & (target_distance <= other_distance) & (target_distance <= 10)
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
    return [mask_panel(attributed), *[mask_panel(target) for target in targets]]


def focus_panel(image: Image.Image, box, size=256):
    crop = image.crop(box)
    crop.thumbnail((size - 16, size - 16), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def without_review_points(svg: str) -> str:
    return re.sub(r"<circle\b[^>]*/>", "", svg)


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_number, page_records in sorted(records_by_page.items()):
        with Image.open(args.pages_dir / f"page-{page_number:03d}.pbm") as page:
            for record in page_records:
                result = selected[(record["unicode"], record["source"])]
                row = rows_by_unicode[record["unicode"]]
                ids = [int(value) for value in row["candidates"]]
                panels = [pdf_cell(page, record["bbox"], page_sizes[page_number])]
                topology_inputs = [panels[0]]
                labels = [f"PDF {record['source']}"]
                for glyph_id in ids:
                    if glyph_id not in svg_images:
                        svg_images[glyph_id] = render_svg_review_image(
                            row["candidateSvgs"][str(glyph_id)],
                        )
                    panels.append(ImageOps.contain(svg_images[glyph_id], (256, 256)))
                    topology_inputs.append(
                        render_svg_review_image(
                            without_review_points(row["candidateSvgs"][str(glyph_id)])
                        )
                    )
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
                height = 580 if args.topology_row else 300
                montage = Image.new("RGB", (width, height), "white")
                draw = ImageDraw.Draw(montage)
                for index, (panel, label) in enumerate(zip(panels, labels)):
                    montage.paste(panel, (index * 256, 36))
                    draw.text((index * 256 + 8, 10), label, fill="black")
                if args.topology_row:
                    if args.focus_color or args.focus_differences:
                        if args.focus_color:
                            topology_inputs = structure_guided_focus(
                                topology_inputs[0],
                                topology_inputs[1:],
                                args.focus_color,
                            )
                        else:
                            focus_box = discriminative_box(topology_inputs[1:])
                            topology_inputs = [
                                focus_panel(panel, focus_box)
                                for panel in topology_inputs
                            ]
                    draw.text(
                        (8, 306),
                        "Raster topology (colored-leaf ROI): red=endpoints, blue=junctions"
                        if args.focus_color
                        else "Raster topology (candidate-difference ROI): red=endpoints, blue=junctions"
                        if args.focus_differences
                        else "Raster topology: red=endpoints, blue=junctions",
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
    if args.topology_report:
        args.topology_report.parent.mkdir(parents=True, exist_ok=True)
        args.topology_report.write_text(
            json.dumps(topology_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
