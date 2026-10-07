"""Extract chart glyph vector outlines from Poppler-generated page SVG."""

from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from shapely import affinity
from shapely.geometry import box

from .geometry import (
    normalize_geometry,
    svg_path_geometry,
    union_geometries,
)

SOURCE_RE = re.compile(r"^(KP|UK|[GHJKMSTUV])(?=[A-Z0-9_-])")
SOURCE_MAP = {"KP": "N", "UK": "B"}
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


@dataclass(frozen=True)
class PdfCell:
    unicode: int
    source: str
    page: int
    bbox: tuple[float, float, float, float]


def _source(value: str) -> str | None:
    match = SOURCE_RE.match(value)
    if not match:
        return None
    return SOURCE_MAP.get(match.group(1), match.group(1))


def parse_cells(path: Path) -> list[PdfCell]:
    cells: list[PdfCell] = []
    page_number = 0
    for _, page in ET.iterparse(path, events=("end",)):
        if page.tag.rsplit("}", 1)[-1] != "page":
            continue
        page_number += 1
        width = float(page.attrib["width"])
        words = [
            {
                "text": child.text,
                "x0": float(child.attrib["xMin"]),
                "y0": float(child.attrib["yMin"]),
                "x1": float(child.attrib["xMax"]),
                "y1": float(child.attrib["yMax"]),
            }
            for child in page
            if child.tag.rsplit("}", 1)[-1] == "word" and child.text
        ]
        codepoints = []
        for word in words:
            if not re.fullmatch(r"[0-9A-F]{4}", word["text"]):
                continue
            value = int(word["text"], 16)
            if (
                0x4E00 <= value <= 0x9FFF
                and word["y0"] >= 80
                and (80 <= word["x0"] <= 115 or 305 <= word["x0"] <= 340)
            ):
                codepoints.append(
                    {
                        **word,
                        "unicode": value,
                        "column": 0 if word["x0"] < width / 2 else 1,
                    }
                )
        codepoints.sort(key=lambda item: (item["column"], item["y0"]))
        for codepoint in codepoints:
            later = [
                item["y0"]
                for item in codepoints
                if item["column"] == codepoint["column"]
                and item["y0"] > codepoint["y0"]
            ]
            row_end = min(
                min(later) if later else codepoint["y0"] + 34, codepoint["y0"] + 34
            )
            half_start = 0 if codepoint["column"] == 0 else width / 2
            half_end = width / 2 if codepoint["column"] == 0 else width
            glyph_words = [
                word
                for word in words
                if word["text"] == chr(codepoint["unicode"])
                and half_start <= (word["x0"] + word["x1"]) / 2 < half_end
                and codepoint["y0"] - 3 <= word["y0"] <= row_end
            ]
            seen: set[str] = set()
            for word in words:
                source = _source(word["text"])
                center = (word["x0"] + word["x1"]) / 2
                if (
                    source is None
                    or source in seen
                    or not half_start <= center < half_end
                    or not codepoint["y0"] + 13 <= word["y0"] <= row_end + 1
                ):
                    continue
                nearest = min(
                    glyph_words,
                    key=lambda glyph: abs((glyph["x0"] + glyph["x1"]) / 2 - center),
                    default=None,
                )
                if (
                    nearest is None
                    or abs((nearest["x0"] + nearest["x1"]) / 2 - center) > 8
                ):
                    continue
                seen.add(source)
                cells.append(
                    PdfCell(
                        unicode=codepoint["unicode"],
                        source=source,
                        page=page_number,
                        bbox=(
                            nearest["x0"],
                            nearest["y0"],
                            nearest["x1"],
                            nearest["y1"],
                        ),
                    )
                )
        page.clear()
    return cells


def find_cell(cells: list[PdfCell], unicode_value: int, source: str) -> PdfCell:
    matches = [
        cell
        for cell in cells
        if cell.unicode == unicode_value and cell.source == source
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one U+{unicode_value:04X}-{source} cell, found {len(matches)}"
        )
    return matches[0]


def export_page_svg(pdf: Path, page_number: int) -> str:
    with tempfile.TemporaryDirectory(prefix="glyph-decomposer-") as directory:
        output = Path(directory) / "page.svg"
        completed = subprocess.run(
            [
                "pdftocairo",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-svg",
                str(pdf),
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip())
        return output.read_text(encoding="utf-8")


def extract_cell_geometry(page_svg: str, cell: PdfCell):
    root = ET.fromstring(page_svg)
    symbols = {
        element.attrib["id"]: element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "symbol" and "id" in element.attrib
    }
    target = box(*cell.bbox)
    candidates = []
    for use in root.iter():
        if use.tag.rsplit("}", 1)[-1] != "use":
            continue
        href = use.attrib.get(XLINK_HREF) or use.attrib.get("href")
        if not href or not href.startswith("#"):
            continue
        x = float(use.attrib.get("x", 0))
        y = float(use.attrib.get("y", 0))
        if not (
            cell.bbox[0] - 2 <= x <= cell.bbox[2] + 2
            and cell.bbox[1] - 4 <= y <= cell.bbox[3] + 4
        ):
            continue
        symbol = symbols.get(href[1:])
        if symbol is None:
            continue
        paths = [
            svg_path_geometry(element.attrib["d"])
            for element in symbol.iter()
            if element.tag.rsplit("}", 1)[-1] == "path" and element.attrib.get("d")
        ]
        geometry = affinity.translate(union_geometries(paths), xoff=x, yoff=y)
        overlap = geometry.intersection(target).area
        if overlap > 0:
            candidates.append((overlap, geometry))
    if not candidates:
        raise ValueError(f"no vector glyph overlaps U+{cell.unicode:04X}-{cell.source}")
    geometry = max(candidates, key=lambda item: item[0])[1]
    return normalize_geometry(geometry)
