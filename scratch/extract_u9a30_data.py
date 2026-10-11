import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from glyph_decomposer.primitive_audit import parse_cells, find_cell, export_page_svg, extract_cell_geometry, generate_skeleton
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate

print("1. Extracting U+9A30 K...")
cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x9A30, "K")

page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

bx0, by0, bx1, by1 = cell.bbox
bw = bx1 - bx0
bh = by1 - by0

def ring_to_d(ring):
    coords = list(ring.coords)
    if not coords: return ""
    d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}"
    for c in coords[1:]:
        d += f" L {c[0]:.2f},{c[1]:.2f}"
    d += " Z"
    return d

parts = []
if hasattr(geom, "geoms"):
    polys = list(geom.geoms)
else:
    polys = [geom]

for poly in polys:
    parts.append(ring_to_d(poly.exterior))
    for inner in poly.interiors:
        parts.append(ring_to_d(inner))
svg_outline = " ".join(parts)

skel_pixels = []
for y in range(256):
    for x in range(256):
        if skel[y, x]:
            skel_pixels.append([round(x * 100.0 / 256.0, 2), round(y * 100.0 / 256.0, 2), int(x), int(y)])

print(f"Extracted {len(skel_pixels)} skeleton points.")

raw_data = {
    "unicode": "U+9A30",
    "character": "騰",
    "source": "K",
    "glyph_id": 133032,
    "bbox": list(cell.bbox),
    "svg_outline": svg_outline,
    "skel_pixels": skel_pixels
}

out_file = Path("scratch/u9a30_raw_data.json")
out_file.write_text(json.dumps(raw_data, ensure_ascii=False), encoding="utf-8")
print(f"Saved to {out_file}")
