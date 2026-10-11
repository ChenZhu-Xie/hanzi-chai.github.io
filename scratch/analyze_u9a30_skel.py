import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import shapely.geometry
from glyph_decomposer.primitive_audit import (
    parse_cells, find_cell, export_page_svg,
    extract_cell_geometry, generate_skeleton
)

cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x9A30, "K")
page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

bx0, by0, bx1, by1 = cell.bbox
bw = bx1 - bx0
bh = by1 - by0

skel_pixels = []
for y in range(256):
    for x in range(256):
        if skel[y, x]:
            skel_pixels.append((round(x * 100.0 / 256.0, 2), round(y * 100.0 / 256.0, 2), int(x), int(y)))

print(f"Total skel pixels: {len(skel_pixels)}")
xs = [p[0] for p in skel_pixels]
ys = [p[1] for p in skel_pixels]
print(f"Skel bounds: x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")

# Let's see horizontal clusters and vertical clusters
from collections import defaultdict
y_bins = defaultdict(list)
x_bins = defaultdict(list)
for p in skel_pixels:
    y_bins[round(p[1], 1)].append(p[0])
    x_bins[round(p[0], 1)].append(p[1])

print("\n--- Prominent Horizontal Lines (y bins with > 20 points) ---")
for y, x_vals in sorted(y_bins.items()):
    if len(x_vals) >= 15:
        print(f"y={y:5.1f} (count={len(x_vals):2d}): x from {min(x_vals):.1f} to {max(x_vals):.1f}")

print("\n--- Prominent Vertical Lines (x bins with > 25 points) ---")
for x, y_vals in sorted(x_bins.items()):
    if len(y_vals) >= 20:
        print(f"x={x:5.1f} (count={len(y_vals):2d}): y from {min(y_vals):.1f} to {max(y_vals):.1f}")
