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
cell = find_cell(cells, 0x7740, "J")
page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

# Convert shapely polygon to SVG path in 0..100 coordinates
minx, miny, maxx, maxy = geom.bounds
print(f"Geom bounds: {minx}, {miny}, {maxx}, {maxy}")

def poly_to_svg(poly):
    paths = []
    # Normalize with cell bbox
    bx0, by0, bx1, by1 = cell.bbox
    bw = bx1 - bx0
    bh = by1 - by0
    
    def pt_to_str(p):
        x = (p[0] - bx0) / bw * 100.0
        y = (p[1] - by0) / bh * 100.0
        return f"{x:.2f},{y:.2f}"

    def ring_to_d(ring):
        coords = list(ring.coords)
        if not coords: return ""
        d = f"M {pt_to_str(coords[0])}"
        for c in coords[1:]:
            d += f" L {pt_to_str(c)}"
        d += " Z"
        return d

    if isinstance(poly, shapely.geometry.Polygon):
        parts = [ring_to_d(poly.exterior)]
        for inner in poly.interiors:
            parts.append(ring_to_d(inner))
        return " ".join(parts)
    elif isinstance(poly, shapely.geometry.MultiPolygon):
        parts = []
        for p in poly.geoms:
            parts.append(ring_to_d(p.exterior))
            for inner in p.interiors:
                parts.append(ring_to_d(inner))
        return " ".join(parts)
    return ""

svg_outline = poly_to_svg(geom)
print(f"SVG outline path len: {len(svg_outline)}")

# Also extract skeleton points in 0..100 space
skel_pts = []
for y in range(256):
    for x in range(256):
        if skel[y, x]:
            skel_pts.append([round(x * 100.0 / 256.0, 2), round(y * 100.0 / 256.0, 2)])

print(f"Extracted {len(skel_pts)} skeleton points")
