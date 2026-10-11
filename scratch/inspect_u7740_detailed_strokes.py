import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from scipy.ndimage import distance_transform_edt
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import (
    _compile_audit_candidate, parse_cells, find_cell, export_page_svg,
    extract_cell_geometry, generate_skeleton
)

cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x7740, "J")
page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

print(f"Ink pixels: {ink.sum()}, Skeleton pixels: {skel.sum()}")

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
cand = _compile_audit_candidate(repo, catalog, 0x7740, 900019)

# Analyze skeleton coordinates in 0..100 space
skel_y, skel_x = np.where(skel)
skel_coords = list(zip(skel_x * 100.0 / 256.0, skel_y * 100.0 / 256.0))
print(f"Skel points in 0..100 space: min_x={min(x for x,y in skel_coords):.2f}, max_x={max(x for x,y in skel_coords):.2f}, min_y={min(y for x,y in skel_coords):.2f}, max_y={max(y for x,y in skel_coords):.2f}")

# Print each stroke from candidate
for i, s in enumerate(cand.strokes):
    pts = s.points
    print(f"\nStroke {i:02d}: {s.feature} ({s.commands})")
    print(f"  Leaf: {s.leaf_id}, Component Path: {s.component_path}")
    print(f"  Candidate bounds: {[round(b, 1) for b in s.bounds]}")
    print(f"  Start: {pts[0]}, End: {pts[-1]}")
    if len(pts) > 2:
        mid = pts[len(pts)//2]
        print(f"  Mid: {mid}")
