import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.primitive_route import enumerate_primitive_routes, select_joint_routes, _reuses_same_leaf_segment
sys.path.insert(0, str(Path("scratch").resolve()))
from check_u9a30_diff import skel_grid, cand, sol1
import glyph_decomposer.primitive_route as pr

alignments = {a.component_path: a for a in sol1.alignments}
routes_by_stroke = tuple(
    enumerate_primitive_routes(
        skel_grid,
        cand,
        stroke,
        leaf_alignment=alignments.get(stroke.component_path),
    )
    for stroke in cand.strokes
)

# Test joint route search with candidate
sol = select_joint_routes(routes_by_stroke, int(skel_grid.sum()), candidate=cand)

print("Original select_joint_routes:")
for i, r in enumerate(sol.routes):
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    print(f"  Stroke {i:02d} ({cand.strokes[i].feature}): len={len(r.pixels)}, p0={p0}, p1={p1}")

# Check why r4[0] or r4[11] was not in sol.routes[4]
chosen_4 = sol.routes[4]
p0 = (round(chosen_4.pixels[0][1]/2.56, 1), round(chosen_4.pixels[0][0]/2.56, 1))
p1 = (round(chosen_4.pixels[-1][1]/2.56, 1), round(chosen_4.pixels[-1][0]/2.56, 1))
print(f"\nChosen Stroke 4: p0={p0}, p1={p1}")

# What other strokes share pixels with r4[0] or r4[11]?
r4_0 = routes_by_stroke[4][0]
r4_11 = routes_by_stroke[4][11]

for i in range(len(routes_by_stroke)):
    if i == 4: continue
    sel = sol.routes[i]
    sh0 = len(set(r4_0.pixels) & set(sel.pixels))
    sh11 = len(set(r4_11.pixels) & set(sel.pixels))
    if sh0 > 0 or sh11 > 0:
        print(f"  Shares with stroke {i} ({cand.strokes[i].feature}): sh0={sh0}, sh11={sh11}")
