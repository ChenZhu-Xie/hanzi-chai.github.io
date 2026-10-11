import sys
import json
from pathlib import Path
import re
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.stdout.reconfigure(encoding="utf-8")

from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import search_leaf_alignments, select_joint_leaf_alignments
from glyph_decomposer.primitive_route import enumerate_primitive_routes, select_joint_routes

# 1. Load U+9A30 skeleton
html_text = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html").read_text(encoding="utf-8")
start = html_text.find("const CASES=") + len("const CASES=")
depth = 0
end = -1
for i in range(start, len(html_text)):
    c_char = html_text[i]
    if c_char == "[":
        depth += 1
    elif c_char == "]":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
cases = json.loads(html_text[start:end])
c = [x for x in cases if x["unicode"] == "U+9A30"][0]

skel_grid = np.zeros((256, 256), dtype=bool)
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", c["skeletonPath"]):
    ix = int(round(float(m.group(1)) * 2.56))
    iy = int(round(float(m.group(2)) * 2.56))
    if 0 <= ix < 256 and 0 <= iy < 256:
        skel_grid[iy, ix] = True

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
cand = _compile_audit_candidate(repo, catalog, 0x9A30, 133032)

print(f"Candidate has {len(cand.strokes)} strokes:")
for i, s in enumerate(cand.strokes):
    print(f"  Stroke {i:02d}: path={s.component_path}, feat={s.feature}, cmds={s.commands}")

dist = distance_transform_edt(~skel_grid, return_indices=True)
paths = sorted(list(set(s.component_path for s in cand.strokes)))
pools = {p: search_leaf_alignments(skel_grid, cand, p, distance_field=dist) for p in paths}
sol_al = select_joint_leaf_alignments(pools, cand, skeleton=skel_grid)
alignments = {a.component_path: a for a in sol_al.alignments}

print("\nLeaf alignments:")
for a in sol_al.alignments:
    print(f"  Align {a.component_path}: bounds={[round(x,1) for x in a.bounds]}, scale=({a.scale_x:.2f},{a.scale_y:.2f}), offset=({a.offset_x:.2f},{a.offset_y:.2f})")

# Test enumeration for strokes 0..10
for i in range(min(11, len(cand.strokes))):
    stroke = cand.strokes[i]
    routes = enumerate_primitive_routes(
        skel_grid,
        cand,
        stroke,
        leaf_alignment=alignments.get(stroke.component_path),
    )
    print(f"\nStroke {i:02d} ({stroke.feature}, leaf {stroke.leaf_id}, path {stroke.component_path}): found {len(routes)} routes")
    if routes:
        best_r = routes[0]
        p0 = (round(best_r.pixels[0][1]/2.56, 1), round(best_r.pixels[0][0]/2.56, 1))
        p1 = (round(best_r.pixels[-1][1]/2.56, 1), round(best_r.pixels[-1][0]/2.56, 1))
        print(f"  Best route: score={best_r.score:.3f}, len={len(best_r.pixels)}, p0={p0}, p1={p1}, fit_rmse={best_r.fit.rmse:.3f}")
        for prim in best_r.fit.primitives:
            print(f"    primitive {prim.command} ({prim.kind}): controls={prim.controls}")
