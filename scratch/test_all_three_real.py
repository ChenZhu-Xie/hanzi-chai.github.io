import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from scipy.ndimage import distance_transform_edt
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import search_leaf_alignments, select_joint_leaf_alignments
from glyph_decomposer.primitive_route import enumerate_primitive_routes, select_joint_routes

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

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))

def evaluate(unicode_val, glyph_id, name):
    c = [x for x in cases if x["unicode"] == unicode_val][0]
    skel_grid = np.zeros((256, 256), dtype=bool)
    for m in re.finditer(r"M([\d\.]+),([\d\.]+)", c["skeletonPath"]):
        ix = int(round(float(m.group(1)) * 2.56))
        iy = int(round(float(m.group(2)) * 2.56))
        if 0 <= ix < 256 and 0 <= iy < 256:
            skel_grid[iy, ix] = True
    cand = _compile_audit_candidate(repo, catalog, int(unicode_val[2:], 16), glyph_id)
    dist = distance_transform_edt(~skel_grid, return_indices=True)
    paths = sorted(list(set(s.component_path for s in cand.strokes)))
    pools = {p: search_leaf_alignments(skel_grid, cand, p, distance_field=dist) for p in paths}
    sol_al = select_joint_leaf_alignments(pools, cand, skeleton=skel_grid)
    alignments = {a.component_path: a for a in sol_al.alignments}

    routes_by_stroke = tuple(
        enumerate_primitive_routes(
            skel_grid,
            cand,
            stroke,
            leaf_alignment=alignments.get(stroke.component_path),
        )
        for stroke in cand.strokes
    )
    sol_rt = select_joint_routes(routes_by_stroke, int(skel_grid.sum()), candidate=cand)
    print("=" * 70)
    print(f"Target: {name} ({unicode_val}, glyph {glyph_id})")
    if sol_rt is None:
        print("  FAILED: select_joint_routes returned None")
    else:
        print(f"  Residual: {sol_rt.residual_pixel_count}, Repeated: {sol_rt.repeated_pixel_count}, Score: {sol_rt.score:.3f}")
        for a in sol_al.alignments:
            print(f"    Align {a.component_path}: bounds={[round(x,1) for x in a.bounds]}, scale=({a.scale_x},{a.scale_y}), offset=({a.offset_x},{a.offset_y})")
        for i, r in enumerate(sol_rt.routes):
            feat = cand.strokes[i].feature
            leaf = cand.strokes[i].leaf_id
            p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
            p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
            print(f"    Stroke {i:02d} ({feat:2s}, leaf {leaf}): len={len(r.pixels):3d}, p0={p0}, p1={p1}")

print("\n--- Testing Target 1: U+6903 椃 · H ---")
evaluate("U+6903", 900004, "椃 · H")

print("\n--- Testing Target 2: U+7740 着 · J ---")
evaluate("U+7740", 900019, "着 · J")

print("\n--- Testing Target 3: U+9A30 騰 · K ---")
evaluate("U+9A30", 133032, "騰 · K")
