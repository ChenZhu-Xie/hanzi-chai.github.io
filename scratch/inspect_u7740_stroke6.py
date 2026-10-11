import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.primitive_route import enumerate_primitive_routes, select_joint_routes

html_text = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html").read_text(encoding="utf-8")
idx = html_text.find("const CASES=[")
start = idx + len("const CASES=")
depth = 0
end = -1
for i in range(start, len(html_text)):
    c = html_text[i]
    if c == '[':
        depth += 1
    elif c == ']':
        depth -= 1
        if depth == 0:
            end = i + 1
            break

cases = json.loads(html_text[start:end])
c = [x for x in cases if x["unicode"] == "U+7740"][0]

skel_str = c["skeletonPath"]
skel_grid = np.zeros((256, 256), dtype=bool)
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", skel_str):
    x = float(m.group(1))
    y = float(m.group(2))
    ix = int(round(x * 2.56))
    iy = int(round(y * 2.56))
    if 0 <= ix < 256 and 0 <= iy < 256:
        skel_grid[iy, ix] = True

glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
candidate = _compile_audit_candidate(repo, catalog, 0x7740, 900019)

stroke6 = candidate.strokes[6]
stroke7 = candidate.strokes[7]

print(f"Stroke 6: {stroke6.feature}, leaf={stroke6.leaf_id}, path={stroke6.component_path}")
print(f"Stroke 7: {stroke7.feature}, leaf={stroke7.leaf_id}, path={stroke7.component_path}")

# Enumerate stroke 6 routes
routes6 = enumerate_primitive_routes(skel_grid, candidate, stroke6, endpoint_count=5)
print(f"\nStroke 6 routes: count={len(routes6)}")
for i, r in enumerate(routes6):
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    # Check bounding box of route
    xs = [p[1]/2.56 for p in r.pixels]
    ys = [p[0]/2.56 for p in r.pixels]
    print(f"  [{i:02d}] score={r.score:.3f}, len={len(r.pixels)}, p0={p0}, p1={p1}, x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}], ep={r.endpoint_cost:.2f}, guide={r.guide_cost:.2f}, dir={r.direction_cost:.2f}")

# Let's inspect skeleton in the left bottom area (where stroke 6 should sweep):
# x in [10, 40], y in [45, 90]
sweep_pts = []
for iy in range(int(45*2.56), int(95*2.56)):
    for ix in range(int(10*2.56), int(45*2.56)):
        if skel_grid[iy, ix]:
            sweep_pts.append((ix/2.56, iy/2.56))
print(f"\nSkeleton points in left sweep area (x:10..45, y:45..95): {len(sweep_pts)}")
for y_band in range(45, 95, 5):
    pts_b = [p for p in sweep_pts if y_band <= p[1] < y_band+5]
    if pts_b:
        xs = [p[0] for p in pts_b]
        print(f"  y=[{y_band}, {y_band+5}): {len(pts_b)} pts, x range: [{min(xs):.1f}, {max(xs):.1f}], xs: {sorted(list(set(round(x, 1) for x in xs)))}")
