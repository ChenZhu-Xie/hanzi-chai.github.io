import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.primitive_route import enumerate_primitive_routes

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
c = [x for x in cases if x["unicode"] == "U+6903"][0]

import re
skel_str = c["skeletonPath"]
# build 256x256 skeleton grid
skel_grid = np.zeros((256, 256), dtype=bool)
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", skel_str):
    x = float(m.group(1))
    y = float(m.group(2))
    ix = int(round(x * 2.56))
    iy = int(round(y * 2.56))
    if 0 <= ix < 256 and 0 <= iy < 256:
        skel_grid[iy, ix] = True

print(f"Reconstructed skeleton grid with {skel_grid.sum()} pixels")

glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
candidate_catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
candidate = _compile_audit_candidate(repo, candidate_catalog, 0x6903, 900004)

stroke10 = candidate.strokes[10]
stroke11 = candidate.strokes[11]

# What leaf alignment was actually chosen in the audit for (1, 1)?
leaf_al_sel = c.get("leafAlignmentSelection", {})
print("leafAlignmentSelection:", leaf_al_sel)

routes10 = enumerate_primitive_routes(skel_grid, candidate, stroke10, endpoint_count=5)
print(f"\nStroke 10 routes (free): count={len(routes10)}")
for i, r in enumerate(routes10):
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    print(f"  [{i}] score={r.score:.3f}, len={len(r.pixels)}, p0={p0}, p1={p1}, ep={r.endpoint_cost:.2f}, guide={r.guide_cost:.2f}, dir={r.direction_cost:.2f}, reg={r.region_cost:.2f}")

routes11 = enumerate_primitive_routes(skel_grid, candidate, stroke11, endpoint_count=5)
print(f"\nStroke 11 routes (free): count={len(routes11)}")
for i, r in enumerate(routes11):
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    print(f"  [{i}] score={r.score:.3f}, len={len(r.pixels)}, p0={p0}, p1={p1}, ep={r.endpoint_cost:.2f}, guide={r.guide_cost:.2f}, dir={r.direction_cost:.2f}, reg={r.region_cost:.2f}")
