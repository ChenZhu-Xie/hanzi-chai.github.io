import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.primitive_route import enumerate_primitive_routes
from glyph_decomposer.leaf_alignment import LeafAlignment

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

als = c["leafAlignmentSelection"]["alignments"]
alignments = {
    tuple(a["componentPath"]): LeafAlignment(
        component_path=tuple(a["componentPath"]),
        bounds=tuple(a["bounds"]),
        score=a["score"],
        scale_x=a["scaleX"],
        scale_y=a["scaleY"],
        offset_x=a["offsetX"],
        offset_y=a["offsetY"],
        claimed_pixels=(),
    )
    for a in als
}

stroke6 = candidate.strokes[6]
alts = enumerate_primitive_routes(skel_grid, candidate, stroke6, leaf_alignment=alignments.get(stroke6.component_path))

for i in [6, 11, 22, 23, 24]:
    r = alts[i]
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    xs = [p[1]/2.56 for p in r.pixels]
    ys = [p[0]/2.56 for p in r.pixels]
    print(f"[{i:02d}] score={r.score:.3f}, len={len(r.pixels)}, p0={p0}, p1={p1}")
    print(f"     fit.rmse={r.fit.rmse:.3f}, fit.max_err={r.fit.maximum_error:.3f}")
    print(f"     endpoint_cost={r.endpoint_cost:.2f}, guide_cost={r.guide_cost:.2f}")
    print(f"     length_cost={r.length_cost:.2f}, dir_cost={r.direction_cost:.2f}, reg_cost={r.region_cost:.2f}")
