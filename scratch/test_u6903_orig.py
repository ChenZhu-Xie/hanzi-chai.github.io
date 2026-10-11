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
c = [x for x in cases if x["unicode"] == "U+6903"][0]

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
candidate = _compile_audit_candidate(repo, catalog, 0x6903, 900004)

als = c["leafAlignmentSelection"]["alignments"]
original_alignments = {
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

print("Original alignments:")
for p, a in original_alignments.items():
    print(f"  {p}: bounds={a.bounds}")

# Route all 12 strokes with original alignments
alts = tuple(
    enumerate_primitive_routes(
        skel_grid,
        candidate,
        stroke,
        leaf_alignment=original_alignments[stroke.component_path],
    )
    for stroke in candidate.strokes
)

sol = select_joint_routes(alts, int(skel_grid.sum()), candidate=candidate)
if sol:
    print("\nWith original alignments, select_joint_routes succeeded!")
    print(f"Residual: {sol.residual_pixel_count}, Repeated: {sol.repeated_pixel_count}, Score: {sol.score:.3f}")
    for i, r in enumerate(sol.routes):
        p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
        p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
        print(f"  Stroke {i:02d}: pts={len(r.pixels)}, p0={p0}, p1={p1}")
else:
    print("\nWith original alignments, select_joint_routes returned None!")
