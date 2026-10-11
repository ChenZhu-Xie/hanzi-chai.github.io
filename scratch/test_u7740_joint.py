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

alts = tuple(
    enumerate_primitive_routes(
        skel_grid,
        candidate,
        stroke,
        leaf_alignment=alignments.get(stroke.component_path),
    )
    for stroke in candidate.strokes
)

print(f"Strokes: {len(candidate.strokes)}")
for i, alt in enumerate(alts):
    print(f"  Stroke {i:02d} ({candidate.strokes[i].feature}): {len(alt)} alternatives, min_score={alt[0].score:.3f}")

# Look at alternatives for stroke 6
print("\nStroke 6 alternatives:")
for j, r in enumerate(alts[6]):
    p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
    p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
    xs = [p[1]/2.56 for p in r.pixels]
    ys = [p[0]/2.56 for p in r.pixels]
    print(f"  [{j:02d}] score={r.score:.3f}, len={len(r.pixels)}, p0={p0}, p1={p1}, x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")

sol = select_joint_routes(alts, int(skel_grid.sum()), candidate=candidate)
if sol:
    print(f"\nJoint solution: residual={sol.residual_pixel_count}, repeated={sol.repeated_pixel_count}, score={sol.score:.3f}")
    r6 = sol.routes[6]
    p0 = (round(r6.pixels[0][1]/2.56, 1), round(r6.pixels[0][0]/2.56, 1))
    p1 = (round(r6.pixels[-1][1]/2.56, 1), round(r6.pixels[-1][0]/2.56, 1))
    print(f"  Stroke 6 chosen: len={len(r6.pixels)}, p0={p0}, p1={p1}")
