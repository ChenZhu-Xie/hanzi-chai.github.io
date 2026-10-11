import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate

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
c = [x for x in cases if x["unicode"] == "U+7740"][0]

print("Key:", c["key"])
print("Glyph ID:", c["candidateGlyphId"])
print("Skeleton SVG path len:", len(c["skeletonPath"]))
print("Skeleton path snippet:", c["skeletonPath"][:120])

skel_pts = []
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", c["skeletonPath"]):
    skel_pts.append((float(m.group(1)), float(m.group(2))))
print(f"Total skeleton points: {len(skel_pts)}")
xs = [p[0] for p in skel_pts]
ys = [p[1] for p in skel_pts]
print(f"Skeleton bounds: x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
cand = _compile_audit_candidate(repo, catalog, 0x7740, 900019)

print(f"\nTotal strokes: {len(cand.strokes)}")
for i, s in enumerate(cand.strokes):
    print(f"Stroke {i:02d}: {s.feature:3s} leaf={s.leaf_id} path={s.component_path} bounds={[round(x,1) for x in s.bounds]} commands={s.commands} pts_len={len(s.points)} start={s.points[0]} end={s.points[-1]}")
