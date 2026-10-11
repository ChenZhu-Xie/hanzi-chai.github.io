import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import search_leaf_alignments, select_joint_leaf_alignments, LeafAlignment

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

# Get pools for U+7740
paths = sorted(list(set(s.component_path for s in candidate.strokes)))
pools = {path: search_leaf_alignments(skel_grid, candidate, path) for path in paths}

print("Available paths:", paths)
for p in paths:
    print(f"Pool {p}: count={len(pools[p])}")

# Let's inspect candidate nominal bounds for each leaf
for p in paths:
    strokes = [s for s in candidate.strokes if s.component_path == p]
    pts = np.asarray([pt for s in strokes for pt in s.points])
    print(f"  Nominal {p}: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
