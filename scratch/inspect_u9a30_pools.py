import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import search_leaf_alignments

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
c = [x for x in cases if x["unicode"] == "U+9A30"][0]

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
candidate = _compile_audit_candidate(repo, catalog, 0x9A30, 133032)

path_100 = (1, 0, 0)
pool_100 = search_leaf_alignments(skel_grid, candidate, path_100)
print(f"Pool for (1, 0, 0): {len(pool_100)} candidates")
for i, a in enumerate(pool_100[:15]):
    print(f"  [{i:02d}] score={a.score:.3f}, bounds={[round(x, 1) for x in a.bounds]}, scale=({a.scale_x}, {a.scale_y}), offset=({a.offset_x}, {a.offset_y}), claims={len(a.claimed_pixels)}")

path_101 = (1, 0, 1)
pool_101 = search_leaf_alignments(skel_grid, candidate, path_101)
print(f"\nPool for (1, 0, 1): {len(pool_101)} candidates")
for i, a in enumerate(pool_101[:15]):
    print(f"  [{i:02d}] score={a.score:.3f}, bounds={[round(x, 1) for x in a.bounds]}, scale=({a.scale_x}, {a.scale_y}), offset=({a.offset_x}, {a.offset_y}), claims={len(a.claimed_pixels)}")
