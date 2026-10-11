import json
import sys
import math
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import (
    search_leaf_alignments, select_joint_leaf_alignments, LeafAlignment,
    _leaf_points, _candidate_bounds, _skeleton_bounds, _align_points,
    _hypothesis_bounds, _inside_canvas, _score_points, _transformed_points, _diverse,
    _alignment_ids_specs, _alignment_ids_cost
)
from scipy.ndimage import distance_transform_edt

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

# Build leaf nominal bounds dictionary
leaf_nominal = {}
for stroke in candidate.strokes:
    p = stroke.component_path
    if p not in leaf_nominal:
        leaf_nominal[p] = list(stroke.bounds)
    else:
        leaf_nominal[p][0] = min(leaf_nominal[p][0], stroke.bounds[0])
        leaf_nominal[p][1] = min(leaf_nominal[p][1], stroke.bounds[1])
        leaf_nominal[p][2] = max(leaf_nominal[p][2], stroke.bounds[2])
        leaf_nominal[p][3] = max(leaf_nominal[p][3], stroke.bounds[3])

print("Leaf nominal bounds:")
for p, b in leaf_nominal.items():
    print(f"  {p}: {b}")

# Find pairs of leaves that are nominally left-right or top-bottom separated
nominal_pairs = []
leaf_paths = sorted(leaf_nominal.keys())
for i in range(len(leaf_paths)):
    for j in range(i + 1, len(leaf_paths)):
        p1, p2 = leaf_paths[i], leaf_paths[j]
        b1, b2 = leaf_nominal[p1], leaf_nominal[p2]
        # Check horizontal separation
        if b1[2] <= b2[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
            nominal_pairs.append(("horizontal", p1, p2))
        elif b2[2] <= b1[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
            nominal_pairs.append(("horizontal", p2, p1))
        # Check vertical separation
        elif b1[3] <= b2[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
            nominal_pairs.append(("vertical", p1, p2))
        elif b2[3] <= b1[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
            nominal_pairs.append(("vertical", p2, p1))

print("\nNominal pairs separated:")
for pair in nominal_pairs:
    print(f"  {pair[0]}: {pair[1]} -> {pair[2]}")
