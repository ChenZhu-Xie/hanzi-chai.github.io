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
    _alignment_ids_specs, _alignment_ids_cost, ordered_axis_overlap
)
from glyph_decomposer.primitive_route import enumerate_primitive_routes, select_joint_routes, _longest_shared_run
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

# Build leaf nominal bounds
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

# Identify nominally separated leaf pairs
nominal_separated_specs = []
leaf_paths = sorted(leaf_nominal.keys())
for i in range(len(leaf_paths)):
    for j in range(i + 1, len(leaf_paths)):
        p1, p2 = leaf_paths[i], leaf_paths[j]
        b1, b2 = leaf_nominal[p1], leaf_nominal[p2]
        # horizontal separation
        if b1[2] <= b2[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
            nominal_separated_specs.append(("horizontal", p1, p2))
        elif b2[2] <= b1[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
            nominal_separated_specs.append(("horizontal", p2, p1))
        # vertical separation
        elif b1[3] <= b2[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
            nominal_separated_specs.append(("vertical", p1, p2))
        elif b2[3] <= b1[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
            nominal_separated_specs.append(("vertical", p2, p1))

print("Nominal separated specs for U+7740:", nominal_separated_specs)

# Custom select_joint_leaf_alignments that includes nominal separated specs
import glyph_decomposer.leaf_alignment as la

def custom_select_joint(alignments_by_path, cand, *, skeleton=None, beam_width=400):
    paths = tuple(sorted(alignments_by_path))
    specs = list(_alignment_ids_specs(cand, paths))
    # Add pairwise nominal leaf specs: ('⿰', frozenset({p1}), frozenset({p2})) or ('⿱', frozenset({p1}), frozenset({p2}))
    for kind, p1, p2 in nominal_separated_specs:
        op = "⿰" if kind == "horizontal" else "⿱"
        spec = (op, frozenset({p1}), frozenset({p2}))
        if spec not in specs:
            specs.append(spec)
            print(f"Added spec: {spec}")

    # Now solve with extended specs
    path_order = {path: index for index, path in enumerate(paths)}
    specs_by_completion = {path: [] for path in paths}
    for spec in specs:
        _op, f_paths, s_paths = spec
        rel_paths = f_paths | s_paths
        completion = max(rel_paths, key=path_order.__getitem__)
        specs_by_completion[completion].append(spec)

    states = [(0.0, (), frozenset(), 0, 0.0)]
    for path in paths:
        expanded = []
        for score, selected, occupied, repeated, prev_structure in states:
            for alignment in alignments_by_path[path]:
                claims = frozenset(alignment.claimed_pixels)
                overlap = len(claims & occupied)
                charged_overlap = max(0, overlap - 4)
                new_claims = len(claims - occupied)
                prop_selected = (*selected, alignment)
                structure = prev_structure + _alignment_ids_cost(
                    tuple(specs_by_completion[path]), prop_selected
                )
                prop_score = score + alignment.score + 0.8 * charged_overlap - 0.12 * new_claims
                expanded.append((prop_score, prop_selected, occupied | claims, repeated + charged_overlap, structure))
        expanded.sort(key=lambda item: (item[0] + 40.0 * item[4], item[3], tuple(a.bounds for a in item[1])))
        states = expanded[:beam_width]
    score, selected, _occ, repeated, structure = min(states, key=lambda item: (item[0] + 40.0 * item[4], item[3]))
    return la.JointLeafAlignmentSolution(selected, score + 40.0 * structure, repeated, structure)

# Get pools
paths = sorted(list(set(s.component_path for s in candidate.strokes)))
pools = {p: search_leaf_alignments(skel_grid, candidate, p) for p in paths}
sol = custom_select_joint(pools, candidate, skeleton=skel_grid)

print("\nCustom joint solution for U+7740:")
for a in sol.alignments:
    print(f"  Path {a.component_path}: bounds={[round(x, 1) for x in a.bounds]}, scale=({a.scale_x}, {a.scale_y}), offset=({a.offset_x}, {a.offset_y})")
