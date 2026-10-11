import sys
import json
import math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from scipy.ndimage import distance_transform_edt
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import (
    select_joint_leaf_alignments,
    search_leaf_alignments,
    LeafAlignment,
    _leaf_points, _candidate_bounds, _skeleton_bounds, _align_points,
    _hypothesis_bounds, _inside_canvas, _score_points, _transformed_points, _diverse,
    _alignment_ids_specs,
)
import glyph_decomposer.leaf_alignment as la

def patched_search_leaf_alignments(
    skeleton: np.ndarray,
    candidate,
    component_path: tuple[int, ...],
    *,
    count: int = 8,
    distance_field: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[LeafAlignment, ...]:
    points = _leaf_points(candidate, component_path)
    point_minimum = points.min(axis=0)
    point_maximum = points.max(axis=0)
    span = np.maximum(point_maximum - point_minimum, 1e-6)
    normalized = (points - point_minimum) / span
    aligned = _align_points(
        tuple(map(tuple, points)),
        _candidate_bounds(candidate),
        _skeleton_bounds(skeleton),
    )
    minimum = aligned.min(axis=0)
    maximum = aligned.max(axis=0)
    base = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    distances, nearest = (
        distance_field
        if distance_field is not None
        else distance_transform_edt(~skeleton, return_indices=True)
    )
    scales = (0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5)
    offsets = tuple(float(value) for value in range(-30, 31, 5))
    hypotheses = []
    for scale_x in scales:
        for scale_y in scales:
            scale_penalty = 0.04 * (
                abs(math.log(scale_x)) + abs(math.log(scale_y))
            ) + 0.5 * abs(math.log(scale_x / scale_y))
            for offset_x in offsets:
                for offset_y in offsets:
                    bounds = _hypothesis_bounds(base, scale_x, scale_y, offset_x, offset_y)
                    if not _inside_canvas(bounds):
                        continue
                    offset_penalty = 0.02 * math.hypot(offset_x, offset_y)
                    score = _score_points(normalized, bounds, distances) + scale_penalty + offset_penalty
                    transformed = _transformed_points(normalized, bounds)
                    height, width = skeleton.shape
                    xs = np.clip(np.rint(transformed[:, 0] * width / 100 - 0.5), 0, width - 1).astype(int)
                    ys = np.clip(np.rint(transformed[:, 1] * height / 100 - 0.5), 0, height - 1).astype(int)
                    claimed = tuple(sorted({(int(nearest[0, y, x]), int(nearest[1, y, x])) for y, x in zip(ys, xs)}))
                    hypotheses.append(LeafAlignment(component_path, bounds, score, scale_x, scale_y, offset_x, offset_y, claimed))

    hypotheses.sort(key=lambda item: (item.score, item.bounds))
    bucketed = {}
    for hypothesis in hypotheses:
        x0, y0, x1, y1 = hypothesis.bounds
        signature = (round((x0 + x1) / 20), round((y0 + y1) / 20), hypothesis.scale_x, hypothesis.scale_y)
        bucketed.setdefault(signature, hypothesis)
    ordered = sorted(bucketed.values(), key=lambda item: (item.score, item.bounds))
    selected: list[LeafAlignment] = []
    near_origin = [h for h in ordered if math.hypot(h.offset_x, h.offset_y) <= 12]
    for hypothesis in near_origin[:2]:
        if _diverse(hypothesis, selected):
            selected.append(hypothesis)
    for hypothesis in ordered:
        if _diverse(hypothesis, selected):
            selected.append(hypothesis)
            if len(selected) == count:
                break
    return tuple(selected)

html_text = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html").read_text(encoding="utf-8")
start = html_text.find("const CASES=") + len("const CASES=")
depth = 0
end = -1
for i in range(start, len(html_text)):
    c_char = html_text[i]
    if c_char == '[':
        depth += 1
    elif c_char == ']':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
cases = json.loads(html_text[start:end])
c = [x for x in cases if x["unicode"] == "U+9A30"][0]

skel_grid = np.zeros((256, 256), dtype=bool)
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", c["skeletonPath"]):
    ix = int(round(float(m.group(1)) * 2.56))
    iy = int(round(float(m.group(2)) * 2.56))
    if 0 <= ix < 256 and 0 <= iy < 256:
        skel_grid[iy, ix] = True

glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
cand = _compile_audit_candidate(repo, catalog, 0x9A30, 133032)

dist = distance_transform_edt(~skel_grid, return_indices=True)
paths = sorted(list(set(s.component_path for s in cand.strokes)))
pools = {p: patched_search_leaf_alignments(skel_grid, cand, p, distance_field=dist) for p in paths}

print("Candidate pools for (1, 0, 0):")
for a in pools[(1, 0, 0)]:
    print(f"  score={a.score:.3f}, bounds={[round(x,1) for x in a.bounds]}, scale=({a.scale_x},{a.scale_y}), offset=({a.offset_x},{a.offset_y})")

sol1 = select_joint_leaf_alignments(pools, cand, skeleton=skel_grid)
print("\nSelected alignments for U+9A30:")
for a in sol1.alignments:
    print(f"  {a.component_path}: bounds={[round(x, 1) for x in a.bounds]}, scale=({a.scale_x}, {a.scale_y}), offset=({a.offset_x}, {a.offset_y})")
