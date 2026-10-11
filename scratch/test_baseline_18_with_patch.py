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
    search_leaf_alignments, select_joint_leaf_alignments, LeafAlignment,
    _leaf_points, _candidate_bounds, _skeleton_bounds, _align_points,
    _hypothesis_bounds, _inside_canvas, _score_points, _transformed_points, _diverse,
    _alignment_ids_specs, _alignment_ids_cost
)
from glyph_decomposer.primitive_route import (
    enumerate_primitive_routes, select_joint_routes, _longest_shared_run,
)
import glyph_decomposer.primitive_route as pr
import glyph_decomposer.leaf_alignment as la

# 1. search_leaf_alignments with offset penalty and near-origin preservation
def patched_search_leaf_alignments(
    skeleton: np.ndarray,
    candidate,
    component_path: tuple[int, ...],
    *,
    count: int = 8,
    distance_field: tuple[np.ndarray, np.ndarray] | None = None,
    allow_extreme_scales: bool = False,
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
    base = (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )
    distances, nearest = (
        distance_field
        if distance_field is not None
        else distance_transform_edt(~skeleton, return_indices=True)
    )
    scales = (
        (0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5)
        if allow_extreme_scales
        else (0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5)
    )
    offsets = tuple(float(value) for value in range(-30, 31, 5))
    hypotheses = []
    for scale_x in scales:
        for scale_y in scales:
            scale_penalty = 0.04 * (
                abs(math.log(scale_x)) + abs(math.log(scale_y))
            ) + 0.5 * abs(math.log(scale_x / scale_y))
            for offset_x in offsets:
                for offset_y in offsets:
                    bounds = _hypothesis_bounds(
                        base, scale_x, scale_y, offset_x, offset_y
                    )
                    if not _inside_canvas(bounds):
                        continue
                    offset_penalty = 0.02 * math.hypot(offset_x, offset_y)
                    score = _score_points(normalized, bounds, distances) + scale_penalty + offset_penalty
                    transformed = _transformed_points(normalized, bounds)
                    height, width = skeleton.shape
                    xs = np.clip(
                        np.rint(transformed[:, 0] * width / 100 - 0.5),
                        0,
                        width - 1,
                    ).astype(int)
                    ys = np.clip(
                        np.rint(transformed[:, 1] * height / 100 - 0.5),
                        0,
                        height - 1,
                    ).astype(int)
                    claimed = tuple(
                        sorted(
                            {
                                (int(nearest[0, y, x]), int(nearest[1, y, x]))
                                for y, x in zip(ys, xs)
                            }
                        )
                    )
                    hypotheses.append(
                        LeafAlignment(
                            component_path,
                            bounds,
                            score,
                            scale_x,
                            scale_y,
                            offset_x,
                            offset_y,
                            claimed,
                        )
                    )
    hypotheses.sort(key=lambda item: (item.score, item.bounds))
    bucketed = {}
    for hypothesis in hypotheses:
        x0, y0, x1, y1 = hypothesis.bounds
        signature = (
            round((x0 + x1) / 20),
            round((y0 + y1) / 20),
            hypothesis.scale_x,
            hypothesis.scale_y,
        )
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

# 2. _alignment_ids_specs with nominal pairwise constraints
orig_alignment_ids_specs = la._alignment_ids_specs

def patched_alignment_ids_specs(candidate, leaf_paths):
    specs = list(orig_alignment_ids_specs(candidate, leaf_paths))
    leaf_nominal = {}
    for stroke in candidate.strokes:
        p = stroke.component_path
        if p in leaf_paths:
            if p not in leaf_nominal:
                leaf_nominal[p] = list(stroke.bounds)
            else:
                leaf_nominal[p][0] = min(leaf_nominal[p][0], stroke.bounds[0])
                leaf_nominal[p][1] = min(leaf_nominal[p][1], stroke.bounds[1])
                leaf_nominal[p][2] = max(leaf_nominal[p][2], stroke.bounds[2])
                leaf_nominal[p][3] = max(leaf_nominal[p][3], stroke.bounds[3])
    paths_list = sorted(leaf_nominal.keys())
    for i in range(len(paths_list)):
        for j in range(i + 1, len(paths_list)):
            p1, p2 = paths_list[i], paths_list[j]
            b1, b2 = leaf_nominal[p1], leaf_nominal[p2]
            if b1[2] <= b2[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
                spec = ("⿰", frozenset({p1}), frozenset({p2}))
                if spec not in specs:
                    specs.append(spec)
            elif b2[2] <= b1[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
                spec = ("⿰", frozenset({p2}), frozenset({p1}))
                if spec not in specs:
                    specs.append(spec)
            elif b1[3] <= b2[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
                spec = ("⿱", frozenset({p1}), frozenset({p2}))
                if spec not in specs:
                    specs.append(spec)
            elif b2[3] <= b1[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
                spec = ("⿱", frozenset({p2}), frozenset({p1}))
                if spec not in specs:
                    specs.append(spec)
    return tuple(specs)

# 3. Patch _reuses_same_leaf_segment to prevent cross-leaf theft
def patched_reuses_segment(
    candidate,
    selected,
    proposed,
    *,
    maximum_shared_run: int = 8,
    maximum_retrace_fraction: float = 0.75,
) -> bool:
    for route in selected:
        shared = _longest_shared_run(proposed.pixels, frozenset(route.pixels))
        if shared > maximum_shared_run:
            return True
        shorter_length = min(len(proposed.pixels), len(route.pixels))
        if (
            shorter_length > 0
            and shared >= maximum_retrace_fraction * shorter_length
            and shared > 4
        ):
            return True
    return False

# 4. Patch _single_curve_direction_compatible: for 撇 require minimum_cosine >= 0.88 against candidate
orig_single_curve_direction = pr._single_curve_direction_compatible

def patched_single_curve_direction_compatible(
    fitted_path,
    grammar,
    candidate_path=None,
    *,
    minimum_cosine: float = 0.80,
) -> bool:
    if grammar.feature == "撇":
        effective_min = max(minimum_cosine, 0.88)
    else:
        effective_min = minimum_cosine
    return orig_single_curve_direction(
        fitted_path, grammar, candidate_path=candidate_path, minimum_cosine=effective_min
    )

la.search_leaf_alignments = patched_search_leaf_alignments
la._alignment_ids_specs = patched_alignment_ids_specs
pr._reuses_same_leaf_segment = patched_reuses_segment
pr._single_curve_direction_compatible = patched_single_curve_direction_compatible

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))

html_text = Path("artifacts/deterministic-decomposer/primitive-fit-two-tier-18.html").read_text(encoding="utf-8")
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

print(f"Loaded {len(cases)} baseline cases. Testing all cases...")
passed = 0
total = len(cases)
for idx, c in enumerate(cases):
    u = int(c["unicode"][2:], 16)
    gid = c["candidateGlyphId"]
    name = c["key"]
    skel_grid = np.zeros((256, 256), dtype=bool)
    for m in re.finditer(r"M([\d\.]+),([\d\.]+)", c["skeletonPath"]):
        ix = int(round(float(m.group(1)) * 2.56))
        iy = int(round(float(m.group(2)) * 2.56))
        if 0 <= ix < 256 and 0 <= iy < 256:
            skel_grid[iy, ix] = True
    cand = _compile_audit_candidate(repo, catalog, u, gid)
    dist = distance_transform_edt(~skel_grid, return_indices=True)
    paths = sorted(list(set(s.component_path for s in cand.strokes)))
    pools = {p: patched_search_leaf_alignments(skel_grid, cand, p, distance_field=dist) for p in paths}
    sol_al = select_joint_leaf_alignments(pools, cand, skeleton=skel_grid)
    alignments = {a.component_path: a for a in sol_al.alignments}

    routes_by_stroke = tuple(
        enumerate_primitive_routes(
            skel_grid,
            cand,
            stroke,
            leaf_alignment=alignments.get(stroke.component_path),
        )
        for stroke in cand.strokes
    )
    sol_rt = select_joint_routes(routes_by_stroke, int(skel_grid.sum()), candidate=cand)
    if sol_rt is not None:
        passed += 1
        print(f"  [{idx+1:02d}/{total:02d}] PASS: {name:<20} res={sol_rt.residual_pixel_count:3d}, rep={sol_rt.repeated_pixel_count:3d}, score={sol_rt.score:.2f}")
    else:
        print(f"  [{idx+1:02d}/{total:02d}] FAIL: {name:<20} select_joint_routes returned None!")

print(f"\nFinal Result: {passed}/{total} passed.")
