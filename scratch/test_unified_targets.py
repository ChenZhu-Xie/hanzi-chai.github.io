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

def evaluate_target(unicode_val, glyph_id):
    c = [x for x in cases if x["unicode"] == unicode_val][0]
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
    candidate = _compile_audit_candidate(repo, catalog, int(unicode_val[2:], 16), glyph_id)

    # Search with offset prior & near origin preservation
    def search_with_offset_prior(skel, cand, comp_path, count=8):
        points = _leaf_points(cand, comp_path)
        point_minimum = points.min(axis=0)
        point_maximum = points.max(axis=0)
        span = np.maximum(point_maximum - point_minimum, 1e-6)
        normalized = (points - point_minimum) / span
        aligned = _align_points(
            tuple(map(tuple, points)),
            _candidate_bounds(cand),
            _skeleton_bounds(skel),
        )
        minimum = aligned.min(axis=0)
        maximum = aligned.max(axis=0)
        base = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
        distances, nearest = distance_transform_edt(~skel, return_indices=True)
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
                        offset_dist = math.hypot(offset_x, offset_y)
                        offset_penalty = 0.02 * offset_dist
                        score = _score_points(normalized, bounds, distances) + scale_penalty + offset_penalty
                        transformed = _transformed_points(normalized, bounds)
                        height, width = skel.shape
                        xs = np.clip(np.rint(transformed[:, 0] * width / 100 - 0.5), 0, width - 1).astype(int)
                        ys = np.clip(np.rint(transformed[:, 1] * height / 100 - 0.5), 0, height - 1).astype(int)
                        claimed = tuple(sorted({(int(nearest[0, y, x]), int(nearest[1, y, x])) for y, x in zip(ys, xs)}))
                        hypotheses.append(LeafAlignment(comp_path, bounds, score, scale_x, scale_y, offset_x, offset_y, claimed))

        hypotheses.sort(key=lambda item: (item.score, item.bounds))
        bucketed = {}
        for h in hypotheses:
            x0, y0, x1, y1 = h.bounds
            sig = (round((x0+x1)/20), round((y0+y1)/20), h.scale_x, h.scale_y)
            bucketed.setdefault(sig, h)
        ordered = sorted(bucketed.values(), key=lambda item: (item.score, item.bounds))
        selected = []
        near_origin = [h for h in ordered if math.hypot(h.offset_x, h.offset_y) <= 12]
        for h in near_origin[:2]:
            if _diverse(h, selected):
                selected.append(h)
        for h in ordered:
            if _diverse(h, selected):
                selected.append(h)
                if len(selected) == count:
                    break
        return tuple(selected)

    paths = sorted(list(set(s.component_path for s in candidate.strokes)))
    pools = {p: search_with_offset_prior(skel_grid, candidate, p) for p in paths}
    sol_al = select_joint_leaf_alignments(pools, candidate, skeleton=skel_grid)
    alignments = {a.component_path: a for a in sol_al.alignments}

    # Route strokes
    alts = tuple(
        enumerate_primitive_routes(
            skel_grid,
            candidate,
            stroke,
            leaf_alignment=alignments.get(stroke.component_path),
        )
        for stroke in candidate.strokes
    )

    # Cross-leaf retrace filter
    import glyph_decomposer.primitive_route as pr
    orig_reuses = pr._reuses_same_leaf_segment

    def new_reuses(cand, selected, proposed, *, maximum_shared_run=8, maximum_retrace_fraction=0.75):
        strokes = {s.index: s for s in cand.strokes}
        prop_s = strokes[proposed.stroke_index]
        for route in selected:
            sel_s = strokes[route.stroke_index]
            shared = _longest_shared_run(proposed.pixels, frozenset(route.pixels))
            # Same leaf: 8 pixels. Different leaf: 8 pixels!
            if shared > maximum_shared_run:
                return True
            shorter = min(len(proposed.pixels), len(route.pixels))
            if shorter > 0 and shared >= maximum_retrace_fraction * shorter and shared > 4:
                return True
        return False

    pr._reuses_same_leaf_segment = new_reuses
    sol = pr.select_joint_routes(alts, int(skel_grid.sum()), candidate=candidate)
    pr._reuses_same_leaf_segment = orig_reuses

    print("=" * 70)
    print(f"EVALUATION: {unicode_val} (Glyph {glyph_id})")
    if sol:
        print(f"  Residual: {sol.residual_pixel_count}, Repeated: {sol.repeated_pixel_count}, Score: {sol.score:.3f}")
        for i, r in enumerate(sol.routes):
            p0 = (round(r.pixels[0][1]/2.56, 1), round(r.pixels[0][0]/2.56, 1))
            p1 = (round(r.pixels[-1][1]/2.56, 1), round(r.pixels[-1][0]/2.56, 1))
            feat = candidate.strokes[i].feature
            leaf = candidate.strokes[i].leaf_id
            print(f"    Stroke {i:02d} ({feat:2s}, leaf {leaf}): pts={len(r.pixels):3d}, p0={p0}, p1={p1}")
    else:
        print("  select_joint_routes returned None!")

print("Running test on U+6903-H:")
evaluate_target("U+6903", 900004)

print("\nRunning test on U+7740-J:")
evaluate_target("U+7740", 900019)

print("\nRunning test on U+9A30-K:")
evaluate_target("U+9A30", 133032)
