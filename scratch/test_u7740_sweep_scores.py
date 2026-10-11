import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import re
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import (
    _leaf_points, _candidate_bounds, _skeleton_bounds, _align_points,
    _hypothesis_bounds, _inside_canvas, _score_points
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

path = (0, 1)
points = _leaf_points(candidate, path)
point_minimum = points.min(axis=0)
point_maximum = points.max(axis=0)
span = np.maximum(point_maximum - point_minimum, 1e-6)
normalized = (points - point_minimum) / span
aligned = _align_points(
    tuple(map(tuple, points)),
    _candidate_bounds(candidate),
    _skeleton_bounds(skel_grid),
)
minimum = aligned.min(axis=0)
maximum = aligned.max(axis=0)
base = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
distances = distance_transform_edt(~skel_grid)

print(f"Base bounds for (0, 1): {[round(x, 1) for x in base]}")

# Test scores with scale=(1.0, 1.0) and scale=(1.0, 0.8) and scale=(0.8, 0.8)
for sx in [0.8, 1.0, 1.3]:
    for sy in [0.8, 1.0]:
        for off_x in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
            for off_y in [-5, 0, 5, 10]:
                bounds = _hypothesis_bounds(base, sx, sy, off_x, off_y)
                sc = _score_points(normalized, bounds, distances)
                if sc < 3.5:
                    print(f"  s=({sx}, {sy}), off=({off_x}, {off_y}), bounds={[round(x, 1) for x in bounds]}, score={sc:.3f}")
