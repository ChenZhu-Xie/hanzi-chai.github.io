import sys
import json
from pathlib import Path
from scratch.analyze_u9a30_skel import skel_pixels

left_pts = [p for p in skel_pixels if p[0] < 42]
print(f"Left points count: {len(left_pts)}")
xs = [p[0] for p in left_pts]
ys = [p[1] for p in left_pts]
print(f"Left bounds: x=[{min(xs):.1f}, {max(xs):.1f}], y=[{min(ys):.1f}, {max(ys):.1f}]")

# Find vertical rails in 月:
from collections import defaultdict
x_bins = defaultdict(list)
y_bins = defaultdict(list)
for p in left_pts:
    x_bins[round(p[0], 1)].append(p[1])
    y_bins[round(p[1], 1)].append(p[0])

print("\nVertical structures in 月:")
for x, y_vals in sorted(x_bins.items()):
    if len(y_vals) >= 15:
        print(f"x={x:5.1f} (count={len(y_vals):2d}): y from {min(y_vals):.1f} to {max(y_vals):.1f}")

print("\nHorizontal structures in 月:")
for y, x_vals in sorted(y_bins.items()):
    if len(x_vals) >= 10:
        print(f"y={y:5.1f} (count={len(x_vals):2d}): x from {min(x_vals):.1f} to {max(x_vals):.1f}")
