import sys
import json
from pathlib import Path
from collections import defaultdict

data = json.loads(Path("scratch/u9a30_raw_data.json").read_text(encoding="utf-8"))
skel_pts = data["skel_pixels"]

# Right Bottom: x >= 42, y >= 50
rb = [p for p in skel_pts if p[0] >= 42 and p[1] >= 50]
print(f"Total points in 馬: {len(rb)}")

# Find vertical structures in 馬
x_bins = defaultdict(list)
y_bins = defaultdict(list)
for p in rb:
    x_bins[round(p[0], 1)].append(p[1])
    y_bins[round(p[1], 1)].append(p[0])

print("\nVertical rails in 馬:")
for x, y_vals in sorted(x_bins.items()):
    if len(y_vals) >= 12:
        print(f"x={x:5.1f} (count={len(y_vals):2d}): y in [{min(y_vals):.1f}, {max(y_vals):.1f}]")

print("\nHorizontal rails in 馬:")
for y, x_vals in sorted(y_bins.items()):
    if len(x_vals) >= 12:
        print(f"y={y:5.1f} (count={len(x_vals):2d}): x in [{min(x_vals):.1f}, {max(x_vals):.1f}]")

# Inspect the 4 dots
dots = [p for p in rb if p[1] > 78]
print(f"\nFour dots total points: {len(dots)}")
# Cluster by x into 4 dots
dot_clusters = defaultdict(list)
for p in dots:
    if p[0] < 63: dot_clusters[0].append(p)
    elif p[0] < 72: dot_clusters[1].append(p)
    elif p[0] < 80: dot_clusters[2].append(p)
    else: dot_clusters[3].append(p)

for i in range(4):
    c = dot_clusters[i]
    if c:
        print(f"Dot {i}: {len(c)} pts, x in [{min(p[0] for p in c):.1f}, {max(p[0] for p in c):.1f}], y in [{min(p[1] for p in c):.1f}, {max(p[1] for p in c):.1f}]")
