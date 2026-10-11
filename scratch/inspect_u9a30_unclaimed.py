import sys
sys.path.append(".")
import json
from shapely.geometry import Point
from scratch.test_u9a30_claims import skel_pixels, stroke_lines, strokes

skel_pts = [Point(p[0], p[1]) for p in skel_pixels]

unclaimed = []
for p_idx, pt in enumerate(skel_pts):
    dists = [pt.distance(sl) for sl in stroke_lines]
    if min(dists) > 2.5:
        unclaimed.append((pt.x, pt.y))

print(f"Unclaimed points (>2.5): {len(unclaimed)}")

# Cluster unclaimed by region
left = [p for p in unclaimed if p[0] < 42]
rt1 = [p for p in unclaimed if p[0] >= 42 and p[1] < 26]
rt2 = [p for p in unclaimed if p[0] >= 42 and 26 <= p[1] < 53]
rb = [p for p in unclaimed if p[0] >= 42 and p[1] >= 53]

print(f"  Left unclaimed: {len(left)}")
print(f"  RT1 unclaimed: {len(rt1)}")
print(f"  RT2 unclaimed: {len(rt2)}")
print(f"  RB unclaimed: {len(rb)}")

# Print samples of each
for name, pts in [("Left", left), ("RT1", rt1), ("RT2", rt2), ("RB", rb)]:
    print(f"\nSample {name} unclaimed:")
    for p in pts[:8]:
        print(f"  ({p[0]:.2f}, {p[1]:.2f})")
