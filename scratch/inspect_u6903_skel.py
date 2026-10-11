import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

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
c = [x for x in cases if x["unicode"] == "U+6903"][0]

# Parse SVG skeleton path M x,y ...
import re
skel_str = c["skeletonPath"]
# Each pixel in skeletonPath is M{x:.3f},{y:.3f}h0.391v0.391h-0.391z
pts = []
for m in re.finditer(r"M([\d\.]+),([\d\.]+)", skel_str):
    pts.append((float(m.group(1)), float(m.group(2))))

print(f"Total skeleton pixels in U+6903-H: {len(pts)}")
rb_pts = [p for p in pts if p[0] > 50 and p[1] > 40]
print(f"Skeleton pixels in x>50, y>40: {len(rb_pts)}")

# Group skeleton pixels in x>50, y>40 by x and y ranges
for y_band in range(40, 95, 5):
    band_pts = [p for p in rb_pts if y_band <= p[1] < y_band + 5]
    if band_pts:
        xs = [p[0] for p in band_pts]
        print(f"  y=[{y_band:02d}, {y_band+5:02d}): {len(band_pts):2d} pts, x range: [{min(xs):.1f}, {max(xs):.1f}], x_clusters: {sorted(list(set(round(x, 0) for x in xs)))}")
