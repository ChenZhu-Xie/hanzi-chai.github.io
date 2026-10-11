import sys
import json
from pathlib import Path
from collections import defaultdict

data = json.loads(Path("scratch/u9a30_raw_data.json").read_text(encoding="utf-8"))
skel_pts = data["skel_pixels"]
print(f"Total skel points: {len(skel_pts)}")

# 1. Inspect Left (月): x < 42
left = [p for p in skel_pts if p[0] < 42]
print(f"\n--- LEFT (月): {len(left)} points ---")
# Stroke 0: 撇 (left vertical rail)
# Stroke 1: 横折钩 (top horizontal + right vertical rail + hook)
# Stroke 2: 首内横 (y ~ 35)
# Stroke 3: 次内横 (y ~ 54)

# Find coordinates in 月
x_left_rail = [p for p in left if p[0] < 22]
x_right_rail = [p for p in left if p[0] > 26 and p[1] > 20]
top_horiz = [p for p in left if p[1] < 20]
mid1_horiz = [p for p in left if 30 < p[1] < 42]
mid2_horiz = [p for p in left if 49 < p[1] < 60]
hook_pts = [p for p in left if p[1] > 75 and p[0] > 24]

print(f"Left rail (撇): {len(x_left_rail)} pts, x in [{min(p[0] for p in x_left_rail):.1f}, {max(p[0] for p in x_left_rail):.1f}], y in [{min(p[1] for p in x_left_rail):.1f}, {max(p[1] for p in x_left_rail):.1f}]")
print(f"Right rail (折/钩): {len(x_right_rail)} pts, x in [{min(p[0] for p in x_right_rail):.1f}, {max(p[0] for p in x_right_rail):.1f}], y in [{min(p[1] for p in x_right_rail):.1f}, {max(p[1] for p in x_right_rail):.1f}]")
print(f"Top horiz: {len(top_horiz)} pts, x in [{min(p[0] for p in top_horiz):.1f}, {max(p[0] for p in top_horiz):.1f}], y in [{min(p[1] for p in top_horiz):.1f}, {max(p[1] for p in top_horiz):.1f}]")
print(f"Mid1 horiz: {len(mid1_horiz)} pts, x in [{min(p[0] for p in mid1_horiz):.1f}, {max(p[0] for p in mid1_horiz):.1f}], y in [{min(p[1] for p in mid1_horiz):.1f}, {max(p[1] for p in mid1_horiz):.1f}]")
print(f"Mid2 horiz: {len(mid2_horiz)} pts, x in [{min(p[0] for p in mid2_horiz):.1f}, {max(p[0] for p in mid2_horiz):.1f}], y in [{min(p[1] for p in mid2_horiz):.1f}, {max(p[1] for p in mid2_horiz):.1f}]")

# 2. Inspect Right Top 1 (八): x >= 42, y < 26
rt1 = [p for p in skel_pts if p[0] >= 42 and p[1] < 26]
print(f"\n--- RIGHT TOP 1 (八): {len(rt1)} points ---")
# Stroke 4: 撇 (left dot/pie)
# Stroke 5: 捺 (right na)
rt1_left = [p for p in rt1 if p[0] < 68]
rt1_right = [p for p in rt1 if p[0] >= 68]
print(f"RT1 撇: {len(rt1_left)} pts, x in [{min(p[0] for p in rt1_left):.1f}, {max(p[0] for p in rt1_left):.1f}], y in [{min(p[1] for p in rt1_left):.1f}, {max(p[1] for p in rt1_left):.1f}]")
print(f"RT1 捺: {len(rt1_right)} pts, x in [{min(p[0] for p in rt1_right):.1f}, {max(p[0] for p in rt1_right):.1f}], y in [{min(p[1] for p in rt1_right):.1f}, {max(p[1] for p in rt1_right):.1f}]")

# 3. Inspect Right Top 2 (天/大): x >= 42, 24 <= y < 53
rt2 = [p for p in skel_pts if p[0] >= 42 and 24 <= p[1] < 53]
print(f"\n--- RIGHT TOP 2 (天/大): {len(rt2)} points ---")
# Stroke 6: 横 (top horizontal, y ~ 25)
# Stroke 7: 横 (middle horizontal, y ~ 35)
# Stroke 8: 撇 (central pie sweeping southwest, x: 45~75, y: 26~52)
# Stroke 9: 捺 (right na sweeping southeast, x: 73~90, y: 35~52)
y_bins_rt2 = defaultdict(list)
for p in rt2: y_bins_rt2[round(p[1], 1)].append(p[0])
for y, xs in sorted(y_bins_rt2.items()):
    if len(xs) >= 8:
        print(f"  y={y:5.1f} (count={len(xs):2d}): x in [{min(xs):.1f}, {max(xs):.1f}]")

# 4. Inspect Right Bottom (馬): x >= 42, y >= 50
rb = [p for p in skel_pts if p[0] >= 42 and p[1] >= 50]
print(f"\n--- RIGHT BOTTOM (馬): {len(rb)} points ---")
# Let's see the 4 dots (y > 78) and the upper frame of 馬 (50 <= y <= 78)
rb_upper = [p for p in rb if p[1] <= 78]
rb_dots = [p for p in rb if p[1] > 78]
print(f"馬 上部框架: {len(rb_upper)} pts, y in [{min(p[1] for p in rb_upper):.1f}, {max(p[1] for p in rb_upper):.1f}]")
print(f"馬 四点底: {len(rb_dots)} pts, y in [{min(p[1] for p in rb_dots):.1f}, {max(p[1] for p in rb_dots):.1f}]")
