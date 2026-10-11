import json
from shapely.geometry import Point, LineString

data = json.load(open("scratch/u9a30_raw_data.json", encoding="utf-8"))
skel_pts = [p for p in data["skel_pixels"] if p[0] >= 42 and p[1] < 52]
print(f"Top right points: {len(skel_pts)}")

# Find connected branches / corridors
# Horizontal lines:
y24 = [p for p in skel_pts if abs(p[1] - 24.6) <= 1.0]
y35 = [p for p in skel_pts if abs(p[1] - 35.2) <= 1.0]
y47 = [p for p in skel_pts if abs(p[1] - 47.7) <= 1.0]

print(f"Horiz 1 (y~24.6): {len(y24)} pts, x in [{min(p[0] for p in y24):.1f}, {max(p[0] for p in y24):.1f}]")
print(f"Horiz 2 (y~35.2): {len(y35)} pts, x in [{min(p[0] for p in y35):.1f}, {max(p[0] for p in y35):.1f}]")
print(f"Horiz 3 (y~47.7): {len(y47)} pts, x in [{min(p[0] for p in y47):.1f}, {max(p[0] for p in y47):.1f}]")

# Vertical / diagonal lines:
diag_left1 = [p for p in skel_pts if p[1] < 24 and p[0] < 55] # left dot of 八?
diag_mid1 = [p for p in skel_pts if p[1] < 24 and 55 <= p[0] <= 66] # middle spine?
diag_right1 = [p for p in skel_pts if p[1] < 24 and p[0] > 66] # right na of 八?

print(f"Top Left dot (八左): {len(diag_left1)} pts, x in [{min(p[0] for p in diag_left1):.1f}, {max(p[0] for p in diag_left1):.1f}], y in [{min(p[1] for p in diag_left1):.1f}, {max(p[1] for p in diag_left1):.1f}]")
print(f"Top Mid spine (竖/撇): {len(diag_mid1)} pts, x in [{min(p[0] for p in diag_mid1):.1f}, {max(p[0] for p in diag_mid1):.1f}], y in [{min(p[1] for p in diag_mid1):.1f}, {max(p[1] for p in diag_mid1):.1f}]")
print(f"Top Right na (八右): {len(diag_right1)} pts, x in [{min(p[0] for p in diag_right1):.1f}, {max(p[0] for p in diag_right1):.1f}], y in [{min(p[1] for p in diag_right1):.1f}, {max(p[1] for p in diag_right1):.1f}]")
