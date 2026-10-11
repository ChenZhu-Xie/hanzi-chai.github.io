import sys
import json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point

data = json.load(open("scratch/u9a30_raw_data.json", encoding="utf-8"))
skel_pixels = data["skel_pixels"]
print(f"Total skel points: {len(skel_pixels)}")

# Define all 20 strokes with visual parameters
strokes = [
    # --- 月 (0..3) ---
    {
        "index": 0, "name": "月·竖撇", "feature": "撇", "component": "月 (0,)",
        "primitive_type": "bezier", "path_d": "M 17.6,16.0 Q 17.6,58.0 9.0,87.5",
        "controls": [[17.6, 16.0], [17.6, 58.0], [9.0, 87.5]]
    },
    {
        "index": 1, "name": "月·横折钩", "feature": "横折钩", "component": "月 (0,)",
        "primitive_type": "polyline", "path_d": "M 17.6,16.0 L 31.2,16.0 L 31.2,85.0 L 26.2,80.5",
        "controls": [[17.6, 16.0], [31.2, 16.0], [31.2, 85.0], [26.2, 80.5]]
    },
    {
        "index": 2, "name": "月·上内横", "feature": "横", "component": "月 (0,)",
        "primitive_type": "line", "path_d": "M 17.6,35.2 L 31.2,35.2",
        "controls": [[17.6, 35.2], [31.2, 35.2]]
    },
    {
        "index": 3, "name": "月·下内横", "feature": "横", "component": "月 (0,)",
        "primitive_type": "line", "path_d": "M 17.6,54.7 L 31.2,54.7",
        "controls": [[17.6, 54.7], [31.2, 54.7]]
    },

    # --- 八 (4..5) ---
    {
        "index": 4, "name": "八·左撇", "feature": "撇", "component": "八 (1, 0, 0)",
        "primitive_type": "bezier", "path_d": "M 65.2,11.7 Q 55.0,18.0 43.0,25.0",
        "controls": [[65.2, 11.7], [55.0, 18.0], [43.0, 25.0]]
    },
    {
        "index": 5, "name": "八·右捺", "feature": "捺", "component": "八 (1, 0, 0)",
        "primitive_type": "bezier", "path_d": "M 67.6,14.4 Q 76.5,20.0 86.7,25.5",
        "controls": [[67.6, 14.4], [76.5, 20.0], [86.7, 25.5]]
    },

    # --- 天/大 (6..9) ---
    {
        "index": 6, "name": "天·首横", "feature": "横", "component": "天 (1, 0, 1)",
        "primitive_type": "line", "path_d": "M 51.5,24.6 L 83.5,24.6",
        "controls": [[51.5, 24.6], [83.5, 24.6]]
    },
    {
        "index": 7, "name": "天·次横", "feature": "横", "component": "天 (1, 0, 1)",
        "primitive_type": "line", "path_d": "M 44.5,35.2 L 88.7,35.2",
        "controls": [[44.5, 35.2], [88.7, 35.2]]
    },
    {
        "index": 8, "name": "天·长撇", "feature": "撇", "component": "天 (1, 0, 1)",
        "primitive_type": "bezier", "path_d": "M 64.5,24.6 Q 55.0,37.0 42.2,48.0",
        "controls": [[64.5, 24.6], [55.0, 37.0], [42.2, 48.0]]
    },
    {
        "index": 9, "name": "天·大捺", "feature": "捺", "component": "天 (1, 0, 1)",
        "primitive_type": "bezier", "path_d": "M 64.5,35.2 Q 76.0,42.0 89.5,48.5",
        "controls": [[64.5, 35.2], [76.0, 42.0], [89.5, 48.5]]
    },

    # --- 馬 (10..15) ---
    {
        "index": 10, "name": "馬·首横", "feature": "横", "component": "馬 (1, 1, 0)",
        "primitive_type": "line", "path_d": "M 47.3,54.7 L 79.3,54.7",
        "controls": [[47.3, 54.7], [79.3, 54.7]]
    },
    {
        "index": 11, "name": "馬·左竖", "feature": "竖", "component": "馬 (1, 1, 0)",
        "primitive_type": "line", "path_d": "M 47.3,50.0 L 47.3,70.3",
        "controls": [[47.3, 50.0], [47.3, 70.3]]
    },
    {
        "index": 12, "name": "馬·中横", "feature": "横", "component": "馬 (1, 1, 0)",
        "primitive_type": "line", "path_d": "M 47.3,62.1 L 79.7,62.1",
        "controls": [[47.3, 62.1], [79.7, 62.1]]
    },
    {
        "index": 13, "name": "馬·下横", "feature": "横", "component": "馬 (1, 1, 0)",
        "primitive_type": "line", "path_d": "M 47.3,69.5 L 84.8,69.5",
        "controls": [[47.3, 69.5], [84.8, 69.5]]
    },
    {
        "index": 14, "name": "馬·中竖", "feature": "竖", "component": "馬 (1, 1, 0)",
        "primitive_type": "line", "path_d": "M 64.5,50.0 L 64.5,69.5",
        "controls": [[64.5, 50.0], [64.5, 69.5]]
    },
    {
        "index": 15, "name": "馬·横折钩", "feature": "横折钩", "component": "馬 (1, 1, 0)",
        "primitive_type": "polyline", "path_d": "M 79.3,54.7 L 79.3,77.7 L 73.0,77.7",
        "controls": [[79.3, 54.7], [79.3, 77.7], [73.0, 77.7]]
    },

    # --- 四点底 (16..19) ---
    {
        "index": 16, "name": "四点·第1点", "feature": "点", "component": "四点底 (1, 1, 1)",
        "primitive_type": "bezier", "path_d": "M 45.0,78.5 Q 48.0,81.0 52.0,83.5",
        "controls": [[45.0, 78.5], [48.0, 81.0], [52.0, 83.5]]
    },
    {
        "index": 17, "name": "四点·第2点", "feature": "点", "component": "四点底 (1, 1, 1)",
        "primitive_type": "bezier", "path_d": "M 65.5,78.5 Q 67.0,81.5 68.5,84.5",
        "controls": [[65.5, 78.5], [67.0, 81.5], [68.5, 84.5]]
    },
    {
        "index": 18, "name": "四点·第3点", "feature": "点", "component": "四点底 (1, 1, 1)",
        "primitive_type": "bezier", "path_d": "M 74.5,78.5 Q 76.0,81.5 77.5,85.5",
        "controls": [[74.5, 78.5], [76.0, 81.5], [77.5, 85.5]]
    },
    {
        "index": 19, "name": "四点·第4点", "feature": "点", "component": "四点底 (1, 1, 1)",
        "primitive_type": "bezier", "path_d": "M 81.0,78.5 Q 82.5,82.0 84.0,86.0",
        "controls": [[81.0, 78.5], [82.5, 82.0], [84.0, 86.0]]
    }
]

def sample_bezier_pts(p0, p1, p2, n=100):
    t = np.linspace(0, 1, n)
    bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return list(zip(bx, by))

stroke_lines = []
for s in strokes:
    ctrls = s["controls"]
    if s["primitive_type"] == "bezier":
        pts = sample_bezier_pts(ctrls[0], ctrls[1], ctrls[2], 100)
        stroke_lines.append(LineString(pts))
    elif s["primitive_type"] == "polyline":
        stroke_lines.append(LineString(ctrls))
    else:
        stroke_lines.append(LineString([ctrls[0], ctrls[1]]))

skel_pts = [Point(p[0], p[1]) for p in skel_pixels]

claims = {i: [] for i in range(len(strokes))}
unclaimed = []
for p_idx, pt in enumerate(skel_pts):
    dists = [pt.distance(sl) for sl in stroke_lines]
    min_d = min(dists)
    best_s = int(np.argmin(dists))
    if min_d <= 2.2:
        claims[best_s].append(p_idx)
    else:
        unclaimed.append(p_idx)

total = len(skel_pts)
claimed_total = total - len(unclaimed)
pct = claimed_total / total * 100
print(f"Total: {total}, Claimed: {claimed_total} ({pct:.1f}%), Unclaimed: {len(unclaimed)}")
for i, s in enumerate(strokes):
    print(f"Stroke {i:02d} [{s['name']}]: {len(claims[i]):3d} pts")
