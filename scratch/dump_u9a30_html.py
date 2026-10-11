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
c = [x for x in cases if x["unicode"] == "U+9A30"][0]

print("U+9A30 Leaf Alignments:")
for a in c.get("leafAlignmentSelection", {}).get("alignments", []):
    print(f"  Path {a['componentPath']}: bounds={[round(x, 1) for x in a['bounds']]}, scale=({a['scaleX']}, {a['scaleY']}), offset=({a['offsetX']}, {a['offsetY']})")

print("\nRefinements:")
print("  leafRouteRefinements:", c.get("leafRouteRefinements"))
print("  leafGlobalRouteRefinements:", c.get("leafGlobalRouteRefinements"))

print("\nFits summary:")
for i, f in enumerate(c.get("fits", [])):
    r = f.get("route")
    poly = r if isinstance(r, list) else (r.get("polyline") if isinstance(r, dict) else [])
    p0 = poly[0] if poly else None
    p1 = poly[-1] if poly else None
    print(f"  Stroke {i:02d}: leaf={f.get('leafId')}, len={len(poly)}, p0={p0}, p1={p1}")
