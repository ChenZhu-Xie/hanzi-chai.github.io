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

for c in cases:
    if c["unicode"] == "U+6903":
        print("=" * 80)
        print(f"U+6903 Details:")
        fits = c.get("fits", [])
        for i in [9, 10, 11]:
            f = fits[i]
            r = f.get("route")
            print(f"Stroke {i}:")
            print(f"  grammar: {f.get('grammarName')}, leaf: {f.get('leafId')}")
            print(f"  score: {f.get('score')}")
            if r:
                poly = r if isinstance(r, list) else r.get('polyline', [])
                print(f"  route polyline len: {len(poly)}")
                print(f"  route start: {poly[0] if poly else None}")
                print(f"  route end: {poly[-1] if poly else None}")
        print("Residual points count:", len(c.get("residualPoints", [])))
        # Let's inspect where residual points are located
        res_pts = c.get("residualPoints", [])
        xs = [p[0] for p in res_pts]
        ys = [p[1] for p in res_pts]
        if xs:
            print(f"Residual bbox: x=[{min(xs)}, {max(xs)}], y=[{min(ys)}, {max(ys)}]")
        # How many residual points in right-bottom area (x > 50, y > 50)?
        rb_res = [p for p in res_pts if p[0] > 50 and p[1] > 50]
        print(f"Residual points in x>50, y>50: {len(rb_res)}")
        # print some sample residual points in x>50, y>50
        print("Sample residual points in bottom-right (x>50, y>50):", rb_res[:10])
