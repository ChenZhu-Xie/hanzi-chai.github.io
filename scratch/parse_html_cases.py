import json
from pathlib import Path

html_text = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html").read_text(encoding="utf-8")
idx = html_text.find("const CASES=[")
if idx == -1:
    print("CASES not found")
    exit(1)

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
print(f"Loaded {len(cases)} cases.")

for c in cases:
    if c["unicode"] in ("U+6903", "U+7740", "U+9A30"):
        print("=" * 80)
        print(f"Unicode: {c['unicode']} {c['character']} · {c['source']} (Glyph {c['candidateGlyphId']})")
        joint = c.get("jointSelection") or {}
        print(f"Residual: {joint.get('residualPixelCount')}, Repeated: {joint.get('repeatedPixelCount')}, Relation Cost: {joint.get('strokeRelationCost')}")
        fits = c.get("fits", [])
        for i, fit in enumerate(fits):
            r = fit.get("route")
            poly = r if isinstance(r, list) else (r.get("polyline") if isinstance(r, dict) else [])
            s = poly[0] if poly else None
            e = poly[-1] if poly else None
            pts = len(poly)
            leaf = fit.get("leafId")
            leaf_box = fit.get("leafBounds")
            grammar = fit.get("grammarName")
            print(f"  Stroke {i:02d}: leaf={leaf}, grammar={grammar}, pts={pts}, start={s}, end={e}, leaf_box={leaf_box}")
