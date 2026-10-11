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
c = [x for x in cases if x["unicode"] == "U+7740"][0]

print("U+7740 Leaf Alignments:")
for a in c.get("leafAlignmentSelection", {}).get("alignments", []):
    print(f"  Path {a['componentPath']}: bounds={[round(x, 1) for x in a['bounds']]}, scale=({a['scaleX']}, {a['scaleY']}), offset=({a['offsetX']}, {a['offsetY']})")

print("\nRefinements:")
print("  leafRouteRefinements:", c.get("leafRouteRefinements"))
print("  leafGlobalRouteRefinements:", c.get("leafGlobalRouteRefinements"))
