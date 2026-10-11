import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

html_text = Path("artifacts/deterministic-decomposer/primitive-fit-two-tier-18.html").read_text(encoding="utf-8")
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

def longest_shared(p1, p2):
    s2 = set(tuple(p) for p in p2)
    cur = m = 0
    for p in p1:
        if tuple(p) in s2:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m

print(f"Loaded {len(cases)} cases from two-tier-18.html")
all_shares = []
for c in cases:
    fits = c.get("fits", [])
    max_cross = 0
    cross_pairs = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            leaf_i = fits[i].get("leafId")
            leaf_j = fits[j].get("leafId")
            if leaf_i == leaf_j:
                continue
            r_i = fits[i].get("route")
            r_j = fits[j].get("route")
            p_i = r_i if isinstance(r_i, list) else (r_i.get("polyline") if isinstance(r_i, dict) else [])
            p_j = r_j if isinstance(r_j, list) else (r_j.get("polyline") if isinstance(r_j, dict) else [])
            if not p_i or not p_j:
                continue
            share = longest_shared(p_i, p_j)
            if share > 0:
                cross_pairs.append((i, j, leaf_i, leaf_j, share))
            max_cross = max(max_cross, share)
    print(f"{c['key']:<20}: max cross-leaf shared run = {max_cross}, pairs with share > 0: {len(cross_pairs)}")
    for p in cross_pairs:
        if p[4] > 5:
            print(f"    strokes {p[0]}-{p[1]} (leaves {p[2]}-{p[3]}): share={p[4]}")
