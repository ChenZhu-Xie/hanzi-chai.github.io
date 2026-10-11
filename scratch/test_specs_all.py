import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import _alignment_ids_specs

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
html_text = Path("artifacts/deterministic-decomposer/primitive-fit-two-tier-18.html").read_text(encoding="utf-8")
start = html_text.find("const CASES=") + len("const CASES=")
depth = 0
end = -1
for i in range(start, len(html_text)):
    if html_text[i] == "[":
        depth += 1
    elif html_text[i] == "]":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
cases = json.loads(html_text[start:end])

for c in cases:
    try:
        cand = _compile_audit_candidate(repo, catalog, int(c["unicode"][2:], 16), c["candidateGlyphId"])
    except Exception:
        continue
    paths = sorted(list(set(s.component_path for s in cand.strokes)))
    specs = list(_alignment_ids_specs(cand, paths))
    leaf_nominal = {}
    for s in cand.strokes:
        p = s.component_path
        if p not in leaf_nominal:
            leaf_nominal[p] = list(s.bounds)
        else:
            leaf_nominal[p][0] = min(leaf_nominal[p][0], s.bounds[0])
            leaf_nominal[p][1] = min(leaf_nominal[p][1], s.bounds[1])
            leaf_nominal[p][2] = max(leaf_nominal[p][2], s.bounds[2])
            leaf_nominal[p][3] = max(leaf_nominal[p][3], s.bounds[3])
    new_specs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1, p2 = paths[i], paths[j]
            b1, b2 = leaf_nominal[p1], leaf_nominal[p2]
            if b1[2] <= b2[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
                spec = ("⿰", frozenset({p1}), frozenset({p2}))
                if spec not in specs:
                    new_specs.append(spec)
            elif b2[2] <= b1[0] + 5.0 and min(b1[3], b2[3]) - max(b1[1], b2[1]) > 5.0:
                spec = ("⿰", frozenset({p2}), frozenset({p1}))
                if spec not in specs:
                    new_specs.append(spec)
            elif b1[3] <= b2[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
                spec = ("⿱", frozenset({p1}), frozenset({p2}))
                if spec not in specs:
                    new_specs.append(spec)
            elif b2[3] <= b1[1] + 5.0 and min(b1[2], b2[2]) - max(b1[0], b2[0]) > 5.0:
                spec = ("⿱", frozenset({p2}), frozenset({p1}))
                if spec not in specs:
                    new_specs.append(spec)
    print(f"{c['key']}: {len(specs)} base specs, {len(new_specs)} new nominal specs")
