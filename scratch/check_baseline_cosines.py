import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.stroke_grammar import canonical_stroke_grammar

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
glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))

min_cosines = []
for c in cases:
    try:
        candidate = _compile_audit_candidate(repo, catalog, int(c["unicode"][2:], 16), c["candidateGlyphId"])
    except Exception:
        continue
    fits = c.get("fits", [])
    for stroke in candidate.strokes:
        if stroke.index >= len(fits):
            continue
        grammar = canonical_stroke_grammar(stroke.feature, stroke.commands)
        if grammar.commands != ("c",):
            continue
        fit = fits[stroke.index]
        r = fit.get("route")
        poly = r if isinstance(r, list) else (r.get("polyline") if isinstance(r, dict) else [])
        if len(poly) < 2:
            continue
        chord = np.asarray(poly[-1]) - np.asarray(poly[0])
        cand_chord = np.asarray(stroke.points[-1]) - np.asarray(stroke.points[0])
        norm_chord = np.linalg.norm(chord)
        norm_cand = np.linalg.norm(cand_chord)
        if norm_chord > 1e-6 and norm_cand > 1e-6:
            cos_cand = float(np.dot(chord, cand_chord) / (norm_chord * norm_cand))
            vec = np.asarray(grammar.direction_vectors[0], dtype=float)
            cos_gram = float(np.dot(chord, vec) / (norm_chord * np.linalg.norm(vec)))
            min_cosines.append((cos_cand, cos_gram, c["key"], stroke.index, stroke.feature))

min_cosines.sort()
print(f"Total curve strokes in baseline: {len(min_cosines)}")
print("Lowest 10 cosines against candidate chord:")
for item in min_cosines[:10]:
    print(f"  cos_cand={item[0]:.3f}, cos_gram={item[1]:.3f}, case={item[2]}, stroke={item[3]}, feat={item[4]}")
