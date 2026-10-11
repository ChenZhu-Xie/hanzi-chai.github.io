import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import glyph_decomposer.primitive_audit as pa
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import search_leaf_alignments, select_joint_leaf_alignments
from glyph_decomposer.leaf_route_selection import refine_suspicious_leaf_alignments
from glyph_decomposer.primitive_route import enumerate_primitive_routes

pdf = Path(".local/unicode/U4E00.pdf")
bbox_cache = Path(".local/unicode/U4E00-bbox.html")
glyph_data = Path(".local/latest-glyphs.json")
candidate_catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
cache_dir = Path(".local/glyph-decomposer-cache")

repo = GlyphRepository.load(glyph_data)

# Load U+6903-H
case_spec = {"unicode": "U+6903", "source": "H", "candidateGlyphId": 900004}
cases = pa.build_primitive_fit_cases(
    pdf,
    bbox_cache,
    glyph_data,
    Path(".local/reviewed-visual-benchmark-current.json"),
    [],
    blind_cases=[case_spec],
    cache_dir=cache_dir,
)

c = cases[0]
fits = c.get("fits", [])
print(f"Residual: {c.get('jointSelection', {}).get('residualPixelCount')}")
print(f"Repeated: {c.get('jointSelection', {}).get('repeatedPixelCount')}")

# Inspect leaf alignments
alignments = c.get("leafAlignments", {})
print("\nSelected Leaf Alignments:")
for path, al in c.get("jointLeafAlignments", {}).items():
    print(f"  Path {path}: bounds={al.get('bounds')}, score={al.get('score')}")

# Inspect stroke 10 alternatives
candidate = _compile_audit_candidate(repo, candidate_catalog, 0x6903, 900004)
skeleton = c.get("skeleton") # wait, does c have skeleton array or points?
print("Keys in c:", list(c.keys()))
