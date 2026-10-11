import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import glyph_decomposer.primitive_audit as pa

pdf = Path(".local/unicode/U4E00.pdf")
bbox_cache = Path(".local/unicode/U4E00-bbox.html")
glyph_data = Path(".local/latest-glyphs.json")
candidate_catalog = Path(".local/reviewed-visual-benchmark-current.json")
cache_dir = Path(".local/glyph-decomposer-cache")

specs_path = Path(".local/upstream-hard-18-cases.json")
hard_18_cases = json.loads(specs_path.read_text(encoding="utf-8"))

# Test just U+6903-H with maximum_ids_regression=0.01
target_case = [c for c in hard_18_cases if c["unicode"] == "U+6903"]

from glyph_decomposer import leaf_route_selection

orig_fn = leaf_route_selection.refine_suspicious_leaf_alignments
def patched_refine(*args, **kwargs):
    kwargs["maximum_ids_regression"] = 0.01
    return orig_fn(*args, **kwargs)

leaf_route_selection.refine_suspicious_leaf_alignments = patched_refine

cases = pa.build_primitive_fit_cases(
    pdf,
    bbox_cache,
    glyph_data,
    candidate_catalog,
    [],
    blind_cases=target_case,
    cache_dir=cache_dir,
)

c = cases[0]
joint = c.get("jointSelection") or {}
fits = c.get("fits", [])
print(f"U+6903-H with maximum_ids_regression=0.01:")
print(f"  Residual: {joint.get('residualPixelCount')}")
print(f"  Repeated: {joint.get('repeatedPixelCount')}")
for i in [9, 10, 11]:
    f = fits[i]
    r = f.get("route")
    poly = r.get("polyline") if isinstance(r, dict) else r
    p0 = poly[0] if poly else None
    p1 = poly[-1] if poly else None
    print(f"  Stroke {i:02d}: pts={len(poly) if poly else 0}, p0={p0}, p1={p1}")
