import json
from pathlib import Path
import glyph_decomposer.primitive_audit as pa

pdf = Path(".local/unicode/U4E00.pdf")
bbox_cache = Path(".local/unicode/U4E00-bbox.html")
glyph_data = Path(".local/latest-glyphs.json")
candidate_catalog = Path(".local/reviewed-visual-benchmark-current.json")
cache_dir = Path(".local/glyph-decomposer-cache")

targets = [
    {"unicode": "U+6903", "source": "H", "candidateGlyphId": 900004},
    {"unicode": "U+7740", "source": "J", "candidateGlyphId": 900019},
    {"unicode": "U+9A30", "source": "K", "candidateGlyphId": 133032},
]

cases = pa.build_primitive_fit_cases(
    pdf,
    bbox_cache,
    glyph_data,
    candidate_catalog,
    [],
    blind_cases=targets,
    cache_dir=cache_dir,
)

for c in cases:
    print("=" * 60)
    print(f"Target: {c['key']} (Glyph {c.get('candidateGlyphId')})")
    joint = c.get("jointSelection") or {}
    print(f"Residual: {joint.get('residualPixelCount')}, Repeated: {joint.get('repeatedPixelCount')}")
    fits = c.get("fits", [])
    print(f"Total fits: {len(fits)}")
    for i, fit in enumerate(fits):
        leaf_id = fit.get("leafId")
        route = fit.get("route")
        if route:
            poly = route.get("polyline", [])
            start = poly[0] if poly else None
            end = poly[-1] if poly else None
            print(f"  Stroke {i:02d} (Leaf {leaf_id}): points={len(poly)}, start={start}, end={end}")
        else:
            print(f"  Stroke {i:02d} (Leaf {leaf_id}): NO ROUTE")
