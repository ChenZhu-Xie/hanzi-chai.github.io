import time
import json
from pathlib import Path
import glyph_decomposer.primitive_audit as pa

pdf = Path(".local/unicode/U4E00.pdf")
bbox_cache = Path(".local/unicode/U4E00-bbox.html")
glyph_data = Path(".local/latest-glyphs.json")
candidate_catalog = Path(".local/reviewed-visual-benchmark-current.json")
cache_dir = Path(".local/glyph-decomposer-cache")

specs_path = Path(".local/upstream-hard-18-cases.json")
hard_18_cases = json.loads(specs_path.read_text(encoding="utf-8"))

output_html = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html")
output_html.parent.mkdir(parents=True, exist_ok=True)

print(f"Starting benchmark for {len(hard_18_cases)} upstream difficult sources...")
start_time = time.time()

cases = pa.build_primitive_fit_cases(
    pdf,
    bbox_cache,
    glyph_data,
    candidate_catalog,
    [],
    blind_cases=hard_18_cases,
    cache_dir=cache_dir,
)

elapsed = time.time() - start_time
print(f"Fitting completed in {elapsed:.1f} seconds. Rendering HTML...")

html = pa.render_primitive_fit_audit(cases)
output_html.write_text(html, encoding="utf-8")
print(f"Saved report to {output_html}")

complete_count = 0
zero_count = 0
tot_residual = 0
tot_repeated = 0

print("\n" + "="*80)
print(f"{'Key':<20} {'Fits':<8} {'Residual':<10} {'Repeated':<10} {'Truth'}")
print("="*80)

for c in cases:
    fits = sum(1 for x in c.get("fits", []) if x.get("route") is not None)
    expected = c.get("expectedStrokeCount", len(c.get("fits", [])))
    joint = c.get("jointSelection") or {}
    residual = joint.get("residualPixelCount", len(c.get("residualPoints", [])))
    repeated = joint.get("repeatedPixelCount", 0)
    
    if fits == expected:
        complete_count += 1
        tot_residual += residual
        tot_repeated += repeated
    elif fits == 0:
        zero_count += 1
        
    print(f"{c['key']:<20} {fits:>2}/{expected:<5} {residual:<10} {repeated:<10} Blind")

print("="*80)
print(f"SUMMARY:")
print(f"  Complete: {complete_count}/{len(cases)}")
print(f"  0/N: {zero_count}")
print(f"  Sum Repeated: {tot_repeated}")
print(f"  Sum Residual: {tot_residual}")
print(f"  Elapsed: {elapsed:.1f}s")
print("="*80)
