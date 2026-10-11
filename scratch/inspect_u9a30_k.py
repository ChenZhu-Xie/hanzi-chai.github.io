import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from glyph_decomposer.primitive_audit import parse_cells, find_cell, export_page_svg, extract_cell_geometry, generate_skeleton
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate

print("1. Parsing cells...")
cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x9A30, "K")
if cell is None:
    print("Cell not found for 0x9A30 K! Searching all cells for 0x9A30...")
    c_list = [c for c in cells if getattr(c, 'codepoint', None) == 0x9A30 or c.cell_id == 0x9A30]
    print("Found cells:", [(c.cell_id, c.source, c.page) for c in c_list])
else:
    print("Cell found:", cell.unicode, cell.source, cell.page, cell.bbox)

    page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
    geom = extract_cell_geometry(page_svg, cell)
    ink, skel = generate_skeleton(geom, 256)
    skel_count = skel.sum()
    print(f"Skel pixels count: {skel_count}")

repo = GlyphRepository.load(Path(".local/latest-glyphs.json"))
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))

# Look for cases in primitive-fit-upstream-hard-18.html
html_text = Path("artifacts/deterministic-decomposer/primitive-fit-upstream-hard-18.html").read_text(encoding="utf-8")
start = html_text.find("const CASES=") + len("const CASES=")
depth = 0
end = -1
for i in range(start, len(html_text)):
    if html_text[i] == "[": depth += 1
    elif html_text[i] == "]":
        depth -= 1
        if depth == 0: end = i + 1; break
cases = json.loads(html_text[start:end])
c9a30 = [x for x in cases if x["unicode"] == "U+9A30"]
print(f"Cases for U+9A30 in hard-18: {len(c9a30)}")
if c9a30:
    c = c9a30[0]
    print("Key:", c["key"], "Glyph ID:", c["candidateGlyphId"], "Source:", c["source"])
    cand = _compile_audit_candidate(repo, catalog, 0x9A30, c["candidateGlyphId"])
    print(f"Strokes count: {len(cand.strokes)}")
    for i, s in enumerate(cand.strokes):
        print(f"Stroke {i:02d}: {s.feature:3s} leaf={s.leaf_id} path={s.component_path} bounds={[round(x,1) for x in s.bounds]} cmds={s.commands}")
