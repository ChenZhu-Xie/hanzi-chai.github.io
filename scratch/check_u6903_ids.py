import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate
from glyph_decomposer.leaf_alignment import alignment_ids_cost, LeafAlignment

glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
candidate_catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
candidate = _compile_audit_candidate(repo, candidate_catalog, 0x6903, 900004)

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
c = [x for x in cases if x["unicode"] == "U+6903"][0]

als = c["leafAlignmentSelection"]["alignments"]
al_objs = [
    LeafAlignment(
        component_path=tuple(a["componentPath"]),
        bounds=tuple(a["bounds"]),
        score=a["score"],
        scale_x=a["scaleX"],
        scale_y=a["scaleY"],
        offset_x=a["offsetX"],
        offset_y=a["offsetY"],
        claimed_pixels=(),
    )
    for a in als
]

base_ids = alignment_ids_cost(candidate, tuple(al_objs))
print("Base IDS cost with original alignments:", base_ids)

# Now check with the replaced alignment for (1, 1)
rep_al_objs = []
for a in al_objs:
    if a.component_path == (1, 1):
        rep = LeafAlignment(
            component_path=(1, 1),
            bounds=(53.984375, 24.689431662087905, 84.53125, 72.92045844780219),
            score=2.7,
            scale_x=1.7,
            scale_y=1.3,
            offset_x=-10.0,
            offset_y=-30.0,
            claimed_pixels=(),
        )
        rep_al_objs.append(rep)
    else:
        rep_al_objs.append(a)

rep_ids = alignment_ids_cost(candidate, tuple(rep_al_objs))
print("Replaced IDS cost with upper alignment:", rep_ids)
print("Difference:", rep_ids - base_ids)
