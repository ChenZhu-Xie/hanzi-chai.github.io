import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from glyph_decomposer.candidate_graph import compile_candidate_graph
from glyph_decomposer.grammar import GlyphRepository
from glyph_decomposer.primitive_audit import _compile_audit_candidate

glyph_data = Path(".local/latest-glyphs.json")
repo = GlyphRepository.load(glyph_data)
catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))

targets = [
    (0x6903, 900004, "U+6903-H 椃"),
    (0x7740, 900019, "U+7740-J 着"),
    (0x9A30, 133032, "U+9A30-K 騰"),
]

for cp, glyph_id, name in targets:
    print("=" * 80)
    print(f"Target: {name} (Glyph {glyph_id})")
    graph = _compile_audit_candidate(repo, catalog, cp, glyph_id)
    print(f"Graph root: glyph_id={graph.root.glyph_id}, path={graph.root.path}, op={graph.root.operator}")
    def print_node(node, depth=0):
        rec = repo.record(node.glyph_id) if node.glyph_id in repo._records else {}
        name = rec.get("name", str(node.glyph_id))
        print("  " * depth + f"- {name} (path: {node.path}), op={node.operator}, bounds={[round(x, 1) for x in node.bounds]}, strokes={node.stroke_indices}")
        for ch in node.children:
            print_node(ch, depth + 1)
    print_node(graph.root)
    print("Strokes:")
    for s in graph.strokes:
        print(f"  Stroke {s.index:02d}: leaf={s.leaf_id}, path={s.component_path}, feat={s.feature}, bounds={[round(x, 1) for x in s.bounds]}, cmds={s.commands}")
