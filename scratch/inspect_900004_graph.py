import json
from pathlib import Path
import glyph_decomposer.primitive_audit as pa
from glyph_decomposer.candidate_graph import compile_candidate_graph

candidate_catalog = json.loads(Path(".local/reviewed-visual-benchmark-current.json").read_text(encoding="utf-8"))
candidate_dict = None
for row in candidate_catalog.get("rows", []):
    if row.get("candidateGlyphId") == 900004:
        candidate_dict = row
        break

graph = compile_candidate_graph(candidate_dict)
print(f"Graph root: {graph.root.component_name} (path: {graph.root.component_path})")
print(f"Total strokes: {len(graph.strokes)}")
for s in graph.strokes:
    print(f"  Stroke {s.index}: {s.feature_name} (grammar: {s.grammar_name}), path: {s.component_path}")

print("Candidate tree:")
def print_node(node, depth=0):
    print("  " * depth + f"- {node.component_name} ({node.component_path}), glyph_id={node.glyph_id}, bounds={node.nominal_bounding_box}")
    for ch in node.children:
        print_node(ch, depth + 1)
print_node(graph.root)
