import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INK = load_module(ROOT / "unihan-ink-diffusion.py", "unihan_ink_for_directed_test")
DIRECTED = load_module(ROOT / "unihan-directed-skeleton.py", "unihan_directed_tested")


class DirectedSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            DIRECTED,
            "unihan-directed-skeleton.py must provide the directed half-edge graph",
        )

    @staticmethod
    def plus_graph():
        skeleton = np.zeros((11, 11), dtype=bool)
        skeleton[5, 1:10] = True
        skeleton[1:10, 5] = True
        return INK.build_skeleton_graph(skeleton)

    def test_cross_has_four_gated_roads_and_eight_directed_half_edges(self):
        directed = DIRECTED.build_directed_skeleton(self.plus_graph())
        self.assertEqual(len({edge.road_id for edge in directed.edges}), 4)
        self.assertEqual(len(directed.edges), 8)
        junctions = [gate for gate in directed.gates if len(gate.outgoing) == 4]
        self.assertEqual(len(junctions), 1)

    def test_orthogonal_junction_has_no_diagonal_bypass(self):
        graph = self.plus_graph()
        directed = DIRECTED.build_directed_skeleton(graph)
        center = graph.point_index[(5, 5)]
        roads = [
            edge
            for edge in directed.edges
            if center in edge.point_indices
        ]
        self.assertEqual(len(roads), 8)
        for edge in roads:
            points = graph.points[np.asarray(edge.point_indices)]
            steps = np.diff(points, axis=0)
            self.assertFalse(
                any(abs(int(dy)) == 1 and abs(int(dx)) == 1 for dy, dx in steps),
                f"road {edge.edge_id} bypassed the orthogonal gate diagonally",
            )

    def test_horizontal_route_rejects_vertical_exits_at_cross(self):
        directed = DIRECTED.build_directed_skeleton(self.plus_graph())
        incoming = next(
            edge
            for edge in directed.edges
            if edge.end_sector == "E" and len(directed.gates[edge.end_gate].outgoing) == 4
        )
        exits = DIRECTED.legal_exit_edges(
            directed, incoming.edge_id, expected_sector="E", expected_turns=("straight",)
        )
        self.assertEqual(len(exits), 1)
        self.assertEqual(directed.edges[exits[0]].start_sector, "E")

    def test_t_junction_can_reuse_contact_without_selecting_branch(self):
        skeleton = np.zeros((11, 11), dtype=bool)
        skeleton[5, 1:10] = True
        skeleton[1:6, 5] = True
        directed = DIRECTED.build_directed_skeleton(INK.build_skeleton_graph(skeleton))
        incoming = next(
            edge
            for edge in directed.edges
            if edge.end_sector == "E" and len(directed.gates[edge.end_gate].outgoing) == 3
        )
        exits = DIRECTED.legal_exit_edges(
            directed, incoming.edge_id, expected_sector="E", expected_turns=("straight",)
        )
        self.assertEqual(len(exits), 1)
        self.assertEqual(directed.edges[exits[0]].start_sector, "E")
        gate = directed.gates[incoming.end_gate]
        self.assertEqual(len(gate.point_indices), 1)


if __name__ == "__main__":
    unittest.main()
