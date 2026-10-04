import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
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


INK = load_module(ROOT / "unihan-ink-diffusion.py", "unihan_ink_for_decoder_test")
DIRECTED = load_module(ROOT / "unihan-directed-skeleton.py", "unihan_directed_for_decoder_test")
DECODER = load_module(ROOT / "unihan-residual-decoder.py", "unihan_residual_decoder_tested")


@dataclass
class Route:
    points: np.ndarray
    pixels: frozenset[tuple[int, int]]
    score: float
    evidence: dict


def stroke(component, feature, points, occurrence=0, expected_turns=()):
    return {
        "componentId": component,
        "occurrence": occurrence,
        "feature": feature,
        "points": np.asarray(points, dtype=float),
        "expectedTurns": tuple(expected_turns),
    }


def route_for_edges(graph, directed, edge_ids, score=0.0, **evidence):
    indices = []
    for edge_id in edge_ids:
        current = list(directed.edges[edge_id].point_indices)
        indices.extend(current if not indices else current[1:])
    points = graph.points[np.asarray(indices)][:, ::-1].astype(float)
    return Route(
        points=points,
        pixels=frozenset((int(x), int(y)) for x, y in points),
        score=score,
        evidence=evidence,
    )


class ResidualDecoderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            DECODER,
            "unihan-residual-decoder.py must orchestrate sequential residual decoding",
        )

    @staticmethod
    def plus_fixture():
        skeleton = np.zeros((15, 15), dtype=bool)
        skeleton[7, 1:14] = True
        skeleton[1:14, 7] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        target = cv2.dilate(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        junction = next(gate for gate in directed.gates if len(gate.outgoing) == 4)

        def through(sector):
            incoming = next(
                edge
                for edge in directed.edges
                if edge.end_gate == junction.gate_id and edge.end_sector == sector
            )
            outgoing = next(
                edge
                for edge in directed.edges
                if edge.start_gate == junction.gate_id
                and edge.start_sector == sector
                and edge.road_id != incoming.road_id
            )
            return route_for_edges(graph, directed, (incoming.edge_id, outgoing.edge_id))

        return target, graph, directed, through("E"), through("S")

    def test_decoder_consumes_strokes_against_updated_residual(self):
        target, graph, _directed, horizontal, vertical = self.plus_fixture()
        result = DECODER.decode_residual_routes(
            target,
            graph,
            [stroke(10, "横", [[0, 0], [10, 0]]), stroke(20, "竖", [[0, 0], [0, 10]])],
            [[horizontal], [vertical]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(result.routes), 2)
        self.assertEqual(int(result.ledger.visible_owner[7, 7]), 1)
        self.assertEqual(int(result.ledger.visible_owner[7, 2]), 0)
        self.assertEqual(int(result.ledger.visible_owner[2, 7]), 1)

    def test_decoder_rejects_forbidden_junction_exit_before_scoring(self):
        target, graph, directed, _horizontal, _vertical = self.plus_fixture()
        junction = next(gate for gate in directed.gates if len(gate.outgoing) == 4)
        west = next(
            edge for edge in directed.edges if edge.end_gate == junction.gate_id and edge.end_sector == "E"
        )
        north = next(
            edge for edge in directed.edges if edge.start_gate == junction.gate_id and edge.start_sector == "N"
        )
        turning = route_for_edges(
            graph,
            directed,
            (west.edge_id, north.edge_id),
            score=-100.0,
            routeJunctionTurns=["counterclockwise"],
        )
        result = DECODER.decode_residual_routes(
            target,
            graph,
            [stroke(10, "横", [[0, 0], [10, 0]], expected_turns=("straight",))],
            [[turning]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(result.routes), 0)
        self.assertEqual(result.status, "needs-review")
        self.assertIn("no-feasible-route-stroke-1", result.review_reasons)

    def test_decoder_cannot_reenter_closed_component_occurrence(self):
        skeleton = np.zeros((11, 15), dtype=bool)
        for y in (2, 5, 8):
            skeleton[y, 1:14] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        east = sorted(
            (edge for edge in directed.edges if edge.start_sector == "E"),
            key=lambda edge: graph.points[edge.point_indices[0]][0],
        )
        routes = [route_for_edges(graph, directed, (edge.edge_id,)) for edge in east]
        result = DECODER.decode_residual_routes(
            skeleton,
            graph,
            [
                stroke(10, "横", [[0, 0], [10, 0]]),
                stroke(20, "横", [[0, 0], [10, 0]]),
                stroke(10, "横", [[0, 0], [10, 0]]),
            ],
            [[routes[0]], [routes[1]], [routes[2]]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(result.routes), 2)
        self.assertIn("closed-component-reentry-stroke-3", result.review_reasons)

    def test_beam_backtracks_when_best_early_route_blocks_later_stroke(self):
        skeleton = np.zeros((9, 15), dtype=bool)
        skeleton[2, 1:14] = True
        skeleton[6, 1:14] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        east = sorted(
            (edge for edge in directed.edges if edge.start_sector == "E"),
            key=lambda edge: graph.points[edge.point_indices[0]][0],
        )
        top = route_for_edges(graph, directed, (east[0].edge_id,), score=0.0)
        bottom = route_for_edges(graph, directed, (east[1].edge_id,), score=0.5)
        result = DECODER.decode_residual_routes(
            skeleton,
            graph,
            [stroke(10, "横", [[0, 0], [10, 0]]), stroke(20, "横", [[0, 0], [10, 0]])],
            [[top, bottom], [top]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(result.routes), 2)
        self.assertIs(result.routes[0], bottom)
        self.assertIs(result.routes[1], top)

    def test_infeasible_later_stroke_returns_needs_review_without_forced_route(self):
        skeleton = np.zeros((5, 15), dtype=bool)
        skeleton[2, 1:14] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        east = next(edge for edge in directed.edges if edge.start_sector == "E")
        only = route_for_edges(graph, directed, (east.edge_id,))
        result = DECODER.decode_residual_routes(
            skeleton,
            graph,
            [stroke(10, "横", [[0, 0], [10, 0]]), stroke(20, "横", [[0, 0], [10, 0]])],
            [[only], [only]],
            source="G",
            codepoint=0x4E00,
        )
        self.assertEqual(len(result.routes), 1)
        self.assertEqual(result.status, "needs-review")
        self.assertIn("no-feasible-route-stroke-2", result.review_reasons)

    def test_learned_prior_cannot_resurrect_illegal_route(self):
        skeleton = np.zeros((5, 15), dtype=bool)
        skeleton[2, 1:14] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        west = next(edge for edge in directed.edges if edge.start_sector == "W")
        backwards = route_for_edges(graph, directed, (west.edge_id,), score=-1000.0)
        result = DECODER.decode_residual_routes(
            skeleton,
            graph,
            [stroke(10, "横", [[0, 0], [10, 0]])],
            [[backwards]],
            source="G",
            codepoint=0x4E00,
            learned_model={"examplesBySignature": {}},
        )
        self.assertEqual(len(result.routes), 0)
        self.assertEqual(result.status, "needs-review")


if __name__ == "__main__":
    unittest.main()
