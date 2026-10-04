import importlib.util
import sys
import unittest
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


INK = load_module(ROOT / "unihan-ink-diffusion.py", "unihan_ink_for_residual_test")
DIRECTED = load_module(ROOT / "unihan-directed-skeleton.py", "unihan_directed_for_residual_test")
RESIDUAL = load_module(ROOT / "unihan-residual-ink.py", "unihan_residual_ink_tested")


class ResidualInkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            RESIDUAL,
            "unihan-residual-ink.py must implement route-bounded width recovery",
        )

    @staticmethod
    def plus_fixture():
        skeleton = np.zeros((15, 15), dtype=bool)
        skeleton[7, 1:14] = True
        skeleton[1:14, 7] = True
        graph = INK.build_skeleton_graph(skeleton)
        directed = DIRECTED.build_directed_skeleton(graph)
        target = cv2.dilate(
            skeleton.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
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
            edge_ids = (incoming.edge_id, outgoing.edge_id)
            indices = list(incoming.point_indices) + list(outgoing.point_indices[1:])
            points = graph.points[np.asarray(indices)][:, ::-1].astype(float)
            return edge_ids, points

        return target, graph, directed, through("E"), through("S")

    def test_horizontal_width_stops_at_unselected_vertical_half_edges(self):
        target, graph, directed, horizontal, _vertical = self.plus_fixture()
        ledger = RESIDUAL.initial_ledger(target)
        region = RESIDUAL.recover_stroke_region(
            target, graph, directed, horizontal[0], horizontal[1], ledger, contact_radius=1
        )
        self.assertTrue(region.mask[7, 2])
        self.assertFalse(region.mask[2, 7])
        self.assertFalse(region.mask[12, 7])
        self.assertFalse(region.forbidden_leak_mask.any())

    def test_vertical_hook_does_not_claim_later_rising_stroke(self):
        target, graph, directed, _horizontal, vertical = self.plus_fixture()
        ledger = RESIDUAL.initial_ledger(target)
        region = RESIDUAL.recover_stroke_region(
            target, graph, directed, vertical[0], vertical[1], ledger, contact_radius=1
        )
        self.assertTrue(region.mask[2, 7])
        self.assertFalse(region.mask[7, 2])
        self.assertFalse(region.mask[7, 12])

    def test_contact_disk_remains_reusable_after_first_stroke(self):
        target, graph, directed, horizontal, _vertical = self.plus_fixture()
        ledger = RESIDUAL.initial_ledger(target)
        region = RESIDUAL.recover_stroke_region(
            target, graph, directed, horizontal[0], horizontal[1], ledger, contact_radius=1
        )
        updated = RESIDUAL.advance_ledger(ledger, region, 0, horizontal[0])
        self.assertTrue(updated.reusable_contact[7, 7])
        self.assertFalse(updated.unexplained[7, 7])

    def test_later_stroke_visibly_overpaints_shared_contact(self):
        target, graph, directed, horizontal, vertical = self.plus_fixture()
        ledger = RESIDUAL.initial_ledger(target)
        first = RESIDUAL.recover_stroke_region(
            target, graph, directed, horizontal[0], horizontal[1], ledger, contact_radius=1
        )
        ledger = RESIDUAL.advance_ledger(ledger, first, 0, horizontal[0])
        second = RESIDUAL.recover_stroke_region(
            target, graph, directed, vertical[0], vertical[1], ledger, contact_radius=1
        )
        ledger = RESIDUAL.advance_ledger(ledger, second, 1, vertical[0])
        self.assertEqual(int(ledger.visible_owner[7, 7]), 1)
        self.assertEqual(int(ledger.visible_owner[7, 2]), 0)
        self.assertEqual(int(ledger.visible_owner[2, 7]), 1)

    def test_explained_non_contact_ink_is_not_available_to_later_start(self):
        target, graph, directed, horizontal, _vertical = self.plus_fixture()
        ledger = RESIDUAL.initial_ledger(target)
        region = RESIDUAL.recover_stroke_region(
            target, graph, directed, horizontal[0], horizontal[1], ledger, contact_radius=1
        )
        updated = RESIDUAL.advance_ledger(ledger, region, 0, horizontal[0])
        self.assertFalse(updated.unexplained[7, 2])
        self.assertFalse(updated.reusable_contact[7, 2])
        self.assertTrue(updated.available[7, 7])
        self.assertFalse(updated.available[7, 2])


if __name__ == "__main__":
    unittest.main()
