import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("unihan-ink-diffusion.py")
SPEC = importlib.util.spec_from_file_location("unihan_ink_diffusion_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DirectedInkDiffusionTests(unittest.TestCase):
    def test_crossing_number_distinguishes_path_and_t_junction(self):
        skeleton = np.zeros((9, 9), dtype=bool)
        skeleton[1:8, 4] = True
        skeleton[4, 4:8] = True
        _count, crossing = MODULE.crossing_numbers(skeleton)
        self.assertEqual(int(crossing[1, 4]), 1)
        self.assertEqual(int(crossing[2, 4]), 2)
        self.assertGreaterEqual(int(crossing[4, 4]), 3)

    def test_contact_signature_is_scale_and_translation_independent(self):
        horizontal = np.asarray([[0.0, 5.0], [10.0, 5.0]])
        vertical = np.asarray([[4.0, 0.0], [4.0, 10.0]])
        original = MODULE.contact_signature(horizontal, vertical)[:2]
        transformed = MODULE.contact_signature(horizontal * 8 + 37, vertical * 8 + 37)[:2]
        self.assertTrue(np.allclose(original, transformed, atol=0.03))
        self.assertTrue(np.allclose(original, (0.4, 0.5), atol=0.04))

    def test_directed_contact_roles_change_when_a_stroke_is_reversed(self):
        first = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        second = np.asarray([[10.0, 0.0], [10.0, 8.0]])
        forward = MODULE.contact_signature(first, second)[:2]
        reversed_first = MODULE.contact_signature(first[::-1], second)[:2]
        self.assertTrue(np.allclose(forward, (1.0, 0.0), atol=0.03))
        self.assertTrue(np.allclose(reversed_first, (0.0, 0.0), atol=0.03))

    def test_alternate_route_can_be_found_when_shortest_fork_is_closed(self):
        # A square has two topologically distinct paths from top-left to
        # bottom-right.  Closing the first edge must expose the other one.
        adjacency = [
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (2, 1.0)],
            [(1, 1.0), (3, 1.0)],
            [(0, 1.0), (2, 1.0)],
        ]
        first, _ = MODULE.shortest_indices(adjacency, 0, 2)
        alternate, _ = MODULE.shortest_indices(adjacency, 0, 2, (first[0], first[1]))
        self.assertNotEqual(first, alternate)
        self.assertEqual({tuple(first), tuple(alternate)}, {(0, 1, 2), (0, 3, 2)})


if __name__ == "__main__":
    unittest.main()
