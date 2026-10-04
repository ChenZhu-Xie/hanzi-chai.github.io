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
    def test_html_keeps_native_pdf_fill_below_ink_and_outline_above_it(self):
        points = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        route = MODULE.RouteCandidate(0, 1, points, frozenset({(0, 0), (1, 1)}), 1, 0.0, {})
        page = MODULE.build_html(
            record={"unicode": 0x65E8, "source": "T"},
            glyph={
                "definitions": '<path id="pdf-glyph" style="stroke:none" d="M0 0L1 1"/>',
                "useAttributes": {"href": "#pdf-glyph", "x": 174.42, "y": 621.06},
            },
            glyph_id=127685,
            strokes=[
                {
                    "points": points,
                    "feature": "horizontal",
                    "componentId": 1128,
                    "occurrence": 0,
                    "color": "#f59e0b",
                }
            ],
            target=np.ones((2, 2), dtype=bool),
            skeleton=np.eye(2, dtype=bool),
            graph=None,
            ranked=[[]],
            routes=[route],
            owner=np.zeros((2, 2), dtype=np.int16),
            arrival=np.zeros((2, 2), dtype=float),
            events=[
                {
                    "stroke": 1,
                    "start": {"point": [0, 0], "time": 0},
                    "end": {"point": [1, 1], "time": 1},
                    "startTime": 0,
                    "endTime": 1,
                    "stopReason": "test",
                }
            ],
            maximum_time=1.0,
            decision={},
            evaluation=None,
            truth_lines=[],
        )
        self.assertIn('id="pdf-native-fill"', page)
        self.assertIn('id="ink-diffusion-canvas"', page)
        self.assertIn('id="pdf-native-outline"', page)
        self.assertLess(page.index('id="pdf-native-fill"'), page.index('id="ink-diffusion-canvas"'))
        self.assertLess(page.index('id="ink-diffusion-canvas"'), page.index('id="pdf-native-outline"'))
        self.assertIn('id="show-pdf-fill"', page)
        self.assertIn('id="show-pdf-outline"', page)
        self.assertIn('id="outline-opacity"', page)
        self.assertIn('class="pdf-fill-use pdf-source-fit"', page)
        self.assertIn('class="pdf-outline-use pdf-source-fit"', page)
        self.assertIn("document.querySelectorAll('.pdf-source-fit').forEach(fitPdfSource)", page)
        self.assertEqual(page.count('d="M0 0L1 1"'), 2)
        self.assertEqual(page.count('href="#pdf-glyph"'), 1)
        self.assertEqual(page.count('href="#pdf-glyph-outline"'), 1)
        outline = page[page.index('id="pdf-native-outline"') :]
        self.assertIn("fill:none;stroke:#0f172a", outline)

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
