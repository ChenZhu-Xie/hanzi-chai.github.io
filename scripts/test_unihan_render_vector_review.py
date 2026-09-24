import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("unihan-render-vector-review.py")
SPEC = importlib.util.spec_from_file_location("vector_review", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VectorReviewTest(unittest.TestCase):
    def test_extracts_only_the_placed_glyph_and_prefixes_its_definition(self):
        page = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
          <defs><symbol id="glyph-a"><path d="M0 0L2 2"/></symbol><symbol id="glyph-b"><path d="M0 0L3 3"/></symbol></defs>
          <use xlink:href="#glyph-a" x="10" y="20"/><use xlink:href="#glyph-b" x="50" y="60"/></svg>"""
        result = MODULE.extract_pdf_glyph(page, [9, 19, 15, 27], "case")
        self.assertIn('id="case-glyph-a"', result["definitions"])
        self.assertNotIn("glyph-b", result["definitions"])
        self.assertEqual(result["useAttributes"]["href"], "#case-glyph-a")
        self.assertEqual(result["pathCount"], 1)

    def test_refuses_ambiguous_chart_uses(self):
        page = """<svg xmlns="http://www.w3.org/2000/svg"><defs><symbol id="a"/><symbol id="b"/></defs>
          <use href="#a" x="10" y="20"/><use href="#b" x="11" y="21"/></svg>"""
        with self.assertRaisesRegex(ValueError, "expected one"):
            MODULE.select_chart_use(page, [9, 19, 15, 27])

    def test_document_contains_no_raster_or_canvas(self):
        document = MODULE.build_document(["<section class='case'></section>"])
        self.assertNotIn("<img", document)
        self.assertNotIn("<canvas", document)
        self.assertNotIn("data:image", document)
        self.assertIn("candidate-nodes", MODULE.JS)
        self.assertIn("IntersectionObserver", MODULE.JS)
        self.assertIn(
            "data-component-id",
            MODULE.candidate_group(
                [
                    {
                        "leafId": 5,
                        "occurrence": 0,
                        "strokeIndices": [1],
                        "familyKey": "5",
                        "color": "#fff",
                        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>',
                    }
                ],
                7,
                [
                    {},
                    {
                        "start": [0, 0],
                        "curveList": [{"command": "h", "parameterList": [1]}],
                    },
                ],
            ),
        )

    def test_serializes_real_command_boundaries_not_bezier_controls(self):
        stroke = {
            "start": [2, 3],
            "curveList": [
                {"command": "h", "parameterList": [5]},
                {"command": "c", "parameterList": [99, 99, 88, 88, 4, 6]},
            ],
        }
        self.assertEqual(
            MODULE.stroke_boundaries(stroke),
            [[2.0, 3.0], [7.0, 3.0], [11.0, 9.0]],
        )

    def test_mask_ids_are_scoped_to_the_unicode_source_case(self):
        entry = {
            "leafId": 5,
            "occurrence": 0,
            "strokeIndices": [0],
            "familyKey": "5/6",
            "color": "#123456",
            "svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>',
        }
        markup = MODULE.candidate_panels(
            {"definitions": "", "useAttributes": {"href": "#glyph"}},
            7,
            [entry],
            [{"start": [0, 0], "curveList": []}],
            "A",
            "u4e00-g",
        )
        self.assertIn('id="u4e00-g-mask-7-0"', markup)
        self.assertIn('mask="url(#u4e00-g-mask-7-0)"', markup)


if __name__ == "__main__":
    unittest.main()
