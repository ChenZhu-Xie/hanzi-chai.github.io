import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_montage():
    path = Path(__file__).with_name("unihan-render-montage.py")
    spec = importlib.util.spec_from_file_location("unihan_render_montage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONTAGE = load_montage()


class PdfCellTests(unittest.TestCase):
    def test_enlarges_a_small_pdf_glyph_for_human_review(self):
        page = Image.new("L", (100, 100), "white")
        ImageDraw.Draw(page).rectangle((42, 42, 57, 57), fill="black")

        cell = MONTAGE.pdf_cell(page, (40, 40, 60, 60), (100, 100), size=256)
        ink = np.asarray(cell.convert("L")) < 224
        top, left = np.argwhere(ink).min(axis=0)
        bottom, right = np.argwhere(ink).max(axis=0) + 1

        self.assertGreater(right - left, 180)
        self.assertGreater(bottom - top, 180)

    def test_review_svg_keeps_red_annotation_points(self):
        svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <rect width="100" height="100" fill="white"/>
          <circle cx="50" cy="50" r="20" fill="red"/>
        </svg>'''

        image = MONTAGE.render_svg_review_image(svg.decode("utf-8"), size=100)

        red, green, blue = image.getpixel((50, 50))
        self.assertGreater(red, 200)
        self.assertLess(green, 50)
        self.assertLess(blue, 50)

    def test_extracts_endpoints_and_junction_from_a_t_shape(self):
        binary = np.zeros((101, 101), dtype=bool)
        binary[20:81, 49:52] = True
        binary[19:22, 20:81] = True

        endpoints, junctions, _skeleton = MONTAGE.topology_points(binary)

        self.assertEqual(len(endpoints), 3)
        self.assertEqual(len(junctions), 1)

    def test_structure_guided_focus_abstains_instead_of_claiming_touching_other_part(self):
        size = 128
        pdf = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(pdf)
        draw.line((32, 12, 32, 92), fill="black", width=7)
        draw.line((32, 70, 110, 70), fill="black", width=7)
        candidate = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(candidate)
        draw.line((32, 12, 32, 92), fill="#2563eb", width=7)
        draw.line((32, 70, 110, 70), fill="#16a34a", width=7)

        focused = MONTAGE.structure_guided_focus(
            pdf, [candidate, candidate], "2563eb"
        )
        pdf_mask = np.asarray(focused[0].convert("L")) < 224

        self.assertFalse(pdf_mask[120:136, 180:].any())

    def test_structure_guided_focus_claims_a_disconnected_target_component(self):
        size = 128
        pdf = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(pdf)
        draw.line((32, 12, 32, 60), fill="black", width=7)
        draw.line((60, 80, 110, 80), fill="black", width=7)
        candidate = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(candidate)
        draw.line((32, 12, 32, 60), fill="#2563eb", width=7)
        draw.line((60, 80, 110, 80), fill="#16a34a", width=7)

        focused = MONTAGE.structure_guided_focus(
            pdf, [candidate, candidate], "2563eb"
        )
        pdf_mask = np.asarray(focused[0].convert("L")) < 224

        self.assertTrue(pdf_mask.any())
        self.assertEqual(MONTAGE.topology_signature(focused[0])["components"], 1)


if __name__ == "__main__":
    unittest.main()
