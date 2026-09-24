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


def load_leaf_evaluator():
    path = Path(__file__).with_name("unihan-leaf-topology-eval.py")
    spec = importlib.util.spec_from_file_location("unihan_leaf_topology_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEAF_EVAL = load_leaf_evaluator()


class PdfCellTests(unittest.TestCase):
    def test_topology_overlay_uses_large_distinct_node_colors(self):
        image = Image.new("RGB", (160, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.line((20, 30, 80, 30), fill="black", width=5)
        draw.line((50, 30, 50, 90), fill="black", width=5)
        draw.line((100, 30, 100, 90), fill="black", width=5)
        draw.line((100, 90, 145, 90), fill="black", width=5)

        overlay = MONTAGE.topology_overlay(image)
        pixels = np.asarray(overlay)

        for key in ("endpoint", "junction"):
            color = MONTAGE.TOPOLOGY_COLORS[key]
            matches = np.all(pixels == color, axis=2)
            self.assertGreater(matches.sum(), 40)
        turn_color = MONTAGE.TOPOLOGY_COLORS["corner"]
        turn_pixels = np.all(pixels == turn_color, axis=2)
        self.assertGreater(turn_pixels.sum(), 10)
        endpoints, junctions, skeleton = MONTAGE.topology_points(
            np.asarray(image.convert("L")) < 224
        )
        corners = MONTAGE.corner_points(skeleton, endpoints, junctions)
        x, y = corners[0]
        self.assertEqual(overlay.getpixel((x, y)), turn_color)

    def test_marks_a_degree_two_right_angle_as_a_corner(self):
        binary = np.zeros((128, 128), dtype=bool)
        binary[24:81, 40] = True
        binary[80, 40:101] = True
        endpoints, junctions, skeleton = MONTAGE.topology_points(binary)
        corners = MONTAGE.corner_points(skeleton, endpoints, junctions)

        self.assertEqual(len(endpoints), 2)
        self.assertEqual(len(junctions), 0)
        self.assertTrue(any(abs(x - 40) <= 4 and abs(y - 80) <= 4 for x, y in corners))

    def test_does_not_mark_diagonal_raster_stair_steps_as_corners(self):
        binary = np.zeros((128, 128), dtype=bool)
        for value in range(20, 101):
            binary[value, value] = True
        endpoints, junctions, skeleton = MONTAGE.topology_points(binary)

        corners = MONTAGE.corner_points(skeleton, endpoints, junctions)

        self.assertEqual(corners, [])

    def test_topology_point_distance_uses_positions_not_only_counts(self):
        same = [[0.1, 0.2], [0.8, 0.9]]
        shifted = [[0.1, 0.8], [0.8, 0.2]]

        self.assertEqual(LEAF_EVAL.point_set_distance(same, same), 0)
        self.assertGreater(LEAF_EVAL.point_set_distance(same, shifted), 0)

    def test_full_skeleton_distance_distinguishes_line_direction(self):
        horizontal = Image.new("L", (128, 128), "white")
        ImageDraw.Draw(horizontal).line((15, 64, 113, 64), fill="black", width=5)
        vertical = Image.new("L", (128, 128), "white")
        ImageDraw.Draw(vertical).line((64, 15, 64, 113), fill="black", width=5)

        self.assertEqual(LEAF_EVAL.raster_skeleton_distance(horizontal, horizontal), 0)
        self.assertGreater(LEAF_EVAL.raster_skeleton_distance(horizontal, vertical), 0)

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
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <rect width="100" height="100" fill="white"/>
          <circle cx="50" cy="50" r="20" fill="red"/>
        </svg>"""

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

        signature = MONTAGE.topology_signature(
            Image.fromarray((~binary).astype(np.uint8) * 255)
        )
        self.assertEqual(len(signature["endpointPositions"]), 3)
        self.assertEqual(len(signature["junctionPositions"]), 1)
        self.assertGreater(len(signature["skeletonPositions"]), 3)
        self.assertTrue(
            all(
                0 <= coordinate <= 1
                for point in signature["skeletonPositions"]
                for coordinate in point
            )
        )

    def test_local_topology_rejects_one_curved_stroke_as_split_strokes(self):
        image = Image.new("L", (160, 160), "white")
        draw = ImageDraw.Draw(image)
        for y in (35, 70, 105):
            draw.line((20, y, 140, y), fill="black", width=5)
        draw.line((92, 18, 92, 78), fill="black", width=5)
        draw.line((92, 78, 52, 145), fill="black", width=5)

        evidence = MONTAGE.horizontal_crossing_consensus(image)

        self.assertEqual(evidence["decision"], "abstain")

    def test_local_topology_finds_offset_vertical_and_falling_crossings(self):
        image = Image.new("L", (160, 160), "white")
        draw = ImageDraw.Draw(image)
        for y in (35, 70, 105):
            draw.line((20, y, 140, y), fill="black", width=5)
        draw.line((96, 18, 96, 145), fill="black", width=5)
        draw.line((72, 84, 36, 145), fill="black", width=5)

        evidence = MONTAGE.horizontal_crossing_consensus(image)

        self.assertEqual(evidence["decision"], "split-vertical-and-falling-strokes")
        self.assertIn(evidence["consensus"], {"2/3", "3/3"})

    def test_combines_pdf_crossings_with_candidate_stroke_identity(self):
        evidence = {"decision": "split-vertical-and-falling-strokes"}
        candidates = {
            "18132": [{"hasSeparateVerticalAndFallingLeaves": False}],
            "900104": [{"hasSeparateVerticalAndFallingLeaves": True}],
        }

        choice = MONTAGE.choose_split_stroke_candidate(evidence, candidates)

        self.assertEqual(choice["decision"], "candidate")
        self.assertEqual(choice["candidateId"], 900104)

    def test_candidate_match_abstains_when_structure_is_not_unique(self):
        evidence = {"decision": "split-vertical-and-falling-strokes"}
        candidates = {
            "1": [{"hasSeparateVerticalAndFallingLeaves": True}],
            "2": [{"hasSeparateVerticalAndFallingLeaves": True}],
        }

        choice = MONTAGE.choose_split_stroke_candidate(evidence, candidates)

        self.assertEqual(choice["decision"], "abstain")

    def test_structure_guided_focus_abstains_instead_of_claiming_touching_other_part(
        self,
    ):
        size = 128
        pdf = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(pdf)
        draw.line((32, 12, 32, 92), fill="black", width=7)
        draw.line((32, 70, 110, 70), fill="black", width=7)
        candidate = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(candidate)
        draw.line((32, 12, 32, 92), fill="#2563eb", width=7)
        draw.line((32, 70, 110, 70), fill="#16a34a", width=7)

        focused = MONTAGE.structure_guided_focus(pdf, [candidate, candidate], "2563eb")
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

        focused = MONTAGE.structure_guided_focus(pdf, [candidate, candidate], "2563eb")
        pdf_mask = np.asarray(focused[0].convert("L")) < 224

        self.assertTrue(pdf_mask.any())
        self.assertEqual(MONTAGE.topology_signature(focused[0])["components"], 1)

    def test_candidate_topology_uses_unoccluded_target_only_layer(self):
        size = 128
        pdf = Image.new("RGB", (size, size), "white")
        ImageDraw.Draw(pdf).line((15, 64, 113, 64), fill="black", width=7)

        full = Image.new("RGB", (size, size), "white")
        full_draw = ImageDraw.Draw(full)
        full_draw.line((15, 64, 113, 64), fill="#2563eb", width=7)
        # Simulate a later-drawn neighbouring component obscuring the center.
        full_draw.rectangle((52, 58, 76, 70), fill="black")

        target_only = Image.new("RGB", (size, size), "white")
        ImageDraw.Draw(target_only).line((15, 64, 113, 64), fill="#2563eb", width=7)

        occluded = MONTAGE.structure_guided_focus(pdf, [full], "2563eb")
        isolated = MONTAGE.structure_guided_focus(pdf, [full], "2563eb", [target_only])
        occluded_ink = (np.asarray(occluded[1].convert("L")) < 224).sum()
        isolated_ink = (np.asarray(isolated[1].convert("L")) < 224).sum()

        self.assertGreater(isolated_ink, occluded_ink)

    def test_complete_target_window_keeps_every_disconnected_subpart(self):
        size = 128
        pdf = Image.new("RGB", (size, size), "white")
        pdf_draw = ImageDraw.Draw(pdf)
        pdf_draw.rectangle((52, 18, 96, 52), outline="black", width=5)
        pdf_draw.line((72, 76, 98, 76), fill="black", width=5)

        candidate = Image.new("RGB", (size, size), "white")
        candidate_draw = ImageDraw.Draw(candidate)
        candidate_draw.rectangle((52, 18, 96, 52), outline="#f59e0b", width=5)
        candidate_draw.line((72, 76, 98, 76), fill="#f59e0b", width=5)

        focused = MONTAGE.structure_guided_focus(
            pdf,
            [candidate],
            "f59e0b",
            complete_target_window=True,
        )

        self.assertGreaterEqual(
            MONTAGE.topology_signature(focused[0])["components"], 2
        )

    def test_marks_the_same_feature_in_glyph_and_topology_rows(self):
        montage = Image.new("RGB", (768, 580), "white")

        annotated = MONTAGE.draw_feature_annotations(
            montage,
            {
                "subject": "left component 1019 vs 4110",
                "annotations": [
                    {
                        "candidate": "B",
                        "pdf": [0.4, 0.7],
                        "glyph": [0.5, 0.75],
                        "pdfTopology": [0.45, 0.65],
                        "topology": [0.5, 0.68],
                        "text": "B=4110: center vertical stops at lower horizontal",
                    }
                ],
            },
            True,
        )

        self.assertGreater(annotated.height, montage.height)
        pixels = np.asarray(annotated)
        glyph_y = 36 + round(0.75 * 256)
        topology_y = 324 + round(0.68 * 256)
        x = 2 * 256 + round(0.5 * 256)
        self.assertTrue(
            (pixels[glyph_y - 12 : glyph_y + 13, x - 12 : x + 13] != 255).any()
        )
        self.assertTrue(
            (pixels[topology_y - 12 : topology_y + 13, x - 12 : x + 13] != 255).any()
        )


if __name__ == "__main__":
    unittest.main()
