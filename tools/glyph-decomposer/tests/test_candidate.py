import pytest

from glyph_decomposer.candidate import compile_stroke_seeds
from glyph_decomposer.grammar import GlyphRepository


def test_compile_stroke_seeds_applies_recursive_ids_affines():
    records = [
        {
            "id": 1,
            "type": "component",
            "strokes": [
                {
                    "feature": "横",
                    "start": [0, 50],
                    "curveList": [{"command": "h", "parameterList": [100]}],
                }
            ],
        },
        {
            "id": 2,
            "type": "compound",
            "operator": "⿸",
            "references": [{"id": 1}, {"id": 1}],
        },
    ]
    repository = GlyphRepository(records)

    seeds = compile_stroke_seeds(repository, 2)

    assert [(seed.leaf_id, seed.occurrence) for seed in seeds] == [(1, 0), (1, 1)]
    assert seeds[0].points[0] == pytest.approx((0, 50))
    assert seeds[0].points[-1] == pytest.approx((100, 50))
    assert seeds[1].points[0] == pytest.approx((40, 65))
    assert seeds[1].points[-1] == pytest.approx((90, 65))
    assert [seed.component_path for seed in seeds] == [(0,), (1,)]


def test_compile_stroke_seeds_samples_directed_cubic_curve():
    repository = GlyphRepository(
        [
            {
                "id": 1,
                "type": "component",
                "strokes": [
                    {
                        "feature": "撇",
                        "start": [90, 10],
                        "curveList": [
                            {
                                "command": "c",
                                "parameterList": [-10, 20, -40, 70, -80, 80],
                            }
                        ],
                    }
                ],
            }
        ]
    )

    seed = compile_stroke_seeds(repository, 1)[0]

    assert seed.points[0] == pytest.approx((90, 10))
    assert seed.points[-1] == pytest.approx((10, 90))
    assert len(seed.points) > 2


def test_compile_stroke_seeds_applies_confirmed_component_order_correction():
    repository = GlyphRepository(
        [
            {
                "id": 486,
                "type": "component",
                "strokes": [
                    {
                        "feature": "竖",
                        "start": [37, 6],
                        "curveList": [{"command": "v", "parameterList": [88]}],
                    },
                    {
                        "feature": "横",
                        "start": [7, 50],
                        "curveList": [{"command": "h", "parameterList": [30]}],
                    },
                    {
                        "feature": "竖",
                        "start": [63, 6],
                        "curveList": [{"command": "v", "parameterList": [88]}],
                    },
                    {
                        "feature": "横",
                        "start": [63, 50],
                        "curveList": [{"command": "h", "parameterList": [30]}],
                    },
                ],
            }
        ]
    )

    seeds = compile_stroke_seeds(repository, 486)

    assert [seed.feature for seed in seeds] == ["横", "竖", "竖", "横"]
    assert seeds[0].points[0] == pytest.approx((7, 50))
    assert seeds[1].points[0] == pytest.approx((37, 6))
