from types import SimpleNamespace

import numpy as np

from glyph_decomposer.primitive_audit import (
    _adaptive_leaf_alignment_solution,
    _compile_audit_candidate,
    _ids_segmentation,
    _leaf_stroke_order_issues,
    _truth_consistency,
    load_blind_case_specs,
    render_primitive_fit_audit,
)


def test_compile_audit_candidate_prefers_reviewed_geometry_for_ternary_ids(
    monkeypatch,
):
    reviewed = object()
    repository = object()
    compiled = SimpleNamespace(root=SimpleNamespace(operator="⿳", children=()))
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_catalog_candidate_graph",
        lambda catalog, codepoint, candidate_id: reviewed,
    )
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_candidate_graph",
        lambda repository, candidate_id: compiled,
    )

    result = _compile_audit_candidate(repository, {}, 0x66DA, 57149)

    assert result is reviewed


def test_compile_audit_candidate_keeps_repository_geometry_for_binary_ids(monkeypatch):
    repository = object()
    compiled = SimpleNamespace(root=SimpleNamespace(operator="⿰", children=()))
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_catalog_candidate_graph",
        lambda catalog, codepoint, candidate_id: (_ for _ in ()).throw(
            AssertionError("catalog should not run for a binary repository candidate")
        ),
    )
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_candidate_graph",
        lambda actual_repository, candidate_id: compiled,
    )

    result = _compile_audit_candidate(repository, {}, 0x66DA, 57149)

    assert result is compiled


def test_compile_audit_candidate_uses_catalog_when_repository_has_no_record(
    monkeypatch,
):
    reviewed = object()
    repository = object()
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_candidate_graph",
        lambda actual_repository, candidate_id: (_ for _ in ()).throw(KeyError()),
    )
    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.compile_catalog_candidate_graph",
        lambda catalog, codepoint, candidate_id: reviewed,
    )

    result = _compile_audit_candidate(repository, {}, 0x66DA, 57149)

    assert result is reviewed


def test_truth_consistency_reports_stale_order_and_leaf_label():
    payload = {"metadata": {"candidateStrokeOrder": ["竖", "点"]}}
    truth = [
        {"label": "486", "points": [[1, 1], [9, 1]]},
        {"label": "133", "points": [[2, 2], [4, 8]]},
    ]

    result = _truth_consistency(payload, ["横", "点"], [486, 1128], truth)

    assert result["valid"] is False
    assert [issue["kind"] for issue in result["issues"]] == [
        "candidate-stroke-order",
        "leaf-id",
    ]


def test_truth_consistency_warns_about_obvious_direction_mismatch():
    payload = {"metadata": {"candidateStrokeOrder": ["点"]}}
    truth = [{"label": "499", "points": [[5, 2], [5.1, 20]]}]

    result = _truth_consistency(payload, ["点"], [499], truth)

    assert result["valid"] is True
    assert result["directionWarnings"][0]["reason"] == (
        "dot-candidate-has-nearly-vertical-truth-chord"
    )


def test_leaf_order_audit_detects_a_leaf_local_path_permutation():
    candidate = SimpleNamespace(
        strokes=(
            SimpleNamespace(
                index=0,
                component_path=(1, 0),
                leaf_id=499,
                points=((4, 0), (4, 10)),
            ),
            SimpleNamespace(
                index=1,
                component_path=(1, 0),
                leaf_id=499,
                points=((7, 0), (7, 10)),
            ),
            SimpleNamespace(
                index=2,
                component_path=(1, 0),
                leaf_id=499,
                points=((1, 2), (2, 5)),
            ),
        )
    )
    truth = [
        {"points": [[21, 24], [23, 30]]},
        {"points": [[26, 20], [26, 40]]},
        {"points": [[32, 20], [32, 40]]},
    ]

    issues = _leaf_stroke_order_issues(candidate, truth)

    assert issues[0]["kind"] == "leaf-stroke-order"
    assert issues[0]["leafId"] == 499
    assert issues[0]["permutation"] == [1, 2, 0]


def test_leaf_order_audit_accepts_affine_layout_change_with_same_order():
    candidate = SimpleNamespace(
        strokes=(
            SimpleNamespace(
                index=0,
                component_path=(0,),
                leaf_id=282,
                points=((0, 0), (0, 10)),
            ),
            SimpleNamespace(
                index=1,
                component_path=(0,),
                leaf_id=282,
                points=((0, 0), (10, 0), (10, 10)),
            ),
            SimpleNamespace(
                index=2,
                component_path=(0,),
                leaf_id=282,
                points=((0, 10), (10, 10)),
            ),
        )
    )
    truth = [
        {"points": [[20, 30], [20, 80]]},
        {"points": [[20, 30], [70, 30], [70, 80]]},
        {"points": [[20, 80], [70, 80]]},
    ]

    assert _leaf_stroke_order_issues(candidate, truth) == []


def test_adaptive_leaf_pool_skips_an_ids_solution_below_trigger(monkeypatch):
    solution = SimpleNamespace(ids_structure_cost=0.1, alignments=(), score=5.0)
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "glyph_decomposer.primitive_audit.search_leaf_alignments", unexpected
    )
    pools = {(0,): ("small",)}

    output_pools, output_solution, refinements = _adaptive_leaf_alignment_solution(
        np.zeros((2, 2), dtype=bool), object(), pools, solution
    )

    assert output_pools is pools
    assert output_solution is solution
    assert refinements == ()
    assert called is False


def test_ids_segmentation_unions_selected_leaf_bounds_up_the_tree():
    first = SimpleNamespace(
        glyph_id=220,
        path=(0,),
        kind="component",
        operator=None,
        bounds=(0, 0, 40, 100),
        stroke_indices=(0,),
        children=(),
    )
    second = SimpleNamespace(
        glyph_id=9313,
        path=(1,),
        kind="component",
        operator=None,
        bounds=(40, 0, 100, 100),
        stroke_indices=(1,),
        children=(),
    )
    root = SimpleNamespace(
        glyph_id=17973,
        path=(),
        kind="compound",
        operator="⿰",
        bounds=(0, 0, 100, 100),
        stroke_indices=(0, 1),
        children=(first, second),
    )
    candidate = SimpleNamespace(
        root=root,
        strokes=(
            SimpleNamespace(component_path=(0,), leaf_id=220),
            SimpleNamespace(component_path=(1,), leaf_id=9313),
        ),
    )
    selected = {
        (0,): SimpleNamespace(bounds=(1, 2, 31, 90)),
        (1,): SimpleNamespace(bounds=(45, 5, 80, 88)),
    }
    fits = [
        {"strokeIndex": 0, "componentPath": [0], "route": [[2, 3], [30, 89]]},
        {"strokeIndex": 1, "componentPath": [1], "route": [[46, 6], [79, 87]]},
    ]

    segmentation = _ids_segmentation(candidate, selected, fits)

    assert segmentation["root"]["actualBounds"] == [1, 2, 80, 90]
    assert [leaf["glyphId"] for leaf in segmentation["leaves"]] == [220, 9313]
    assert segmentation["leaves"][1]["strokeIndices"] == [1]


def test_primitive_report_contains_component_and_recursive_ids_views():
    html = render_primitive_fit_audit([{"unicode": "U+6418"}])

    assert "按叶部件着色" in html
    assert 'id="ids-overlay"' in html
    assert 'id="ids-tree"' in html


def test_load_blind_case_specs_accepts_array_and_normalizes_unicode(tmp_path):
    source = tmp_path / "blind.json"
    source.write_text(
        '[{"unicode":"U+827E","source":"H","candidateGlyphId":127412}]',
        encoding="utf-8",
    )

    assert load_blind_case_specs(source) == [
        {"unicode": 0x827E, "source": "H", "candidateGlyphId": 127412}
    ]


def test_load_blind_case_specs_rejects_missing_candidate(tmp_path):
    source = tmp_path / "blind.json"
    source.write_text('[{"unicode":"U+827E","source":"H"}]', encoding="utf-8")

    try:
        load_blind_case_specs(source)
    except ValueError as error:
        assert "candidateGlyphId" in str(error)
    else:
        raise AssertionError("invalid blind case must be rejected")


def test_primitive_report_labels_cases_without_directed_truth():
    html = render_primitive_fit_audit(
        [
            {
                "unicode": "U+827E",
                "character": "艾",
                "source": "H",
                "candidateGlyphId": 127412,
                "truthAvailable": False,
            }
        ]
    )

    assert "人工有向真值未提供" in html
    assert "expectedStrokeCount" in html
