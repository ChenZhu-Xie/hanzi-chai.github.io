from glyph_decomposer.stroke_grammar import canonical_stroke_grammar


def test_hook_and_fold_features_restore_human_minimal_geometry():
    assert canonical_stroke_grammar("竖钩", ("v",)).commands == ("v", "l")
    assert canonical_stroke_grammar("竖钩", ("v",)).control_point_count == 3
    assert canonical_stroke_grammar("横折", ("h", "v")).control_point_count == 3
    assert canonical_stroke_grammar("横折钩", ("h", "v")).control_point_count == 4


def test_curved_strokes_have_fixed_cubic_complexity():
    assert canonical_stroke_grammar("撇", ("c",)).control_point_count == 4
    assert canonical_stroke_grammar("点", ("c",)).control_point_count == 4
    assert canonical_stroke_grammar("竖弯钩", ("v", "h")).commands == ("c", "c")
    assert canonical_stroke_grammar("竖弯钩", ("v", "h")).control_point_count == 7


def test_unknown_feature_preserves_repository_command_count():
    grammar = canonical_stroke_grammar("unknown", ("h", "v", "c"))

    assert grammar.commands == ("h", "v", "c")
    assert grammar.control_point_count == 6
