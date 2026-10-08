import json

from glyph_decomposer.skeleton_editor import render_skeleton_editor


def _case():
    return {
        "key": "U+4E00-G-1",
        "unicode": "U+4E00",
        "character": "一",
        "source": "G",
        "candidateGlyphId": 1,
        "baseFingerprint": "test-fingerprint",
        "skeletonPath": "M0,0h1v1h-1z",
        "leaves": [
            {
                "key": "1@root",
                "glyphId": 1,
                "path": [],
                "colour": "#2563eb",
                "strokeCount": 1,
            }
        ],
        "tree": {
            "glyphId": 1,
            "path": [],
            "kind": "component",
            "operator": None,
            "strokeIndices": [0],
            "children": [],
        },
        "edges": [],
        "nodes": [],
        "proposedCutNodes": [],
    }


def test_editor_exposes_all_manual_operations_and_comments():
    page = render_skeleton_editor([_case()])

    assert "裁剪枝丫" in page
    assert "删除整链" in page
    assert "节点切口" in page
    assert "边内切口" in page
    assert "改归属到叶" in page
    assert "所选对象及 comment" in page
    assert "整字总结 comment" in page
    assert "hanzi-chai-skeleton-edits" in page


def test_editor_embeds_case_without_ascii_escaping_hanzi():
    page = render_skeleton_editor([_case()])
    data = page.split("const CASES=", 1)[1].split("; const NS", 1)[0]

    assert json.loads(data)[0]["character"] == "一"
