"""Canonical minimal geometry for Chinese stroke features.

Repository curve commands describe how hanzi-chai renders a component. They
intentionally suppress some conditional decorations (for example, 竖钩 may be
stored as one vertical command). This grammar instead records the minimal
human-readable centreline program of the stroke feature itself.
"""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class StrokeGrammar:
    feature: str
    commands: tuple[str, ...]
    direction_vectors: tuple[Point, ...]
    segment_ratios: tuple[float, ...]
    tool: str

    @property
    def control_point_count(self) -> int:
        return 1 + sum(3 if command in {"c", "z"} else 1 for command in self.commands)


def _grammar(feature, commands, vectors, ratios, tool):
    total = sum(ratios)
    return StrokeGrammar(
        feature,
        tuple(commands),
        tuple(vectors),
        tuple(value / total for value in ratios),
        tool,
    )


_GRAMMARS = {
    "横": _grammar("横", "h", ((1, 0),), (1,), "line"),
    "竖": _grammar("竖", "v", ((0, 1),), (1,), "line"),
    "提": _grammar("提", "l", ((1, -0.45),), (1,), "line"),
    "撇": _grammar("撇", "c", ((-0.55, 1),), (1,), "bezier"),
    "平撇": _grammar("平撇", "c", ((-1, 0.25),), (1,), "bezier"),
    "捺": _grammar("捺", "c", ((0.75, 1),), (1,), "bezier"),
    "平捺": _grammar("平捺", "c", ((1, 0.35),), (1,), "bezier"),
    "点": _grammar("点", "c", ((0.45, 1),), (1,), "bezier"),
    "竖钩": _grammar("竖钩", ("v", "l"), ((0, 1), (-1, -0.35)), (0.88, 0.12), "polyline"),
    "横钩": _grammar("横钩", ("h", "l"), ((1, 0), (-0.45, 0.7)), (0.88, 0.12), "polyline"),
    "横折": _grammar("横折", ("h", "v"), ((1, 0), (0, 1)), (0.5, 0.5), "polyline"),
    "竖折": _grammar("竖折", ("v", "h"), ((0, 1), (1, 0)), (0.5, 0.5), "polyline"),
    "竖提": _grammar("竖提", ("v", "l"), ((0, 1), (1, -0.45)), (0.7, 0.3), "polyline"),
    "横撇": _grammar("横撇", ("h", "c"), ((1, 0), (-0.65, 0.8)), (0.45, 0.55), "mixed"),
    "横折钩": _grammar(
        "横折钩",
        ("h", "v", "l"),
        ((1, 0), (0, 1), (-0.55, -0.35)),
        (0.42, 0.48, 0.1),
        "polyline",
    ),
    "竖弯钩": _grammar(
        "竖弯钩", ("c", "c"), ((0.2, 1), (1, -0.15)), (0.55, 0.45), "bezier"
    ),
    "弯钩": _grammar(
        "弯钩", ("c", "c"), ((0.3, 1), (-0.35, -0.7)), (0.82, 0.18), "bezier"
    ),
}


def canonical_stroke_grammar(
    feature: str, fallback_commands: tuple[str, ...]
) -> StrokeGrammar:
    known = _GRAMMARS.get(feature)
    if known is not None:
        return known
    vectors = tuple(
        (1.0, 0.0)
        if command == "h"
        else (0.0, 1.0)
        if command == "v"
        else (1.0, 1.0)
        for command in fallback_commands
    )
    ratios = tuple(1.0 for _ in fallback_commands)
    tool = "bezier" if all(x in {"c", "z"} for x in fallback_commands) else "mixed"
    return _grammar(feature, fallback_commands, vectors, ratios, tool)
