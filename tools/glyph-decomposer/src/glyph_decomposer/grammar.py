"""Compile repository glyph records into coordinate-free component programs."""

from __future__ import annotations

import json
from pathlib import Path

from .domain import ComponentProgram


class GlyphRepository:
    def __init__(self, records: list[dict]):
        self._records = {int(record["id"]): record for record in records}

    @classmethod
    def load(cls, path: Path) -> GlyphRepository:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def compile(self, glyph_id: int, active: tuple[int, ...] = ()) -> ComponentProgram:
        if glyph_id in active:
            cycle = " -> ".join(map(str, (*active, glyph_id)))
            raise ValueError(f"cyclic glyph references: {cycle}")
        try:
            record = self._records[glyph_id]
        except KeyError as error:
            raise KeyError(
                f"glyph {glyph_id} is absent from repository data"
            ) from error

        kind = record["type"]
        if kind == "component":
            return ComponentProgram(
                glyphId=glyph_id,
                kind="component",
                strokeFeatures=tuple(
                    stroke.get("feature", "unknown")
                    for stroke in record.get("strokes", ())
                ),
            )
        if kind != "compound":
            raise ValueError(f"unsupported glyph type {kind!r} for {glyph_id}")
        children = tuple(
            self.compile(int(reference["id"]), (*active, glyph_id))
            for reference in record.get("references", ())
        )
        return ComponentProgram(
            glyphId=glyph_id,
            kind="compound",
            operator=record.get("operator"),
            children=children,
        )
