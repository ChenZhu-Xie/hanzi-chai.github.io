# hanzi-chai glyph decomposer

This package is the deterministic, outline-first decomposition prototype. It
does not write to the hanzi-chai API or database.

The first vertical slice performs three independently testable operations:

1. extract one Unicode chart glyph as vector geometry from Poppler's page SVG;
2. compile a repository glyph into a recursive component program;
3. choose a complete root IDS partition with exact bounded enumeration and evaluate it
   against a human annotation that inference never reads.

Run from this directory with `uv run glyph-decomposer --help`.
