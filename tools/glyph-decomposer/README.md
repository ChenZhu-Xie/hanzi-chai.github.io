# hanzi-chai glyph decomposer

This package is the deterministic, outline-first decomposition prototype. It
does not write to the hanzi-chai API or database.

The prototype performs independently testable operations:

1. extract one Unicode chart glyph as vector geometry from Poppler's page SVG;
2. compile a repository glyph into a recursive component program;
3. recursively partition binary `⿰` and `⿱` programs with exact bounded
   enumeration;
4. try a constrained `⿸` partition, but stop with an explicit reason when a
   clean geometric boundary does not exist; then use directed candidate
   centerlines to induce mutually exclusive vector Voronoi ink atoms, guarded
   by truth-free alignment and non-degeneracy checks;
5. evaluate root and recursive terminal regions against a human annotation
   that inference never reads.

Run from this directory with `uv run glyph-decomposer --help`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for accepted and rejected geometric
approaches and their held-out measurements.
