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
   strokes to induce mutually exclusive vector Voronoi ink atoms, with local
   junction ownership, within-stroke continuity and between-stroke
   independence guarded by truth-free checks; disconnected contour fragments
   inherit ownership from trusted whole-stroke cores instead of isolated
   nearest-point sites;
5. evaluate root and recursive terminal regions against a human annotation
   that inference never reads.

Run from this directory with `uv run glyph-decomposer --help`.

To compile one repository candidate into recursive IDS scopes, directed
leaf-owned strokes and pairwise stroke-contact evidence without reading PDF or
human truth:

```powershell
uv run glyph-decomposer candidate-audit <request.json> --output <review.html>
```

To inspect the older directed decoder without any ink colouring, normalize its
routes and the human trajectories onto the same PDF skeleton:

```powershell
uv run glyph-decomposer trajectory-audit <historic-review.html> --output <review.html>
```

The report labels blind endpoint continuation separately from the
teacher-forced skeleton target. The latter intentionally reads human landmarks
and must never be reported as inference accuracy.

To generate an annotation-independent, undirected centre skeleton directly
from Unicode chart vector ink and then evaluate it against existing directed
human strokes:

```powershell
uv run glyph-decomposer skeleton-audit `
  --pdf <U4E00.pdf> `
  --bbox-cache <U4E00-bbox.html> `
  --annotations <annotation.json> [<annotation.json> ...] `
  --output <review.html>
```

The command first extracts and freezes every PDF skeleton, and only then loads
the annotation paths. It uses deterministic vector rasterisation followed by
connectivity-preserving Zhang-Suen thinning; it performs no fitting or model
training. It then losslessly contracts every maximal degree-two pixel chain
between endpoint/junction clusters into a topology graph. Every skeleton pixel
belongs to exactly one node cluster or chain, and the review can toggle and
inspect chain/node IDs. A bounded-error polyline then reduces redundant chain
samples while retaining the original pixels, and records normalized length,
curvature, endpoint tangents, spatial bounds and terminal-shortness evidence.
Each junction also exposes every possible through-pair ranked by angular
continuity. This solves the undirected centreline problem. Stroke order,
pen-down/pen-up direction and the correct continuation through an intersection
remain a separate constrained graph-decomposition problem.

The next matcher treats every proposed logical cut as an explicit hypothesis.
The unchanged/no-cut graph is always retained as the baseline; a cut may be
accepted automatically only when complete downstream stroke coverage improves
by a sufficient absolute score and margin without violating IDS, continuity or
exclusive ownership constraints.

See [EXPERIMENTS.md](EXPERIMENTS.md) for accepted and rejected geometric
approaches and their held-out measurements.
