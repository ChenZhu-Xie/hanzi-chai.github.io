# Deterministic decomposition experiments

## U+6418-G / candidate 17973

Human annotations are loaded only after inference. Re-running without
`annotationPath` produced an identical decomposition tree.

### PDF contour audit

The chart glyph is one SVG path containing five closed contours, while the
candidate has thirteen directed strokes. The font has already merged
intersecting strokes, so SVG subpaths cannot be treated as strokes.

### Root and axis-aligned recursion

- root `⿰(220, 9313)`: 97.17% classified-ink accuracy;
- nested `⿱(4158, 506)`: clean zero-crossing split;
- three-terminal mean IoU: 90.90%, minimum IoU: 87.59%.

### Upper-left surround attempts

An L-shaped `⿸(439, 133)` cut crossed real strokes. Its two tested variants
reached only 83.01% and 86.42% accuracy. The solver now refuses this result
when its best boundary crosses more than 1% of source ink.

### Vector atom assignment

Candidate directed centerlines are laid out through IDS transforms, aligned to
the PDF vector ink without truth, sampled into vector Voronoi atoms, and each
atom is assigned exactly once to a stroke owner and leaf component.

- 184 mutually exclusive vector atoms;
- 93.48% of aligned seed samples lie on PDF ink;
- alignment IoU: 37.04%;
- four-leaf classified-ink accuracy: 93.65%;
- mean leaf IoU: 81.31%; minimum leaf IoU: 63.03%;
- reconstruction error: approximately `4.7e-17`.

One self-consistency alignment pass was attempted but rejected because it did
not improve the truth-free alignment objective.

### Rejected area regularizer

Moving the boundary halfway toward the children’s stroke-count ratio improved
the intrinsic balance score but reduced held-out accuracy from 93.65% to
93.27%. Stroke count is therefore not a reliable proxy for ink area and this
regularizer is not part of the implementation.

### Stroke-identity refinement

Vector atoms are now owned first by directed candidate strokes and only then
merged into leaf components. At exact intersections, writing order gives the
junction point to the later stroke; the surrounding ink remains partitioned
by local centerline distance. Two truth-free invariants guard acceptance:

- each stroke's PDF-visible seed remains continuous through permitted junction
  neighborhoods;
- different strokes own mutually exclusive ink.

For `U+6418-G`, this changes the root `⿰(220, 9313)` from a mechanical
vertical cut to stroke-aware ownership because the best straight cut crosses
real ink. Results:

- root classified-ink accuracy: 98.90% (previously 97.17%);
- recursive four-leaf accuracy: 95.60% (previously 93.65%);
- mean leaf IoU: 84.09% (previously 81.31%);
- minimum leaf IoU: 64.18% (previously 63.03%);
- root: 331 atoms, 13 strokes and 13 junctions;
- nested `4158`: 172 atoms, 6 strokes and 6 junctions;
- measured continuity and independence: 100% for both atom partitions.

This path is deliberately gated. A clean axis partition with no crossed ink
keeps exact enumeration; forcing vector atoms on the clean `U+65E8` split
reduced accuracy from 100% to 97.57%.

### Next unresolved layer

The remaining errors are concentrated inside the `439`/`133` surround. The
stroke-identity layer now supplies local continuity and junction ownership;
the next step is to infer each stroke's exact visible extent around serif and
brush-detail contours without using annotation truth or global ink-area
voting.
