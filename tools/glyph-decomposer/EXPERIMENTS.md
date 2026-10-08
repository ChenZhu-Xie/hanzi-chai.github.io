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

### Whole-stroke inertial ownership

Treating sampled centerline points as independent Voronoi owners allowed one
site's cell to reach disconnected PDF contours, producing small remote colour
islands. The solver now keeps the original directed stroke as the unit of
identity:

1. split every clipped Voronoi cell into connected vector atoms;
2. accept an atom as a trusted core only when it touches a narrow tube around
   its complete candidate stroke;
3. grow remaining ink from the nearest trusted whole-stroke core;
4. retain mutual exclusion and the existing writing-order junction rule.

On `U+6418-G`, without access to annotations during inference:

- root classified-ink accuracy: 99.74% (previously 98.90%);
- recursive four-leaf accuracy: 96.69% (previously 95.60%);
- mean leaf IoU: 85.89% (previously 84.09%);
- minimum leaf IoU: 65.50% (previously 64.18%);
- root directly anchored ink: 81.50%; inherited contour ink: 18.50%;
- nested `4158` directly anchored ink: 97.11%; inherited ink: 2.89%.

The clean `U+65E8-T` case still selects exact enumeration and retains 100%
root and recursive component accuracy. Removing `annotationPath` again yields
an identical inference payload after evaluation-only fields are removed.

### Skeleton-only inertia target

Region ownership was temporarily set aside to recover the more fundamental
object: one directed pen route from pen-down to pen-up. The historic
`U+6418-G` decoder and thirteen human trajectories are rendered on the same
PDF skeleton without coloured ink regions.

A conservative blind continuation follows an unfinished route through an
unambiguous corridor. At a junction it proceeds only when one outgoing tangent
clearly wins; straight `横`, `竖` and `提` routes additionally preserve their
cumulative axis. This improves mean endpoint errors:

- pen-down: 3.60% to 3.22% of the canvas diagonal;
- pen-up: 3.47% to 3.09%;
- the fifth `竖` no longer turns into another stroke (52 erroneous added
  pixels reduced to 2).

Human points are also snapped to the PDF skeleton and joined through their
ordered intermediate landmarks. This teacher-forced normalization reaches
0.24% mean pen-down error and 0.47% mean pen-up error. It explicitly reads
truth and defines the route target for later blind inference; it is not an
inference result.

### Next unresolved layer

The remaining errors are concentrated inside the `439`/`133` surround. The
stroke-identity layer now supplies local continuity and junction ownership;
the next step is to infer each stroke's exact visible extent around serif and
brush-detail contours without using annotation truth or global ink-area
voting.

## Annotation-independent PDF skeleton baseline

Before attempting another directed-stroke predictor, the tool now isolates the
part that can be solved by a small deterministic algorithm. It extracts the
chart glyph as vector geometry, samples only PDF ink, and applies Zhang-Suen
thinning. The historical annotation-canvas normalization is fixed at 8%
padding; no annotation is available to generation or parameter selection.

The generated skeleton is frozen before directed human paths are loaded for
evaluation. On all six complete annotation files currently available (82
directed strokes), using a 1.5/100 canvas-unit tolerance:

- mean human-path coverage by the PDF skeleton: 97.15%;
- mean PDF-skeleton coverage by human paths: 93.71%;
- mean human endpoint distance to the skeleton: 0.59/100 canvas units;
- best path coverage: 99.92% on `U+6418-G`;
- lowest path coverage: 91.35% on `U+65E8-T`.

Visual review confirms that the principal centreline topology is present for
all six glyphs. Most remaining skeleton-only pixels are legitimate unannotated
font details at stroke caps or small calligraphic spurs, while the human route
sometimes deliberately stops short of them. Therefore these coverage numbers
measure geometric agreement, not directed stroke-recognition accuracy.

This establishes a useful boundary: a deterministic algorithm is sufficient
for the undirected PDF skeleton, but it cannot uniquely recover stroke order,
direction, or ownership at shared intersections. Those require candidate
stroke/IDS constraints on top of this fixed graph; they should not be hidden
inside skeleton generation.

### Lossless topology compression

The binary skeleton is now compressed into connected critical-pixel clusters
and maximal non-branching chains. Synthetic straight-line, T-junction,
X-junction and closed-loop tests confirm exact pixel conservation. The same
invariant holds on all six real glyphs: all 7,924 skeleton pixels are owned
exactly once by a node cluster or chain.

The raw six-glyph graph contains 361 nodes and 373 chains. Its endpoint and
junction counts deliberately still include serif/calligraphic branches. In
particular, `U+6418-G` contains 22 raw junction clusters although its thirteen
standard strokes need fewer semantic decisions. No branch is pruned at this
stage: candidate strokes and recursive IDS constraints must first explain the
valid through-routes, after which only unsupported terminal residue can be
classified as decoration. This prevents a short but real hook or dot from
being destroyed by a length-only heuristic.

### Compact topology descriptors

Each maximal chain now has a Ramer-Douglas-Peucker description with a fixed
0.75-pixel deviation bound. The original chain pixels remain available for
audit, while the compact layer records normalized length, chord ratio, total
turn, endpoint tangents, normalized bounds, endpoint incidence and a
continuous terminal-shortness feature. It does not classify or remove a
branch.

At each junction, outward branch tangents are converted into all possible
through-pairs, ranked by deviation from a straight continuation. Candidate
stroke type and IDS placement can therefore select among explicit local
alternatives instead of searching raw pixels.

Across the six real glyphs, chain paths shrink from 7,126 points to 1,047
points (85.66% fewer) without changing nodes, edges or pixel ownership. The
HTML can independently toggle the full skeleton, full chains, compact
descriptors and post-generation human paths. This layer is the intended input
to recursive candidate/IDS path coverage; raw evidence remains available when
a compact decision is ambiguous.

### Candidate constraint graph

Repository candidates are now compiled into the same compact reasoning
vocabulary before any PDF matching. Each directed stroke retains its order,
feature, leaf ID, repeated-leaf occurrence and recursive IDS path. Every IDS
node records its operator, expected scope, child hierarchy and descendant
stroke indices. Pairwise stroke relations record exact contact or separation
and the nearest normalized position along both directed strokes.

On real repository data, candidate `17973` (`U+6418-G`) compiles to thirteen
strokes, four leaf occurrences, 78 pairwise relations and thirteen exact
contacts. Candidate `127685` (`U+65E8-T`) compiles to six strokes, two leaf
occurrences, fifteen pairwise relations and six exact contacts. Neither audit
reads PDF geometry or annotation truth.

This is where joined left/right, upper/lower or surround components begin to
receive evidence for separation, but no PDF skeleton is cut yet. The next
stage must enumerate local logical-cut alternatives at compatible raw
junctions and evaluate each complete future against an explicit no-cut
baseline. A high local geometric score alone is insufficient: the selected
hypothesis must improve whole-candidate route coverage, preserve required
stroke contacts, satisfy recursive IDS placement and avoid unexplained long
residue. Low-margin alternatives remain unresolved rather than being silently
cut.
