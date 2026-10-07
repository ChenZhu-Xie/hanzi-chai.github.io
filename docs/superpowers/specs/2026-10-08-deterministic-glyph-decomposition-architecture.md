# Deterministic Glyph Decomposition Architecture

## Status

Proposed on 2026-10-08. This document defines the architecture for a new,
single-pipeline deterministic decomposer. It does not authorize database writes
or replacement of the existing `/admin` workflow. The existing ink-expert
system remains a benchmark and source of regression fixtures, not the runtime
architecture of the new decomposer.

## Goal

Given one regional source glyph extracted as a vector outline from the Unicode
chart, determine:

1. which recursively nested hanzi-chai components form the glyph;
2. which sibling form of each component is present, or whether a new sibling is
   required;
3. which ordered, directed strokes explain each leaf component;
4. how every part of the observed ink is accounted for;
5. why the chosen decomposition is better than every competing candidate.

Component decomposition is the product. Stroke recovery is supporting evidence
and a consistency check. Character, component, and stroke inference must still
be solved together because each level constrains the other two.

## Technology Decision

### Algorithm language: Python 3.12

Python is the primary language for the deterministic geometry and optimization
pipeline. It is an orchestration language here: the expensive work runs in
native GEOS, NumPy, SciPy, OpenCV, and OR-Tools kernels.

The Python project is managed by `uv`, uses a normal `src/` package layout, and
is locked independently from Bun. Tests use `pytest`; formatting and linting use
`ruff`; public domain models are type-checked.

Reasons:

- the current experiments and truth fixtures are already Python and NumPy;
- the strongest public glyph-segmentation references are Python-oriented;
- Shapely/GEOS and OR-Tools provide mature native geometry and combinatorial
  optimization without implementing either foundation ourselves;
- algorithm rules and diagnostic certificates can be iterated much faster than
  in C++ or Rust;
- Windows x86-64 has prebuilt wheels for the selected stack.

Python code must not become another monolithic research script. The current
`unihan-ink-diffusion.py` is treated as a behavioural reference only.

### Product language: existing TypeScript, Bun, and React

The existing application remains responsible for:

- `/admin` interaction and visualization;
- loading audit/proposal artifacts;
- authenticated API calls and database mutation;
- final human selection and apply operations.

The production website does not require a Python runtime. The decomposer first
runs as a local/offline CLI and emits portable artifacts that `/admin` can load.
This keeps the upstream application architecture intact.

### Geometry and optimization stack

- `fontTools` and a PDF vector extractor: Bézier paths and glyph outlines;
- `Shapely 2.x` / GEOS: planar arrangements, polygon repair, overlays, faces,
  intersections, containment, and spatial indexes;
- NumPy and SciPy: numeric arrays, distance transforms, registration, sparse
  graphs, and bounded continuous optimization;
- OpenCV: raster diagnostics and fallback measurements only, never the primary
  glyph representation;
- OR-Tools CP-SAT: Boolean assignment, exclusivity, candidate selection, and
  global constraint solving;
- a min-cost-flow solver for subproblems that are genuinely flows;
- SVG/HTML generated from vector results for human verification.

NetworkX may be used in small prototypes and tests, but not as a required hot
path. PyTorch and GPU runtimes are deliberately excluded.

## Why Not Another Primary Language

### TypeScript

TypeScript remains ideal for the UI but lacks the mature computational geometry
and exact optimization stack needed by the solver. Implementing the core in
TypeScript would either duplicate difficult geometry or depend on fragile WASM
bindings before the algorithm is stable.

### Rust

Rust is a possible later optimization boundary, not the first implementation
language. Its memory safety and WASM support are attractive, but porting the
rapidly changing inference rules now would slow the work and would not repair a
wrong representation or objective function. A stable, profiled kernel may later
move behind a PyO3 interface without changing the domain protocol.

### C++

C++ offers CGAL and native OR-Tools, but has the highest iteration and Windows
build cost. Existing C/C++ geometry is already reached through Python wheels.

### Julia

Julia is suitable for numerical research but offers no decisive advantage for
this geometry-heavy stack, has weaker integration with the Bun application, and
would introduce a third ecosystem without removing a current constraint.

## System Boundary

The initial executable is a deterministic CLI, not a web service:

```text
glyph-decomposer audit   request.json --output result.json
glyph-decomposer solve   request.json --output result.json
glyph-decomposer render  result.json  --output review.html
```

For interactive use, the same process may later run as a long-lived JSON Lines
worker over stdin/stdout. A versioned JSON Schema is the language-neutral
contract between Python and TypeScript. No Python object, NumPy array, pickle,
or database handle crosses the boundary.

The CLI is read-only with respect to upstream data. It produces proposals and
proof artifacts. Existing TypeScript code performs any selected apply action.

## Domain Model

### `SourceGlyph`

- Unicode code point and IRG source;
- original vector outline and its provenance;
- normalized coordinate transform;
- planar ink faces, holes, and boundary features;
- derived medial evidence and ambiguous junction zones.

### `CandidateProgram`

- candidate glyph ID and recursive component tree;
- IDS operators and ordered child occurrences;
- leaf component IDs and sibling-family membership;
- ordered, directed stroke grammar for each leaf;
- provenance of every structural assertion.

Candidate coordinates are explicitly absent from the semantic program. They may
be retained only as weak diagnostics.

### `InkGraph`

A lossless-enough planar representation derived from the outline:

- disjoint faces whose union reconstructs the observed ink;
- robust outline concavities and terminals;
- stable medial branches outside distorted junction zones;
- explicit T, X, Y, L, half, terminal, and uncertain junction records;
- adjacency and possible continuation relations.

### `Hypothesis`

A hypothesis binds one candidate subtree to observed faces and medial branches.
It contains its hard feasibility state, soft costs, unexplained ink, consumed
evidence, and a structured explanation. Stroke hypotheses are composed into
leaf-component hypotheses; component hypotheses are recursively composed into
the full character.

### `SolutionCertificate`

- selected glyph and component IDs;
- ordered strokes and owned/shared ink regions;
- hard constraints satisfied or violated;
- objective terms with human-readable evidence;
- runner-up hypotheses and score margins;
- minimal conflicting subtree when no candidate is feasible;
- recommendation: accept, review sibling, create sibling, or create component.

## Inference Architecture

### 1. Vector ingestion and normalization

Extract and retain the original PDF paths. Repair winding, self-intersections,
tiny loops, duplicate edges, and page transforms. Rasterization is generated
only for diagnostics and comparison.

### 2. Outline-first geometric decomposition

Following the StrokeStyles and constrained-Delaunay family of methods:

- compute interior/exterior medial evidence and local width;
- identify convex and concave curvilinear outline features;
- detect and temporarily isolate distorted intersection zones;
- connect compatible concavities to form a planar arrangement;
- split the ink into stable faces and reliable stroke-like segments;
- preserve enough information to reconstruct the exact original outline.

This replaces the current practice of reducing the glyph to a noisy skeleton
before structural reasoning begins.

### 3. Candidate compilation

Compile each hanzi-chai candidate into a declarative constraint program:

- recursive IDS layout;
- child order and containment relations;
- leaf occurrence identity;
- stroke count, order, category, direction, and turn grammar;
- expected intersection/contact matrix among strokes;
- sibling-invariant and sibling-distinguishing features.

A constraint is hard within a candidate hypothesis. If the target contradicts
it, that candidate is rejected; the observation is not distorted to fit it.

### 4. Hierarchical hypothesis generation

Solve the IDS tree from leaves upward while retaining a small deterministic
`k`-best frontier per subtree:

- propose paths/regions for one stroke from reliable geometric segments;
- compose strokes into a leaf component;
- compose sibling components according to the parent IDS operator;
- discard dominated or geometrically impossible states immediately;
- propagate consumed ink, topology, bounding relations, and costs upward.

The IDS tree therefore reduces the global combinatorial search instead of
merely adding a late score.

### 5. Local exact solving at ambiguity zones

Use CP-SAT only where several feasible continuations or face assignments remain.
Typical Boolean variables answer:

- which stroke owns a face or reliable medial segment;
- which continuation is selected at a junction;
- which sibling candidate is active;
- whether an explicit overlap/junction region is shared.

Hard constraints require connected strokes, consistent candidate structure,
one active candidate per occurrence, full accounting of ink, and no reuse of a
main path. Final visible coloring may be mutually exclusive while the latent
model explicitly represents physical overlap at a junction.

This bounded use of CP-SAT is cheaper and easier to diagnose than one whole-
glyph unconstrained beam search.

### 6. Global character decision

The root compares complete character hypotheses using lexicographic priorities:

1. hard feasibility;
2. complete component and ink accounting;
3. topology and stroke grammar agreement;
4. recursive IDS spatial agreement;
5. boundary reconstruction and deformation cost;
6. weak stylistic similarity.

No weighted visual term may compensate for a hard topological contradiction.

### 7. Directed stroke recovery

After regions and component identities are stable, recover each stroke's
directed centerline within its assigned region. Stroke order and direction are
used to reject component decompositions and produce review animation; they are
not allowed to pull the component segmentation toward a visually convenient
but structurally wrong candidate.

### 8. Explanation and human correction

Every decision must be inspectable in the existing HTML/admin style. A human
correction records a domain fact such as a junction continuation, sibling ID,
or component boundary. It is not stored as a special-case coordinate patch.
The same fact is replayed against all relevant confirmed cases.

## Hard and Soft Evidence

Hard invariants include:

- valid planar geometry and recursive tree construction;
- the candidate's declared number and order of leaf strokes;
- connectivity of each selected stroke region;
- full accounting of ink at solution completion;
- no unexplained reuse of reliable main-path evidence;
- explicit handling of every ambiguous junction or overlap;
- satisfaction of the active IDS operator's qualitative relation.

Soft evidence includes:

- exact coordinates, proportions, and curvature from another font;
- component and stroke centroids;
- elastic outline similarity;
- local stroke width and decorative terminals;
- regional writing-style priors.

Evidence is tagged by provenance: observed target geometry, repository-confirmed
structure, human confirmation, writing standard, or candidate-derived prior.

## Resource Model

- CPU only;
- one glyph solved in one process initially;
- deterministic seeds and stable tie-breaking;
- cache PDF extraction, repaired outlines, planar arrangements, compiled
  candidates, and subtree hypotheses by content hash;
- use hierarchical pruning before CP-SAT;
- impose time and state limits and return the best certified partial result;
- parallelize independent glyphs only after single-glyph profiling is stable.

GPU acceleration is not part of the architecture. If profiling later identifies
a stable geometry kernel consuming at least a substantial fraction of runtime,
that kernel alone may be ported to Rust/C++.

## Repository Layout

Proposed layout, to be created only after this architecture is accepted:

```text
tools/glyph-decomposer/
  pyproject.toml
  uv.lock
  src/glyph_decomposer/
    domain/
    ingest/
    geometry/
    grammar/
    hypotheses/
    solver/
    trajectory/
    explain/
    cli/
  tests/
    unit/
    golden/
    integration/

schemas/glyph-decomposition/
  request.schema.json
  result.schema.json
  annotation.schema.json
```

Existing `scripts/unihan-*.py` files remain regression fixtures and migration
sources until the new package reproduces their validated outputs. They are not
imported as production modules.

## Initial Milestones

1. Freeze the twelve human-annotated cases and upstream-confirmed glyphs as
   geometry, component, stroke, and explanation golden tests.
2. Define the JSON domain contract and candidate compiler without performing
   inference.
3. Implement reversible vector-outline planar decomposition and verify exact
   outline reconstruction.
4. Solve one leaf component family deterministically, including junction and
   overlap handling.
5. Compose leaves through a two-level IDS subtree and emit a proof artifact.
6. Compare one complete character against all sibling candidates without using
   truth during inference.
7. Only after the small golden set passes, run against newly upstream-confirmed
   characters and then unresolved characters.

## Architectural Success Criteria

- one deterministic pipeline, not a vote among historical experts;
- no raster-only irreversible representation in the primary path;
- every accepted decomposition accounts for all source ink;
- every decision is reproducible and has a machine-readable certificate;
- component correctness is evaluated directly, with stroke correctness as
  supporting evidence;
- a failed candidate produces a localized structural conflict;
- `/admin` remains the sole place where a human-authorized proposal can mutate
  upstream data;
- no GPU and no learned model are required for the baseline.

