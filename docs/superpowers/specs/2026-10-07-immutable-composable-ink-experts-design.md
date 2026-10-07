# Immutable, Composable Ink Experts

## Status

Design approved in conversation on 2026-10-07. This specification defines a
local experimental expert runtime and historical oracle benchmark. It does not
authorize `/admin` integration, automatic source selection, or database writes.

## Intent

Preserve every materially distinct ink-decoding model as an immutable expert,
reproduce the best behaviour previously observed at historical commits, and
make those experts safely composable without letting one expert corrupt
another expert's search universe.

The first milestone must answer two questions with evidence:

1. Can the exact historical experts collectively reproduce the per-case peaks
   already observed on the twelve directed-stroke annotations?
2. Which component families, topology conditions, and resource profiles define
   each expert's real capability boundary?

The runtime must also establish interfaces that later permit an expert to ask
one or more specialists for a bounded proposal, compare the result with an
unassisted shadow run, and materialize a successful collaboration as a new
immutable fusion expert.

## Confirmed Problem

The current `ensemble` decoder does not contain the historical models. Its
`contact-retrace` and `baseline-complete` branches share the current
`build_skeleton_graph`, `rank_routes`, `choose_routes`, scoring rules, and most
hard constraints. They isolate beam state, but not model behaviour.

On the twelve available cases, only U+65E8-T and U+808E-T actually ran two
experts. Four cases accepted the first expert without fallback, and six cases
skipped the second expert because their raw route count exceeded the fallback
budget. A selector cannot recover a historical solution that was never
generated.

The historical audit establishes these initial expert milestones:

| Expert ID | Commit | Distinguishing capability |
|---|---|---|
| `classic-15c` | `15c2b04` | restored classic complete-stroke baseline |
| `junction-23b` | `23b9179` | complete route topology at junctions |
| `scan-front-7fc` | `7fc2c16` | closed-box recovery from diagonal scan fronts |
| `grass-005` | `0058190` | ambiguous grass-head recovery |
| `recursive-ids-8c` | `8c585f2` | recursive IDS structure and sibling order |
| `complete-route-2272` | `2272a6d` | complete directed route construction |
| `contact-dd` | `dd1dd05` | stronger contact/retrace constraints |
| `reserved-dad` | `dad0107` | completed-stroke residual reservation |

These are candidates, not an assertion that every commit deserves permanent
production support. The oracle benchmark decides which experts add independent
value.

## Design Principles

### Immutable lineage

An expert is content-addressed and never edited in place. Its identity includes:

- stable expert ID and schema version;
- exact Git commit;
- source-tree hash;
- command template and interpreter/runtime identity;
- configuration and learned-rule hashes;
- parent expert IDs, if cloned or fused;
- declared capability tags and resource class;
- golden-case result hashes.

Changing code, configuration, rules, or dependencies creates a new expert ID.
Cloning one expert and improving it creates a child. Combining several experts
creates a fusion expert whose manifest records all parents. Historical experts
remain runnable.

### Isolation before composition

Each expert owns its candidate table, beam states, route transformations,
occupied ink, caches local to the run, and final decision. Experts share only
immutable input artifacts and protocol messages. No expert may delete or
mutate another expert's candidates or ledger.

### Truth isolation

Annotations are passed only to post-prediction evaluation. Expert selection,
specialist routing, early stopping, and fusion inference may not read truth.
Oracle selection is explicitly benchmark-only and must be marked
`usesHumanTruth: true`; production-style runs must remain false.

### Evidence before promotion

A collaboration is not silently folded into an existing expert. It first runs
as an observed composition, then as a reproducible fusion candidate, and only
becomes a new expert after golden regressions pass.

## Architecture

### Expert registry

A tracked registry stores immutable manifests, while heavyweight worktrees and
run products remain under ignored `.local/` directories. The registry validates
unique IDs, resolvable commits, hashes, parent lineage, command templates, and
protocol compatibility.

The initial registry is intentionally small and hand-reviewed. Capability tags
such as `grass-head`, `closed-box`, `recursive-ids`, `junction-topology`, and
`contact-retrace` describe routing hints, not proof of correctness. Measured
capability profiles are generated separately by the benchmark.

### Historical worktree runner

The runner creates or reuses detached worktrees at
`.local/expert-worktrees/<expert-id>/`. It executes the exact script from the
pinned commit in a subprocess and passes absolute paths for the shared PDF,
bbox cache, candidates, learned rules, and annotations.

Historical CLI and audit schemas differ. Each manifest therefore names a thin
adapter which:

- builds the command supported by that commit;
- records stdout, stderr, exit status, wall time, and resource observations;
- normalizes the resulting audit into the common schema;
- preserves the raw audit and interactive HTML unchanged;
- never patches the historical worktree.

A failed or unsupported expert produces a normalized failure record rather
than disappearing from the matrix.

### Common case registry

The first benchmark contains the twelve currently evidenced source cases:

- U+6418-G, U+6424-J, U+6461-G, U+6485-J;
- U+64CE-T, U+64CF-H, U+64EC-T, U+65E8-T;
- U+66DA-J, U+66DA-N, U+6726-J, U+808E-T.

Each case record fixes Unicode, source, glyph ID, annotation path, and input
artifact hashes. Absolute machine-specific paths are resolved at runtime and
are not committed.

### Normalized run record

Every expert/case execution produces a common record containing:

- expert identity and hashes;
- case identity and input hashes;
- command and environment summary;
- predicted and expected stroke counts;
- status, review reasons, and completeness;
- stroke and component metrics when truth is available;
- route count, unexplained ink, score, and expert-native evidence;
- duration, peak memory where available, and timeout/budget outcome;
- paths to raw audit and HTML;
- `usesHumanTruth` and the exact phase in which truth was first read.

The oracle matrix chooses the best post-evaluation result per case and reports
both the winning expert and every runner-up. It never feeds that choice back
into inference.

## Instant Specialist Consultation

Later milestones may allow an expert to request bounded help during its own
search through a language-neutral broker.

### Request boundary

A `ProposalRequest` identifies:

- the caller, run, and immutable input hashes;
- Unicode source and candidate tree;
- an IDS subtree or contiguous stroke interval;
- residual-ink and skeleton snapshots by content hash;
- required capability tags;
- accepted proposal schema;
- depth, wall-time, CPU, memory, and candidate-count budgets;
- the visited expert set.

Requests must align with an IDS/component boundary whenever one exists. An
arbitrary crop or a mutable caller beam is not a valid interoperability
boundary.

### Proposal response

A specialist returns an immutable `ProposalBundle` containing candidate routes
or component assignments, confidence evidence, assumptions, resource cost,
and complete provenance. The caller may add those proposals to its own search,
but cannot expose its mutable state to the specialist or rewrite the
specialist's evidence.

Consultation forms a directed acyclic call graph. The broker rejects recursion
cycles, repeated expert/input pairs, excess depth, and exhausted budgets.
Content-addressed responses are cached.

### Shadow comparison

Every assisted run retains an unassisted shadow result or a reproducible link
to one. Reports attribute changes to individual proposal bundles:

- candidates added and selected;
- hard failures removed or introduced;
- coverage and cost changes;
- runtime and memory overhead;
- final metric delta in benchmark mode.

The review HTML must allow switching between unassisted and assisted universes
and highlighting the exact subtree supplied by each specialist.

## Fusion Lifecycle

A useful collaboration can be promoted without mutating its parents:

1. Capture the caller manifest, specialist manifests, broker policy, and
   successful proposal trace.
2. Generate a fusion manifest with those experts as parents.
3. Replace incidental run IDs and caches with deterministic policy rules.
4. Run all parent golden cases plus the case that motivated the fusion.
5. Register the fusion under a new immutable ID only if it preserves required
   parent regressions and records any intentional trade-off.

Fusion may remain an orchestration graph; it does not need to copy source code
into one file. A later optimized reimplementation is another child expert and
must prove output equivalence or document divergence.

## Scheduling and Parallelism

There are three independent parallelism levels:

1. Different cases and top-level experts can run in isolated processes.
2. Specialists for independent IDS subtrees can run concurrently.
3. Beam expansion within one expert can parallelize a single depth only when
   deterministic ordering and tie-breaking are preserved.

The scheduler must not run every expert blindly. It estimates cost from stroke
count, route hypotheses, component depth, historical duration, and memory. It
then applies:

- capability-based shortlist before launch;
- cheap experts before expensive experts;
- bounded worker pools rather than unbounded fan-out;
- early stop only when a truth-free completeness and confidence contract is
  satisfied;
- per-expert and per-consultation time, route, and memory budgets;
- cache reuse for identical immutable inputs;
- deterministic output ordering independent of completion order.

The first milestone defaults to one worker for reproducibility and records
enough timing data to simulate alternative schedules. Parallel execution is
enabled only after serial outputs are proven identical.

## CPU, GPU, and Language Strategy

### Current workload

The expensive path is dominated by branch-heavy whole-character beam search,
Python object creation, dictionaries, sets/frozensets, topology checks, and
state copying. Existing observations show sustained CPU use during the long
runs. The code has no CUDA execution path. A GPU would not automatically
accelerate this control-heavy representation.

Mask operations, pairwise distances, rasterization, IoU, and batched local
route scoring are possible accelerator kernels. Before adding a GPU backend,
the runner must profile wall time by phase. GPU work is justified only when a
vectorizable phase is a measured bottleneck after transfer and launch costs.

### Language boundary

Python remains the first-milestone orchestrator because all historical experts,
PDF/OpenCV tooling, audit generation, and exploratory rules already exist in
Python. Dynamic manifests and subprocess composition suit rapid expert
variation, while process isolation avoids relying on the GIL for top-level
parallelism.

The expert protocol is language-neutral. A Julia, Rust, C++, or other runner
may participate if it consumes and emits the same versioned messages.

- Julia is attractive for numerical experiments and multiple dispatch, but
  does not remove the need to redesign the current object/set-heavy beam and
  would duplicate the existing PDF/HTML integration.
- Rust is the preferred candidate for a future deterministic graph/bitset/beam
  kernel because the hot problem is discrete search, memory layout, and safe
  concurrency rather than dense numerical training.
- GPU kernels may be implemented behind either runtime without changing the
  expert protocol.

No language rewrite belongs to the first milestone. Profiling and protocol
stability come first.

## First Milestone

The first milestone implements only:

1. manifest and case registries;
2. detached historical worktree provisioning;
3. per-commit command adapters;
4. normalized run records;
5. serial execution of the historical expert/case matrix;
6. oracle and capability reports with raw HTML links;
7. deterministic rerun and truth-isolation tests.

It may define and test the consultation message schemas, but it does not yet
allow experts to call each other. It records CPU and wall-time measurements but
does not add GPU code or rewrite an expert in another language.

## Test Strategy

Implementation is test-first.

### Registry and lineage

- duplicate IDs, missing commits, hash mismatches, cycles, and incompatible
  schemas are rejected;
- clone and fusion manifests preserve immutable parent identities;
- changing code, config, or rules changes expert identity.

### Worktree and adapter safety

- provisioning is idempotent and never mutates or removes an unrelated
  worktree;
- adapters construct only commands supported by their pinned commit;
- timeout, non-zero exit, missing output, and malformed historical audit become
  explicit normalized failures;
- raw historical artifacts remain byte-for-byte unchanged.

### Truth isolation

- prediction commands can run without annotation access;
- normalized prediction records exist before evaluation begins;
- oracle code is physically separate from production-style selection code;
- every record states whether and when truth was read.

### Oracle benchmark

- all twelve cases appear for every enabled expert, including failures;
- known historical peaks are reproduced within declared deterministic
  tolerances;
- the report identifies independent expert wins rather than only an average;
- rerunning from warm worktrees and caches yields identical predictions and
  normalized hashes.

### Protocol contract

- request and proposal schemas round-trip independently of Python classes;
- content hashes change when an input snapshot or proposal changes;
- cycle, depth, and budget guards reject invalid consultation graphs;
- provenance survives composition and prospective fusion.

## Acceptance Criteria

The first milestone is accepted when:

- every registered historical expert is reproducibly tied to an exact commit
  and content hash;
- the twelve-case matrix completes or records an explicit failure in every
  cell;
- stored historical peaks are either reproduced or explained by a specific
  input, dependency, or nondeterminism difference;
- the oracle demonstrates whether the expert family covers more cases than the
  current HEAD ensemble;
- no prediction reads human truth;
- no run modifies `/admin`, remote data, or repository character/glyph data;
- runtime and resource reports are sufficient to design the next scheduler;
- old experts remain unchanged after adding a child or fusion manifest.

## Non-goals

- Selecting a production expert from human truth.
- Applying any proposal to upstream character data.
- Replacing `/admin` or the existing glyph workflow.
- Running all experts for every Unicode character.
- Adding GPU dependencies before profiling.
- Rewriting the decoder in Julia, Rust, or C++ during the oracle milestone.
- Hiding collaboration inside shared mutable state.

## Risks and Mitigations

- **Historical dependency drift:** record interpreter and dependency identity;
  normalize failure rather than silently patching a worktree.
- **Worktree disk growth:** create only registered commits, reuse by expert ID,
  and report size; cleanup remains an explicit user operation.
- **Combinatorial expert calls:** enforce DAG, depth, cost, and candidate budgets
  at the broker.
- **False capability labels:** distinguish declared tags from benchmark-derived
  profiles.
- **Metric overfitting:** report per-case winners, topology failures, and visual
  artifacts; do not promote on mean IoU alone.
- **Fusion coupling:** keep protocol messages immutable and parent manifests
  runnable; a fusion cannot replace its parents.
