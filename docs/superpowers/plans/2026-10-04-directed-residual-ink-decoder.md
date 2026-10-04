# Directed Residual-Ink Stroke Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace whole-skeleton ink ownership with a source-aware, sequential decoder that chooses directed stroke routes, gates junction exits, updates residual ink after every stroke, and renders auditable evidence.

**Architecture:** Keep `unihan-ink-diffusion.py` as the CLI and review renderer, but move the new invariants into four focused Python modules: directed half-edge topology, residual-ink ownership, source-aware order constraints, and sequential beam orchestration. The old decoder remains available as a baseline while `--decoder residual` exercises the new path; learned rules may rank only routes that pass deterministic constraints.

**Tech Stack:** Python 3, `unittest`, NumPy, OpenCV, SciPy, existing SVG/PDF extraction utilities, Bun/TypeScript verification for repository-wide regressions.

**Spec:** `docs/superpowers/specs/2026-10-04-directed-residual-ink-decoder-design.md`

## Global Constraints

- Do not call `createGlyph`, `updateCharacter`, or any remote write API.
- Candidate coordinates, size, and occupied area are not target-space truth.
- Stroke order and direction are source-aware; do not project G-source rules onto every IRG source.
- Exact verified source-character and leaf-component truths outrank generic order rules and learned priors.
- A learned score cannot make an illegal pen-down, route, junction exit, or component re-entry legal.
- Intersection evidence may be reused during inference; the later stroke exclusively owns the visible overlap.
- Uncertain source order, new components/siblings, infeasible residuals, and near-tied solutions remain `needs-review`.
- Generated PDF, annotation, model, HTML, and screenshot artifacts remain under ignored `.local/` or `artifacts/` paths.

## Review Focus

- Eight-neighbour raster skeletons can invent diagonal shortcuts beside an orthogonal junction; Task 1 tests that every transition still passes through the junction gate.
- A crossing pixel is reusable evidence but must not make the earlier stroke own a later branch; Task 2 tests inference and final display semantics separately.
- Generic top-to-bottom order must not override an exact T/H/J/K source sequence; Task 3 tests evidence precedence and source isolation.
- An early locally good route can make a later stroke impossible; Task 4 tests beam backtracking rather than greedy commitment.
- Held-out annotations and their derived start positions must never enter training or route selection; Task 5 tests leave-one-out isolation at the decoder boundary.

---

### Task 1: Directed Half-Edge Skeleton and Junction Gates

**Files:**
- Create: `scripts/unihan-directed-skeleton.py`
- Create: `scripts/test_unihan_directed_skeleton.py`
- Reference: `scripts/unihan-ink-diffusion.py:42-180`

**Interfaces:**
- Consumes: the existing `SkeletonGraph` shape (`points`, `point_index`, `matrix`, `critical`, `crossing`).
- Produces: `DirectedHalfEdge`, `JunctionGate`, `DirectedSkeleton`, `build_directed_skeleton(graph)`, `trace_route_edges(directed, route_points)`, and `legal_exit_edges(directed, incoming_edge_id, expected_sector, expected_turns)`.

- [ ] **Step 1: Write failing topology tests**

Add tests named:

- `test_cross_has_four_gated_roads_and_eight_directed_half_edges`
- `test_orthogonal_junction_has_no_diagonal_bypass`
- `test_horizontal_route_rejects_vertical_exits_at_cross`
- `test_t_junction_can_reuse_contact_without_selecting_branch`

Use small plus and T-shaped Boolean skeletons. Assert that each road terminates at a gate, reverse half-edges share one road ID, and an eastbound horizontal stroke returns only the east exit rather than north or south.

- [ ] **Step 2: Run tests and verify the intended failure**

Run: `python scripts/test_unihan_directed_skeleton.py`

Expected: FAIL because `unihan-directed-skeleton.py` and its interfaces do not exist.

- [ ] **Step 3: Implement the half-edge graph**

Implement these exact public types and signatures:

```python
@dataclass(frozen=True)
class DirectedHalfEdge:
    edge_id: int
    road_id: int
    start_gate: int
    end_gate: int
    point_indices: tuple[int, ...]
    start_sector: str
    end_sector: str

@dataclass(frozen=True)
class JunctionGate:
    gate_id: int
    point_indices: frozenset[int]
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]

@dataclass(frozen=True)
class DirectedSkeleton:
    edges: tuple[DirectedHalfEdge, ...]
    gates: tuple[JunctionGate, ...]
    pixel_to_edges: dict[int, tuple[int, ...]]

def build_directed_skeleton(graph) -> DirectedSkeleton: ...
def trace_route_edges(directed: DirectedSkeleton, graph, route_points: np.ndarray) -> tuple[int, ...]: ...
def legal_exit_edges(directed: DirectedSkeleton, incoming_edge_id: int, expected_sector: str, expected_turns: tuple[str, ...]) -> tuple[int, ...]: ...
```

Collapse clustered junction pixels into a single gate before tracing roads. Suppress diagonal raster adjacency when it merely bridges two orthogonal neighbours around the same gate.

- [ ] **Step 4: Run focused and existing topology tests**

Run: `python scripts/test_unihan_directed_skeleton.py; python scripts/test_unihan_ink_diffusion.py`

Expected: both PASS.

- [ ] **Step 5: Commit**

Run: `git add scripts/unihan-directed-skeleton.py scripts/test_unihan_directed_skeleton.py && git commit -m "feat: add directed skeleton junction gates"`

### Task 2: Residual-Ink Ledger and Route-Bounded Width Recovery

**Files:**
- Create: `scripts/unihan-residual-ink.py`
- Create: `scripts/test_unihan_residual_ink.py`
- Modify: `scripts/test_unihan_ink_diffusion.py`
- Reference: `scripts/unihan-ink-diffusion.py:1161-1270`

**Interfaces:**
- Consumes: `DirectedSkeleton`, traced edge IDs, the binary target mask, and ordered stroke index.
- Produces: `ResidualInkLedger`, `StrokeRegion`, `initial_ledger(target)`, `recover_stroke_region(...)`, and `advance_ledger(...)` for Task 4.

- [ ] **Step 1: Replace the old passing ownership assumption with failing invariants**

Retire `test_ink_width_follows_skeleton_ownership_across_a_junction` as the behavioural contract and add:

- `test_horizontal_width_stops_at_unselected_vertical_half_edges`
- `test_vertical_hook_does_not_claim_later_rising_stroke`
- `test_contact_disk_remains_reusable_after_first_stroke`
- `test_later_stroke_visibly_overpaints_shared_contact`
- `test_explained_non_contact_ink_is_not_available_to_later_start`

The grass fixture must assert zero horizontal ownership on both vertical arms outside the contact radius. The hand fixture must assert that the hook owns no pixel on the rising diagonal beyond their attachment disk.

- [ ] **Step 2: Run tests and verify RED**

Run: `python scripts/test_unihan_residual_ink.py; python scripts/test_unihan_ink_diffusion.py`

Expected: new tests FAIL against the old whole-skeleton owner flood.

- [ ] **Step 3: Implement ledger and route-bounded expansion**

Implement:

```python
@dataclass(frozen=True)
class ResidualInkLedger:
    unexplained: np.ndarray
    reusable_contact: np.ndarray
    visible_owner: np.ndarray
    consumed_edge_ids: frozenset[int]

@dataclass(frozen=True)
class StrokeRegion:
    mask: np.ndarray
    contact_mask: np.ndarray
    forbidden_leak_mask: np.ndarray

def initial_ledger(target: np.ndarray) -> ResidualInkLedger: ...
def recover_stroke_region(target: np.ndarray, graph, directed, route_edge_ids: tuple[int, ...], route_points: np.ndarray, ledger: ResidualInkLedger, contact_radius: int = 2) -> StrokeRegion: ...
def advance_ledger(ledger: ResidualInkLedger, region: StrokeRegion, stroke_index: int, consumed_edge_ids: tuple[int, ...]) -> ResidualInkLedger: ...
```

Expand from explicit route samples along local normals. Clip expansion at target outline, medial competition boundaries, and junction gates. Mark only narrow gate disks as reusable; explained non-contact ink is removed from `unexplained`. Assign `visible_owner[region.mask] = stroke_index`, so later strokes win visible overlap.

- [ ] **Step 4: Run residual and legacy suites**

Run: `python scripts/test_unihan_residual_ink.py; python scripts/test_unihan_ink_diffusion.py`

Expected: PASS; legacy functions remain callable only for baseline comparison.

- [ ] **Step 5: Commit**

Run: `git add scripts/unihan-residual-ink.py scripts/test_unihan_residual_ink.py scripts/test_unihan_ink_diffusion.py && git commit -m "feat: add residual ink ownership ledger"`

### Task 3: Source-Aware Stroke Grammar and Residual Pen-Down Selection

**Files:**
- Create: `scripts/unihan-stroke-order.py`
- Create: `scripts/test_unihan_stroke_order.py`
- Modify: `scripts/unihan-stroke-transfer.py`
- Modify: `scripts/test_unihan_stroke_transfer.py`

**Interfaces:**
- Consumes: repository candidate strokes, source letter, Unicode code point, optional local normative catalog, current residual ledger, and directed skeleton.
- Produces: `StrokeExpectation`, `OrderEvidence`, `NormativeCatalog`, `load_normative_catalog(path)`, `compile_stroke_expectations(...)`, and `rank_pen_down_candidates(...)` for Task 4.

- [ ] **Step 1: Write failing source and pen-down tests**

Add tests named:

- `test_old_head_selects_upper_short_horizontal_before_lower_long_horizontal`
- `test_after_subtracting_first_horizontal_next_horizontal_is_lower_one`
- `test_exact_verified_leaf_order_precedes_generic_top_to_bottom_rule`
- `test_t_source_catalog_does_not_leak_into_g_source`
- `test_unknown_source_rule_is_evidence_only_and_forces_review`
- `test_disconnected_grass_head_keeps_verified_horizontal_first_order`

Use a temporary catalog with entries shaped as:

```json
{"schemaVersion":1,"entries":[{"source":"T","unicode":"U+64CE","features":["横","竖","竖","横"],"provenance":"Taiwan MOE verified fixture"}]}
```

The old-head fixture must contain an upper short and lower long horizontal with identical direction so position is the deciding constraint.

- [ ] **Step 2: Run tests and verify RED**

Run: `python scripts/test_unihan_stroke_order.py; python scripts/test_unihan_stroke_transfer.py`

Expected: FAIL because source-aware compilation and residual pen-down ranking are absent.

- [ ] **Step 3: Implement evidence precedence and start selection**

Implement:

```python
@dataclass(frozen=True)
class OrderEvidence:
    level: int
    provenance: str
    rule: str
    hard: bool

@dataclass(frozen=True)
class NormativeEntry:
    source: str
    codepoint: int
    features: tuple[str, ...]
    provenance: str

@dataclass(frozen=True)
class NormativeCatalog:
    entries: dict[tuple[str, int], NormativeEntry]

@dataclass(frozen=True)
class StrokeExpectation:
    stroke_index: int
    component_id: int
    occurrence: int
    component_ordinal: int
    feature: str
    expected_sector: str
    expected_turns: tuple[str, ...]
    closes_component: bool
    evidence: tuple[OrderEvidence, ...]

@dataclass(frozen=True)
class PenDownCandidate:
    point_index: int
    outgoing_edge_ids: tuple[int, ...]
    score: float
    evidence: tuple[dict, ...]
    hard_rejections: tuple[str, ...]

def load_normative_catalog(path: Path | None) -> NormativeCatalog: ...
def compile_stroke_expectations(strokes: list[dict], source: str, codepoint: int, catalog: NormativeCatalog | None = None) -> list[StrokeExpectation]: ...
def rank_pen_down_candidates(expectation: StrokeExpectation, ledger, directed, graph, learned_rule: dict | None = None) -> list[PenDownCandidate]: ...
```

Use the six-level precedence from the spec. Apply top-to-bottom and left-to-right only within compatible component scope and stroke feature. Exclude explained non-contact pixels before scoring learned position priors.

- [ ] **Step 4: Run order, transfer, and diffusion tests**

Run: `python scripts/test_unihan_stroke_order.py; python scripts/test_unihan_stroke_transfer.py; python scripts/test_unihan_ink_diffusion.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add scripts/unihan-stroke-order.py scripts/test_unihan_stroke_order.py scripts/unihan-stroke-transfer.py scripts/test_unihan_stroke_transfer.py && git commit -m "feat: add source aware stroke order constraints"`

### Task 4: Sequential Residual Beam Decoder

**Files:**
- Create: `scripts/unihan-residual-decoder.py`
- Create: `scripts/test_unihan_residual_decoder.py`
- Modify: `scripts/unihan-ink-diffusion.py`
- Modify: `scripts/test_unihan_ink_diffusion.py`

**Interfaces:**
- Consumes: Task 1 directed skeleton, Task 2 ledger/region solver, Task 3 expectations/start candidates, and existing ranked `RouteCandidate` options.
- Produces: `ResidualDecoderState`, `ResidualDecodeResult`, and `decode_residual_routes(...)`; integrates `--decoder legacy|residual` into the existing CLI.

- [ ] **Step 1: Write failing sequential-decoder tests**

Add tests named:

- `test_decoder_consumes_strokes_against_updated_residual`
- `test_decoder_rejects_forbidden_junction_exit_before_scoring`
- `test_decoder_cannot_reenter_closed_component_occurrence`
- `test_beam_backtracks_when_best_early_route_blocks_later_stroke`
- `test_infeasible_later_stroke_returns_needs_review_without_forced_route`
- `test_learned_prior_cannot_resurrect_illegal_route`

The backtracking fixture must provide two legal first-stroke routes: the lower-cost route consumes the only feasible second route, while the runner-up permits completion. Assert that the completed history wins.

- [ ] **Step 2: Run tests and verify RED**

Run: `python scripts/test_unihan_residual_decoder.py`

Expected: FAIL because the sequential decoder does not exist.

- [ ] **Step 3: Implement sequential state expansion**

Implement:

```python
class RankedRoute(Protocol):
    points: np.ndarray
    pixels: frozenset[tuple[int, int]]
    score: float
    evidence: dict

@dataclass(frozen=True)
class ResidualDecoderState:
    stroke_index: int
    routes: tuple[RankedRoute, ...]
    route_edge_ids: tuple[tuple[int, ...], ...]
    ledger: ResidualInkLedger
    closed_occurrences: frozenset[tuple[int, int]]
    score: float
    evidence: tuple[dict, ...]

@dataclass(frozen=True)
class ResidualDecodeResult:
    routes: tuple[RankedRoute, ...]
    regions: tuple[StrokeRegion, ...]
    ledger: ResidualInkLedger
    status: str
    review_reasons: tuple[str, ...]
    best_score: float
    runner_up_margin: float | None
    steps: tuple[dict, ...]

def decode_residual_routes(target: np.ndarray, graph, strokes: list[dict], ranked_routes: list[list[RankedRoute]], source: str, codepoint: int, learned_model: dict | None = None, normative_catalog: NormativeCatalog | None = None, beam_width: int = 350) -> ResidualDecodeResult: ...
```

Filter hard constraints before adding numeric costs. Update the ledger after each accepted stroke. Permit reuse only through `reusable_contact`. Close `(component_id, occurrence)` after its final expected stroke and prohibit later re-entry.

- [ ] **Step 4: Integrate an explicit CLI switch**

Add `--decoder` with choices `legacy` and `residual`, defaulting to `legacy` during validation. Under `residual`, replace `choose_routes + geodesic_owners` with `decode_residual_routes`; keep output JSON fields backward-compatible and add `decoder`, `forbiddenBranchLeakage`, `residualUnexplainedInk`, and junction evidence.

- [ ] **Step 5: Run focused and full Python suites**

Run: `python scripts/test_unihan_residual_decoder.py; python -m unittest discover -s scripts -p "test_unihan_*.py"`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

Run: `git add scripts/unihan-residual-decoder.py scripts/test_unihan_residual_decoder.py scripts/unihan-ink-diffusion.py scripts/test_unihan_ink_diffusion.py && git commit -m "feat: decode strokes against residual ink"`

### Task 5: Leak-Free Learned Priors and Leave-One-Out Evaluation

**Files:**
- Modify: `scripts/unihan-ink-rule-training.py`
- Modify: `scripts/unihan-human-truth-loocv.py`
- Modify: `scripts/test_unihan_human_truth_loocv.py`
- Modify: `scripts/test_unihan_ink_diffusion.py`
- Modify: `docs/unihan-stroke-sequence-decoder.md`

**Interfaces:**
- Consumes: `ResidualDecodeResult` and existing exact-leaf learned examples.
- Produces: schema-version-2 rule models with source convention and constraint audit; residual-decoder LOOCV summaries that never expose held-out truth to prediction.

- [ ] **Step 1: Write failing isolation tests**

Add tests named:

- `test_residual_prediction_interface_cannot_receive_heldout_truth`
- `test_rule_signature_separates_source_conventions`
- `test_training_records_chosen_and_forbidden_junction_half_edges`
- `test_constraint_failure_is_review_not_negative_training_example`
- `test_loocv_reports_forbidden_branch_leakage_and_residual_ink`

Assert that the prediction call receives only the excluded case key, never its annotation path, masks, starts, or routes.

- [ ] **Step 2: Run tests and verify RED**

Run: `python scripts/test_unihan_human_truth_loocv.py; python scripts/test_unihan_ink_diffusion.py`

Expected: FAIL because the model schema and residual metrics are missing.

- [ ] **Step 3: Upgrade learned-rule schema and evaluation**

Change the model to `schemaVersion: 2`. Key reusable examples by source convention plus exact leaf signature, retain supporting case IDs, and record chosen/forbidden half-edges as explainable evidence. Keep minimum independent support at two after excluding the target case. Do not learn from states that violate deterministic constraints.

- [ ] **Step 4: Add residual metrics to LOOCV**

Report candidate correctness, mean pen-down error, directed DTW, per-stroke/component IoU, forbidden-branch leakage, residual unexplained ink, safe/needs-review counts, and hard-constraint violations. A result with any hard violation is never safe regardless of average score.

- [ ] **Step 5: Run training and LOOCV unit suites**

Run: `python scripts/test_unihan_human_truth_loocv.py; python scripts/test_unihan_ink_diffusion.py; python -m unittest discover -s scripts -p "test_unihan_*.py"`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

Run: `git add scripts/unihan-ink-rule-training.py scripts/unihan-human-truth-loocv.py scripts/test_unihan_human_truth_loocv.py scripts/test_unihan_ink_diffusion.py docs/unihan-stroke-sequence-decoder.md && git commit -m "feat: evaluate residual decoder without truth leakage"`

### Task 6: Named PDF Regressions and Auditable Review HTML

**Files:**
- Create: `scripts/unihan-residual-regression.py`
- Create: `scripts/test_unihan_residual_regression.py`
- Modify: `scripts/unihan-ink-diffusion.py`
- Modify: `scripts/test_unihan_ink_diffusion.py`
- Modify: `docs/unihan-directed-ink-diffusion.md`

**Interfaces:**
- Consumes: local Unicode PDF/cell cache/candidate data/annotations and Task 4 decoder output.
- Produces: a machine-readable named-regression report and interactive HTML showing residual states, pen-down alternatives, selected/forbidden half-edges, source-rule provenance, and model/cyan trajectories.

- [ ] **Step 1: Write failing report and HTML tests**

Add tests named:

- `test_named_regression_requires_grass_hand_and_old_head_cases`
- `test_report_fails_when_horizontal_owns_vertical_half_edge`
- `test_html_lists_pen_down_acceptance_and_rejection_reasons`
- `test_html_draws_selected_and_forbidden_half_edges_distinctly`
- `test_html_animates_residual_ink_after_each_stroke`
- `test_html_uses_later_stroke_colour_at_shared_contact`

- [ ] **Step 2: Run tests and verify RED**

Run: `python scripts/test_unihan_residual_regression.py; python scripts/test_unihan_ink_diffusion.py`

Expected: FAIL because the report runner and new evidence layers do not exist.

- [ ] **Step 3: Implement named regression runner**

Implement a CLI accepting `--pdf`, `--bbox-cache`, `--candidates`, `--annotations`, `--learned-model`, `--output`, and `--html-dir`. Require and report:

- U+66DA J and U+6726 J component 228;
- U+6418 G component 220 and component 439;
- U+64CE T component 486.

Missing local cases must be reported as missing, not silently counted as passing.

- [ ] **Step 4: Extend review HTML**

Add toggles for residual-before/residual-after, reusable contact zones, selected half-edges, forbidden exits, and visible owner. Each stroke panel must show source-order provenance, pen-down candidate reasons, hard rejections, score terms, and remaining later-stroke feasibility.

- [ ] **Step 5: Run real local regressions and render artifacts**

Run the regression CLI with `.local/unicode/U4E00.pdf`, current bbox/candidate caches, all available human annotations, and a leave-one-out learned model. Write only to `.local/ink-residual-review/` and `artifacts/unihan-import/`.

Expected:

- grass horizontal owns no vertical half-edge outside the contact disk;
- hand vertical-hook owns no later rising-stroke branch;
- old-head first route starts on the upper short horizontal;
- U+64CE T retains its verified source/component order;
- every unresolved contradiction is `needs-review`.

- [ ] **Step 6: Inspect the generated HTML visually**

Open the index and check each named case at pen-down, junction, pen-up, and residual-after frames. Confirm the native PDF outline remains visible and the colour at a reused contact changes to the later stroke.

- [ ] **Step 7: Run complete verification**

Run:

```powershell
python -m unittest discover -s scripts -p "test_unihan_*.py"
bunx tsc --noEmit
git diff --check
git status --short
```

Expected: all Python tests PASS, TypeScript reports no errors, diff check is clean, and no `.local`, PDF, ZIP, or artifact files are staged.

- [ ] **Step 8: Commit**

Run: `git add scripts/unihan-residual-regression.py scripts/test_unihan_residual_regression.py scripts/unihan-ink-diffusion.py scripts/test_unihan_ink_diffusion.py docs/unihan-directed-ink-diffusion.md && git commit -m "feat: verify residual ink decoder on PDF glyphs"`
