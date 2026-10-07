# Immutable, Composable Ink Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the historically best ink-decoder behaviours as immutable experts, run a truth-isolated twelve-case oracle matrix, and establish versioned consultation/fusion contracts without changing any historical expert or applying data to `/admin`.

**Architecture:** Add a small Python orchestration package beside the existing experimental scripts. Tracked JSON registries identify exact Git commits, CLI capabilities, immutable configuration, lineage, and the twelve benchmark cases. Detached historical worktrees execute unchanged scripts in subprocesses. Every cell runs inference without annotations first, persists a content-addressed prediction, then runs post-prediction evaluation separately and verifies the prediction fingerprint. A normalized matrix, oracle report, and capability report are generated under ignored `.local/`; consultation schemas are implemented and guarded but not invoked by the first milestone.

**Tech Stack:** Python 3 standard library, `unittest`, Git worktrees, JSON/HTML reports, existing OpenCV/NumPy/SciPy/PDF dependencies inside each historical expert, existing Bun repository checks.

**Spec:** `docs/superpowers/specs/2026-10-07-immutable-composable-ink-experts-design.md`

## Global Constraints

- Never patch, format, or otherwise mutate a historical expert worktree.
- Never call `createGlyph`, `updateCharacter`, `/admin` write endpoints, or any remote data API.
- Do not add `--annotations` to an inference command. Human truth is legal only in a separate evaluation pass.
- Do not choose a production result from human metrics. The oracle is benchmark-only and every oracle artifact states `usesHumanTruth: true`.
- Do not commit PDFs, bbox caches, candidate snapshots, learned-rule files, annotations, worktrees, run products, or generated reports.
- Resolve annotations by basename plus explicit runtime roots; never commit `D:\C2D\...` or another machine-specific absolute path.
- Preserve raw historical HTML, JSON, stdout, and stderr byte-for-byte. Normalization writes adjacent new files rather than rewriting them.
- A timeout, missing dependency, unsupported option, malformed audit, or absent output becomes a visible failure cell; it must not disappear from the matrix.
- First-milestone execution is serial and deterministic. Add no GPU backend, Julia/Rust rewrite, runtime expert consultation, `/admin` UI, or full-Unicode batch.
- Historical `4140` learned-rule output may be registered as a ninth evidence expert only if its exact rule artifact and commit can be hashed; otherwise report the U+66DA-N historical peak as unreproducible with a specific missing-input reason.

## Review Focus

- A worktree named for an expert must resolve to that manifest's commit and content identity, not merely an existing directory.
- CLI flags differ at `0058190`, `dad0107`, and `97b40d8`; adapters must never send a newer flag to an older script.
- The same prediction fingerprint must result with and without evaluation truth. A mismatch is a truth-isolation failure, not an ordinary low score.
- Historical audits differ in completeness and metric fields. The normalizer must preserve unknown values as `null`, never manufacture success.
- Oracle ordering must prefer a complete valid prediction before metric values. An incomplete high-IoU prefix cannot beat a complete result.
- Cached cells are reusable only when expert identity, case inputs, command, runtime identity, and phase all match.
- Expert capability tags from manifests are hypotheses; measured capability labels come only from completed matrix evidence.

---

### Task 1: Versioned Domain Models and Canonical Hashing

**Files:**
- Create: `scripts/unihan_experts/__init__.py`
- Create: `scripts/unihan_experts/models.py`
- Create: `scripts/unihan_experts/hashing.py`
- Create: `scripts/test_unihan_expert_models.py`

**Interfaces:**

```python
def canonical_json_bytes(value: object) -> bytes: ...
def sha256_file(path: Path) -> str: ...
def sha256_json(value: object) -> str: ...

@dataclass(frozen=True)
class ExpertManifest:
    expert_id: str
    schema_version: int
    commit: str
    adapter: str
    decoder: str
    config: Mapping[str, object]
    rule_artifacts: tuple[ArtifactRef, ...]
    parents: tuple[str, ...]
    declared_capabilities: tuple[str, ...]
    resource_class: str

@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    unicode: str
    source: str
    glyph_id: int
    annotation_basename: str

@dataclass(frozen=True)
class PredictionRecord: ...

@dataclass(frozen=True)
class EvaluationRecord: ...
```

- [ ] **Step 1: Write failing canonicalization and validation tests**

Add tests:

- `test_canonical_json_hash_is_key_order_independent`
- `test_changing_config_changes_expert_content_identity`
- `test_changing_rule_hash_changes_expert_content_identity`
- `test_prediction_record_rejects_truth_during_inference`
- `test_evaluation_record_requires_prediction_fingerprint`
- `test_case_manifest_accepts_basename_but_rejects_absolute_annotation_path`

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python scripts/test_unihan_expert_models.py`

Expected: FAIL because `unihan_experts.models` and `unihan_experts.hashing` do not exist.

- [ ] **Step 3: Implement immutable records and canonical hashing**

Use sorted-key, separator-stable, UTF-8 JSON. Reject floats that are NaN or infinite. Expert content identity must hash the manifest's semantic fields plus resolved commit, source-tree hash, interpreter identity, and rule/config artifact hashes; it must exclude run IDs, paths to `.local`, timestamps, and human metrics.

Keep `PredictionRecord` and `EvaluationRecord` separate. `PredictionRecord` must contain `usesHumanTruth=False` and `truthFirstReadPhase=None`; do not represent these as optional conventions.

- [ ] **Step 4: Run tests and compile the package**

Run: `python scripts/test_unihan_expert_models.py; python -m compileall -q scripts/unihan_experts`

Expected: PASS.

- [ ] **Step 5: Commit the model layer**

Run: `git add scripts/unihan_experts scripts/test_unihan_expert_models.py; git commit -m "feat: define immutable ink expert records"`

---

### Task 2: Expert, Case, and Consultation Registries

**Files:**
- Create: `scripts/unihan-experts.json`
- Create: `scripts/unihan-expert-cases.json`
- Create: `scripts/unihan_experts/registry.py`
- Create: `scripts/unihan_experts/protocol.py`
- Create: `scripts/test_unihan_expert_registry.py`
- Create: `scripts/test_unihan_expert_protocol.py`

**Initial experts:**

- `classic-15c` at `15c2b04`, decoder `legacy`
- `junction-23b` at `23b9179`, decoder `hybrid`
- `scan-front-7fc` at `7fc2c16`, decoder `hybrid`
- `grass-005` at `0058190`, decoder `hybrid`
- `recursive-ids-8c` at `8c585f2`, decoder `hybrid`
- `complete-route-2272` at `2272a6d`, decoder `hybrid`
- `contact-dd` at `dd1dd05`, decoder `hybrid`
- `reserved-dad` at `dad0107`, decoder `hybrid`
- Optional `learned-414` at the verified historical commit, enabled only when its original `rules.json` hash is available.

**Twelve cases:** `u6418-g-17973`, `u6424-j-900005`, `u6461-g-18049`, `u6485-j-18088`, `u64ce-t-900000`, `u64cf-h-900001`, `u64ec-t-900021`, `u65e8-t-127685`, `u66da-j-18633`, `u66da-n-57149`, `u6726-j-18691`, and `u808e-t-900013`.

- [ ] **Step 1: Write failing registry tests**

Cover:

- duplicate expert/case IDs;
- unresolved commits;
- parent cycles and missing parents;
- unsupported manifest schema and adapter protocol;
- invalid decoder/adapter combinations;
- a clone/fusion preserving parent identities;
- all twelve cases present exactly once;
- annotation basenames resolved only from explicit roots;
- source tree/config/rule hash mismatch rejected in strict mode.

- [ ] **Step 2: Write failing protocol tests**

Define and test JSON round trips for:

```python
@dataclass(frozen=True)
class ProposalRequest:
    protocol_version: int
    caller_expert_id: str
    run_id: str
    input_hashes: Mapping[str, str]
    ids_subtree: Mapping[str, object] | None
    stroke_interval: tuple[int, int] | None
    snapshot_hashes: Mapping[str, str]
    required_capabilities: tuple[str, ...]
    budgets: ConsultationBudgets
    visited_experts: tuple[str, ...]

@dataclass(frozen=True)
class ProposalBundle:
    protocol_version: int
    specialist_expert_id: str
    request_hash: str
    proposals: tuple[Mapping[str, object], ...]
    evidence: Mapping[str, object]
    assumptions: tuple[str, ...]
    resource_cost: Mapping[str, object]
    provenance: Mapping[str, object]
```

Tests must reject cycles, repeated expert/input pairs, excessive depth, expired wall/CPU/memory/candidate budgets, and requests that specify neither an IDS subtree nor a stroke interval. These guards are library-only in milestone one; do not invoke experts through them yet.

- [ ] **Step 3: Run both tests and verify RED**

Run: `python scripts/test_unihan_expert_registry.py; python scripts/test_unihan_expert_protocol.py`

Expected: FAIL because registries and parsers do not exist.

- [ ] **Step 4: Implement registry loaders, validators, and manifests**

Use repository-relative aliases for shared artifacts:

```json
{
  "pdf": ".local/unicode/U4E00.pdf",
  "bbox": ".local/unicode/U4E00-bbox.html",
  "candidates": ".local/reviewed-glyph-svg-candidates.json"
}
```

The committed case registry stores only `annotationBasename`. CLI runtime options supply one or more annotation roots, with `.local/annotations` and an explicit user-provided directory as normal choices. Validate actual file hashes at run creation and store those hashes in the run manifest; do not hard-code hashes that will silently go stale.

- [ ] **Step 5: Run registry/protocol tests**

Run: `python scripts/test_unihan_expert_registry.py; python scripts/test_unihan_expert_protocol.py`

Expected: PASS, including exact count assertions for enabled experts and twelve cases.

- [ ] **Step 6: Commit registries and contracts**

Run: `git add scripts/unihan-experts.json scripts/unihan-expert-cases.json scripts/unihan_experts/registry.py scripts/unihan_experts/protocol.py scripts/test_unihan_expert_registry.py scripts/test_unihan_expert_protocol.py; git commit -m "feat: register historical ink experts and cases"`

---

### Task 3: Non-Destructive Historical Worktree Provisioning

**Files:**
- Create: `scripts/unihan_experts/process.py`
- Create: `scripts/unihan_experts/worktrees.py`
- Create: `scripts/test_unihan_expert_worktrees.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    wall_seconds: float
    cpu_seconds: float | None
    peak_memory_bytes: int | None

def resolve_commit(repo: Path, revision: str) -> str: ...
def source_tree_hash(repo: Path, commit: str) -> str: ...
def ensure_expert_worktree(repo: Path, root: Path, manifest: ExpertManifest) -> Path: ...
```

- [ ] **Step 1: Write failing tests using a temporary Git repository**

Tests:

- provision a detached worktree at the exact commit;
- a second call is idempotent;
- an existing path at the wrong commit is rejected rather than reset;
- an unrelated worktree is never removed;
- a dirty historical worktree is rejected;
- abbreviated commits resolve to the full SHA stored in identity;
- source-tree hash changes when tracked source changes;
- timeout and non-zero process results retain stdout/stderr;
- resource fields are present, with `null` plus a reason when the platform cannot measure one.

- [ ] **Step 2: Run test and verify RED**

Run: `python scripts/test_unihan_expert_worktrees.py`

Expected: FAIL because the worktree/process layer does not exist.

- [ ] **Step 3: Implement safe provisioning and resource capture**

Provision under `.local/expert-worktrees/<expert-id>/` with `git worktree add --detach <path> <full-sha>`. Query `git worktree list --porcelain` before adding. Never use `git worktree remove`, `git clean`, `git reset`, or checkout to repair a mismatch; return an actionable failure.

Implement subprocess execution without `shell=True`. Sample peak RSS when available (optional `psutil`, Windows process APIs, or platform resource APIs); absence must not fail a cell. Record interpreter executable/version and selected dependency versions using a bounded probe inside each worktree.

- [ ] **Step 4: Run focused tests**

Run: `python scripts/test_unihan_expert_worktrees.py`

Expected: PASS.

- [ ] **Step 5: Commit worktree safety**

Run: `git add scripts/unihan_experts/process.py scripts/unihan_experts/worktrees.py scripts/test_unihan_expert_worktrees.py; git commit -m "feat: provision immutable expert worktrees"`

---

### Task 4: Per-Commit CLI Adapters and Truth-Free Commands

**Files:**
- Create: `scripts/unihan_experts/adapters.py`
- Create: `scripts/test_unihan_expert_adapters.py`

**Adapter families:**

- `classic-v1`: common inputs, `--decoder`, learned rules, stroke-order catalog, beam width, output, audit output.
- `centroid-v2`: `classic-v1` plus `--leaf-centroid-weight` for `0058190` through `dd1dd05`.
- `reservation-v3`: `centroid-v2` plus optional `--disable-completed-stroke-reservation` for `dad0107`.
- `head-v4`: current HEAD-only ensemble/budget/partial options; not used to impersonate a historical expert.

- [ ] **Step 1: Write failing command-construction tests**

For every registered expert, assert the complete argv. Specifically assert:

- no inference argv contains `--annotations` or the annotation path;
- pre-`0058190` commands omit `--leaf-centroid-weight`;
- pre-`dad0107` commands omit reservation flags;
- pre-`97b40d8` commands never request decoder `ensemble`;
- all shared inputs and outputs are absolute paths;
- rule/config flags exactly match the manifest and their hashes are captured;
- unknown adapter names and unsupported flags fail before spawning a process.

- [ ] **Step 2: Run test and verify RED**

Run: `python scripts/test_unihan_expert_adapters.py`

Expected: FAIL because `adapters.py` does not exist.

- [ ] **Step 3: Implement explicit adapters**

Do not discover historical interfaces by running `--help` on every cell. The adapter named by the reviewed manifest owns a fixed allowlist. Provide separate builders:

```python
def build_inference_command(ctx: RunContext) -> tuple[str, ...]: ...
def build_evaluation_command(ctx: RunContext, annotation: Path) -> tuple[str, ...]: ...
```

The evaluation builder may add only `--annotations` and phase-specific output paths to the same semantic command. Store a canonical `semanticCommandHash` that ignores output filenames and the evaluation-only annotation flag; this allows phase predictions to be compared.

- [ ] **Step 4: Run adapter tests**

Run: `python scripts/test_unihan_expert_adapters.py`

Expected: PASS for every manifest/adapter pair.

- [ ] **Step 5: Commit adapters**

Run: `git add scripts/unihan_experts/adapters.py scripts/test_unihan_expert_adapters.py; git commit -m "feat: adapt historical ink expert CLIs"`

---

### Task 5: Historical Audit Normalization and Prediction Fingerprints

**Files:**
- Create: `scripts/unihan_experts/normalize.py`
- Create: `scripts/test_unihan_expert_normalize.py`
- Create: `scripts/fixtures/expert-audits/classic-success.json`
- Create: `scripts/fixtures/expert-audits/centroid-success.json`
- Create: `scripts/fixtures/expert-audits/reservation-success.json`
- Create: `scripts/fixtures/expert-audits/partial.json`
- Create: `scripts/fixtures/expert-audits/malformed.json`

Fixtures must be minimal, hand-reduced representations of existing schemas; do not commit generated PDF/source content or whole large audits.

- [ ] **Step 1: Write failing normalization tests**

Cover:

- metric paths under historical `evaluation` objects;
- absent metrics normalized to `null` rather than zero;
- expected/predicted stroke counts and completeness derived conservatively;
- review reasons and route counts preserved;
- malformed or missing audits becoming typed failures;
- native evidence retained under a namespaced field;
- raw artifact hashes computed without rewriting files;
- a fingerprint that excludes evaluation, paths, timestamps, and resource data;
- inference/evaluation fingerprints matching for the same prediction;
- any changed route/step/component assignment changing the fingerprint.

- [ ] **Step 2: Run test and verify RED**

Run: `python scripts/test_unihan_expert_normalize.py`

Expected: FAIL because normalization code does not exist.

- [ ] **Step 3: Implement conservative normalizers**

Use this normalized status set: `complete`, `needs-review`, `partial`, `timeout`, `unsupported`, `process-error`, `missing-output`, `malformed-audit`, `truth-isolation-failure`.

Compute `predictionFingerprint` from stable prediction-only fields: character identity, expected/predicted stroke counts, ordered selected route summaries, component/leaf assignments, decision model, and native decision steps after removal of evaluation-only fields. Do not include floating resource observations or artifact paths.

- [ ] **Step 4: Run focused tests**

Run: `python scripts/test_unihan_expert_normalize.py`

Expected: PASS.

- [ ] **Step 5: Commit normalization**

Run: `git add scripts/unihan_experts/normalize.py scripts/test_unihan_expert_normalize.py scripts/fixtures/expert-audits; git commit -m "feat: normalize historical expert predictions"`

---

### Task 6: Resumable Two-Phase Matrix Runner

**Files:**
- Create: `scripts/unihan_experts/runner.py`
- Create: `scripts/unihan-expert-benchmark.py`
- Create: `scripts/test_unihan_expert_runner.py`

**CLI:**

```text
python scripts/unihan-expert-benchmark.py validate ...
python scripts/unihan-expert-benchmark.py provision ...
python scripts/unihan-expert-benchmark.py run --annotation-root <dir> [--expert ID] [--case ID] [--resume]
python scripts/unihan-expert-benchmark.py report --run <run-id>
```

- [ ] **Step 1: Write failing runner tests with fake expert executables**

Tests must prove:

- inference runs first without access to annotation path/environment;
- `prediction.json` is atomically written before evaluation begins;
- evaluation executes in a distinct directory and can read only the selected annotation;
- differing phase fingerprints produce `truth-isolation-failure`;
- timeout, crash, missing HTML, missing audit, and malformed audit yield explicit cells;
- every requested expert/case pair appears exactly once;
- `--resume` reuses only a matching immutable cache key;
- a changed interpreter/config/input hash invalidates the cache;
- output ordering remains expert-ID/case-ID deterministic;
- raw files remain byte-identical after normalization;
- interruption after one phase resumes without rerunning a valid completed phase.

- [ ] **Step 2: Run test and verify RED**

Run: `python scripts/test_unihan_expert_runner.py`

Expected: FAIL because the runner and CLI do not exist.

- [ ] **Step 3: Implement cell layout and atomic state transitions**

Write each cell under:

```text
.local/ink-experts/runs/<run-id>/<expert-id>/<case-id>/
  manifest.json
  inference/{stdout.log,stderr.log,raw-audit.json,review.html,prediction.json}
  evaluation/{stdout.log,stderr.log,raw-audit.json,review.html,evaluation.json}
  normalized.json
```

Use temporary sibling files plus `Path.replace()` for JSON state. The inference pass must finish and its fingerprint be stored before resolving the annotation path for evaluation. The evaluation pass repeats the historical command with the annotation, then compares fingerprints. On mismatch retain both raw outputs and fail the cell.

Use one worker only. Apply per-expert wall-time limits and retain resource observations. Print progress after every phase so a long serial run remains auditable.

- [ ] **Step 4: Run runner tests and a one-cell smoke test**

Run:

```powershell
python scripts/test_unihan_expert_runner.py
python scripts/unihan-expert-benchmark.py validate --annotation-root "D:\C2D\Downloads" --annotation-root ".local"
python scripts/unihan-expert-benchmark.py provision --expert classic-15c
python scripts/unihan-expert-benchmark.py run --expert classic-15c --case u65e8-t-127685 --annotation-root "D:\C2D\Downloads" --resume
```

Expected: tests PASS; validation identifies all twelve annotations; the smoke cell writes both phases, identical prediction fingerprints, and no tracked files.

- [ ] **Step 5: Commit the runner**

Run: `git add scripts/unihan_experts/runner.py scripts/unihan-expert-benchmark.py scripts/test_unihan_expert_runner.py; git commit -m "feat: run truth-isolated expert benchmarks"`

---

### Task 7: Oracle, Capability, and Resource Reports

**Files:**
- Create: `scripts/unihan_experts/oracle.py`
- Create: `scripts/unihan_experts/report.py`
- Create: `scripts/test_unihan_expert_oracle.py`
- Create: `scripts/test_unihan_expert_report.py`

- [ ] **Step 1: Write failing oracle tests**

Test the exact ordering contract:

1. truth-isolation failures and invalid cells are ineligible;
2. complete beats partial;
3. fewer hard failure/review reasons wins;
4. higher stroke macro IoU wins;
5. higher component macro IoU wins;
6. lower directed-sequence/start/end errors win;
7. deterministic expert ID breaks a true tie.

Also assert that oracle outputs always state `usesHumanTruth: true`, while a truth-free production selector module is neither imported nor implemented in this milestone.

- [ ] **Step 2: Write failing report tests**

Using a synthetic matrix, assert:

- every cell, including failures, appears in JSON and HTML;
- the winner and every runner-up are visible;
- declared versus measured capabilities are visually separate;
- resource totals and per-phase wall/CPU/RSS observations appear;
- report links target preserved raw HTML/audit files;
- assisted/unassisted columns exist as reserved schema fields but are marked `not-run`;
- output order and report hashes are deterministic after excluding generation time.

- [ ] **Step 3: Run tests and verify RED**

Run: `python scripts/test_unihan_expert_oracle.py; python scripts/test_unihan_expert_report.py`

Expected: FAIL because oracle/report modules do not exist.

- [ ] **Step 4: Implement benchmark-only selection and reports**

Generate:

```text
.local/ink-experts/reports/<run-id>/matrix.json
.local/ink-experts/reports/<run-id>/oracle.json
.local/ink-experts/reports/<run-id>/capabilities.json
.local/ink-experts/reports/<run-id>/resources.json
.local/ink-experts/reports/<run-id>/index.html
```

Derive measured capabilities from per-case wins, near-wins, failure patterns, component families, and cost; do not copy manifest tags into the measured field. Add the known historical peak table as a comparison fixture with tolerances and an explanation field for missing rule artifacts, dependency drift, timeout, or nondeterminism.

- [ ] **Step 5: Run report tests**

Run: `python scripts/test_unihan_expert_oracle.py; python scripts/test_unihan_expert_report.py`

Expected: PASS.

- [ ] **Step 6: Commit reporting**

Run: `git add scripts/unihan_experts/oracle.py scripts/unihan_experts/report.py scripts/test_unihan_expert_oracle.py scripts/test_unihan_expert_report.py; git commit -m "feat: report historical expert oracle matrix"`

---

### Task 8: Execute the Historical Matrix and Reconcile Peaks

**Files:**
- Modify only if defects are found: first-milestone modules and their tests
- Create locally only: `.local/ink-experts/runs/<run-id>/...`
- Create locally only: `.local/ink-experts/reports/<run-id>/...`

- [ ] **Step 1: Validate all immutable inputs before expensive work**

Run:

```powershell
python scripts/unihan-expert-benchmark.py validate --annotation-root "D:\C2D\Downloads" --annotation-root ".local"
python scripts/unihan-expert-benchmark.py provision
git worktree list --porcelain
```

Expected: each enabled expert resolves to its full SHA; all twelve annotations and shared artifacts resolve and hash; no worktree is dirty or at the wrong commit.

- [ ] **Step 2: Run a three-cell canary**

Run known historical strengths first:

```powershell
python scripts/unihan-expert-benchmark.py run --expert classic-15c --case u808e-t-900013 --annotation-root "D:\C2D\Downloads" --resume
python scripts/unihan-expert-benchmark.py run --expert grass-005 --case u64ec-t-900021 --annotation-root "D:\C2D\Downloads" --resume
python scripts/unihan-expert-benchmark.py run --expert recursive-ids-8c --case u64cf-h-900001 --annotation-root ".local" --resume
```

Expected: all six inference/evaluation phases finish, prediction fingerprints match within each cell, and raw HTML is inspectable.

- [ ] **Step 3: Run the serial expert/case matrix with resume enabled**

Run:

```powershell
python scripts/unihan-expert-benchmark.py run --annotation-root "D:\C2D\Downloads" --annotation-root ".local" --resume
python scripts/unihan-expert-benchmark.py report --run latest
```

Do not hide long or failed cells. Continue with `--resume` after interruption. If one expert systematically exceeds its declared budget, retain its failure rows and rerun only after a manifest/resource-class change creates a new expert identity.

- [ ] **Step 4: Reconcile every known historical peak**

For each of the twelve cases record:

- historical best commit/artifact and metric;
- reproduced expert and metric;
- deterministic tolerance delta;
- input/rule/interpreter hashes;
- exact reason when not reproduced;
- raw HTML link for visual review.

U+66DA-N must explicitly say whether `learned-414` was reproducible. Do not substitute a later low-scoring complete run for the earlier peak without explanation.

- [ ] **Step 5: Verify deterministic warm reruns**

Select at least one expert/case from each adapter family and rerun with warm worktrees/cache. Assert identical prediction fingerprints and normalized semantic hashes. Then run the complete report generation twice and compare deterministic report hashes.

- [ ] **Step 6: Fix only reproducibility framework defects test-first**

If a framework bug is found, first add the smallest failing unit/integration test, then patch the framework. Never patch a historical expert to make a peak reappear.

- [ ] **Step 7: Commit any tested framework corrections**

Run: `git status --short; git diff --check`

If tracked framework corrections exist: `git add <exact-framework-files>; git commit -m "fix: reproduce historical expert matrix"`

Do not add `.local/` products.

---

### Task 9: Full Verification and Maintainer Handoff

**Files:**
- Create: `docs/unihan-ink-expert-benchmark.md`
- Modify if required: `README.md`

- [ ] **Step 1: Document reproducible local commands and safety boundary**

Document registry validation, worktree provisioning, one-cell smoke run, resumable matrix, report generation, artifact layout, truth isolation, and explicit non-goals. Explain that the oracle is diagnostic only and cannot write upstream data.

- [ ] **Step 2: Run all Python tests**

Run: `python -m unittest discover -s scripts -p 'test_unihan_*.py'`

Expected: all existing 228 tests plus new expert tests PASS.

- [ ] **Step 3: Run repository checks**

Inspect `package.json` for the canonical scripts, then run the existing check/typecheck/build commands without inventing replacements. At minimum run the same Bun checks used by the branch before this milestone.

- [ ] **Step 4: Verify isolation and tracked-file hygiene**

Run:

```powershell
git status --short
git diff --check
git ls-files ".local" "artifacts" "*.pdf" "*annotations.json"
git worktree list --porcelain
```

Expected: no generated expert worktree, PDF, annotation, run, report, or artifact is tracked; historical worktrees remain detached and clean.

- [ ] **Step 5: Perform report-level visual review**

Open the report index and at least the twelve oracle-winning raw HTML files. Confirm links work, winners correspond to their preserved raw outputs, failures remain visible, and no report claims production correctness from oracle truth.

- [ ] **Step 6: Commit documentation**

Run: `git add docs/unihan-ink-expert-benchmark.md README.md; git commit -m "docs: explain immutable ink expert benchmark"`

- [ ] **Step 7: Final handoff contents**

Report:

- enabled expert IDs, commits, hashes, parents, and adapter families;
- complete matrix counts by status;
- per-case oracle winners and metrics;
- reproduced versus unreproduced historical peaks with reasons;
- measured capability boundaries and resource profiles;
- truth-isolation and deterministic-rerun results;
- exact report/index paths;
- test/check commands and outcomes;
- `git diff --stat` and `git status --short`;
- a recommendation for milestone two (specialist broker, scheduler parallelism, or a fused child), without implementing it.

## Milestone Exit Gate

Do not start runtime consultation, fusion promotion, parallel execution, GPU kernels, language rewrites, `/admin` integration, or full-Unicode inference until all of these are true:

- every enabled expert/case pair has a normalized success or explicit failure;
- prediction fingerprints prove truth isolation;
- historical peaks are reproduced or specifically explained;
- warm reruns are deterministic;
- generated files remain ignored;
- the user has visually reviewed the oracle report and approved the next milestone.
