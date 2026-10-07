# Immutable ink expert benchmark

This document describes the local, benchmark-only runtime for comparing
historical ink decoders. It does not select a production result, write Hanzi
Chai data, call `/admin`, or apply any proposal. Human annotations are opened
only after a truth-free prediction has been atomically persisted.

## Reproduce locally

The PDF, bounding-box cache, candidate snapshot, annotations, detached
worktrees, runs, and reports are intentionally ignored under `.local/` (with
annotations also permitted in an explicit external root).

```powershell
python scripts/unihan-expert-benchmark.py validate `
  --annotation-root "D:\C2D\Downloads" `
  --annotation-root ".local"
python scripts/unihan-expert-benchmark.py provision

# One expert or one cell is useful as a canary.
python scripts/unihan-expert-benchmark.py run `
  --expert classic-15c `
  --case u808e-t-900013 `
  --annotation-root "D:\C2D\Downloads" `
  --run-id historical-oracle-v1 `
  --resume

# Run experts serially when memory is constrained. Reusing the same run ID
# accumulates cells, while --resume reuses matching inference/evaluation phases.
python scripts/unihan-expert-benchmark.py run `
  --expert learned-414 `
  --annotation-root "D:\C2D\Downloads" `
  --annotation-root ".local" `
  --run-id historical-oracle-v1 `
  --resume

python scripts/unihan-expert-benchmark.py report --run historical-oracle-v1
```

Artifacts are written to:

```text
.local/ink-experts/runs/historical-oracle-v1/
.local/ink-experts/reports/historical-oracle-v1/index.html
```

Every cell records immutable input hashes, interpreter/dependency identity,
the semantic command hash, raw HTML/audit paths, resource observations, and a
prediction fingerprint. A cell is truth-isolated only when inference reports
no truth use, evaluation is the first phase to read the annotation, and both
phases produce the same prediction fingerprint.

## Registered experts

All imported historical experts are roots in the local registry; none edits or
replaces another expert.

| Expert | Commit | Adapter | Declared specialty |
|---|---:|---|---|
| `learned-414` | `4140eea` | `learned-v0` | learned route prior |
| `classic-15c` | `15c2b04` | `classic-v1` | complete classic strokes |
| `junction-23b` | `23b9179` | `classic-v1` | junction topology |
| `scan-front-7fc` | `7fc2c16` | `classic-v1` | closed boxes and scan fronts |
| `grass-005` | `0058190` | `centroid-v2` | ambiguous grass heads |
| `recursive-ids-8c` | `8c585f2` | `centroid-v2` | recursive IDS and sibling order |
| `complete-route-2272` | `2272a6d` | `centroid-v2` | complete directed routes |
| `contact-dd` | `dd1dd05` | `centroid-v2` | contact and retrace constraints |
| `reserved-dad` | `dad0107` | `reservation-v3` | completed-stroke reservation |

`learned-414` additionally pins
`.local/ink-rule-training/rules.json` to SHA-256
`20b22765fd13d114a266b378ae616de7172e4c9c514fe03ba19248129340beab`.

## `historical-oracle-v1` result

The completed matrix contains 108 cells (nine experts by twelve cases): 18
reported `complete`, 80 reported `needs-review`, and 10 explicit
`process-error` cells. All 98 successful cells matched their inference and
evaluation prediction fingerprints; there were no truth-isolation failures.

`contact-dd` and `reserved-dad` each fail the same five cases because their
historical hard constraints empty the search space: U+6461-G and U+64CF-H at
stroke 12, U+64EC-T at stroke 13, and U+65E8-T/U+808E-T at stroke 6. These are
preserved capability failures (`no globally feasible route hypothesis`), not
hidden or retried with changed parameters.

The oracle is diagnostic and uses human truth. It first excludes invalid or
incomplete cells, then counts only hard structural review reasons
(`candidate-stroke-grammar-mismatch`, `future-feasibility-fallback`, and
`hard-geometry-fallback`) before ranking post-truth metrics. Soft diagnostics
remain visible but cannot beat a much better truth metric merely by being
fewer.

| Case | Oracle expert | Stroke IoU | Component IoU | Status |
|---|---|---:|---:|---|
| U+6418-G | `junction-23b` | 0.919986 | 0.985764 | complete |
| U+6424-J | `complete-route-2272` | 0.894894 | 0.973251 | needs-review |
| U+6461-G | `learned-414` | 0.715996 | 0.905755 | needs-review |
| U+6485-J | `recursive-ids-8c` | 0.648445 | 0.735421 | needs-review |
| U+64CE-T | `complete-route-2272` | 0.930601 | 0.978029 | needs-review |
| U+64CF-H | `recursive-ids-8c` | 0.802997 | 0.903001 | needs-review |
| U+64EC-T | `recursive-ids-8c` | 0.889462 | 0.970801 | needs-review |
| U+65E8-T | `classic-15c` | 0.934904 | 1.000000 | complete |
| U+66DA-J | `recursive-ids-8c` | 0.894093 | 0.945864 | needs-review |
| U+66DA-N | `learned-414` | 0.563851 | 0.586278 | needs-review |
| U+6726-J | `classic-15c` | 0.643240 | 0.975651 | needs-review |
| U+808E-T | `classic-15c` | 0.953967 | 0.996250 | complete |

The matrix confirms that no single historical model dominates. For example,
`complete-route-2272` reaches 0.927473 on U+6726-J but falls to 0.349417 on
U+6485-J, where `recursive-ids-8c` reaches 0.648445. This is evidence for a
future truth-free router or bounded specialist consultation, not permission to
copy oracle winners into production.

## Historical peak reconciliation

The classic U+65E8-T/U+808E-T peaks, scan-front U+6424-J peak, and grass-head
U+64EC-T peak were reproduced at approximately 0.935/0.954, 0.923, and 0.923
stroke IoU respectively. Two previously noted ad-hoc peaks were not reproduced:

- U+64CF-H was previously observed around 0.885; the immutable
  `recursive-ids-8c` run reaches 0.802997.
- U+66DA-N was previously observed around 0.706; `learned-414` reaches
  0.563851 even though commit and rule SHA-256 match.

The current shared hashes are PDF
`9c2527a34623870413a348778a7804475c15e92960017dd3693ba18fd121aaf9`,
bounding boxes
`6f31505f35d8fbcf50fb3d2876c0c6049f5bade504f78689ea64c82690649b01`,
and candidates
`965244ef5843ea3abd8893ca7150995c4fb74175ea52c7d940d50c552f9e93d6`.
The run used CPython 3.12.4, NumPy 2.4.2, OpenCV 4.13.0.92, SciPy 1.17.1,
and psutil 5.9.0. The older ad-hoc runs did not preserve an equivalent complete
manifest, so input/dependency snapshot drift is the specific unresolved cause;
historical code was not patched to force the peaks to reappear.

## Cost and deterministic resume

| Expert | Total wall seconds | Peak RSS MiB |
|---|---:|---:|
| `classic-15c` | 1471 | 1369 |
| `complete-route-2272` | 1960 | 1212 |
| `contact-dd` | 1716 | 732 |
| `grass-005` | 2324 | 2183 |
| `junction-23b` | 655 | 961 |
| `learned-414` | 1280 | 1344 |
| `recursive-ids-8c` | 2149 | 1002 |
| `reserved-dad` | 1864 | 749 |
| `scan-front-7fc` | 2619 | 2258 |

A warm 12-cell `learned-414 --resume` completed in 2.22 seconds without
rerunning inference. The accumulated matrix SHA-256 remained
`e4d3a0a9c1b2c25b6636553f9922663894b24260f6ac56152e11805602a5d00d`,
and the regenerated oracle SHA-256 remained
`cb3475757cee2866e047bb4d0f2dd873e70d3b619a71505f1140068ff902eeb7`.

The report index and all twelve oracle-winning evaluation HTML files were
rendered in a real headless browser. Candidate components, PDF ink, metrics,
and evidence panels were visible; failure rows remained visible in the index;
and the report banner states that it uses human truth and must not choose a
production result.

## Next milestone

Keep every historical expert immutable. The next useful experiment is a
truth-free router that runs a cheap expert first and requests a bounded
IDS-subtree proposal from a specialist only when native evidence is weak. Any
successful composition should be materialized as a new child expert; it must
not mutate either parent. Do not begin full-Unicode inference or `/admin`
application until the router is evaluated on held-out annotations.
