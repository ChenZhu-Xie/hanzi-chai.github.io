# Unihan stroke-sequence decoder notes

This document records the working model behind the PDF glyph decomposition
experiment. It is deliberately narrower than a general handwriting recognizer.

## Human procedure to preserve

1. Read strokes in standard writing order, from pen-down to pen-up.
2. Accumulate the smallest possible prefix that completes one leaf component.
3. Match that completed component to an exact repository component or one of
   its verified siblings. If none matches, propose a new sibling/component; do
   not silently force an existing ID.
4. Close that component instance and continue with the next stroke.
5. After the last component closes, verify every stroke, component boundary,
   component ID, sibling choice, and the reconstructed whole glyph.

The sequence unit is `(componentId, occurrence)`, not just `componentId`. Two
instances of the same component may occur in one glyph. The U+6424 J-source
truth is the concrete example: `117#0 -> 1#0 -> 117#1` is valid and each
instance is contiguous.

## Evidence from the first nine human truths

All nine exported truths are contiguous when component occurrence is retained.
Eight are also contiguous by component ID. U+6424 is the only apparent ID-level
exception, and it disappears when the two occurrences of component 117 are
distinguished. This supports a sequential decoder as a strong prior, while the
candidate reference tree remains the grammar for repeated/enclosing layouts.

## Current deterministic pipeline

- Repository candidates provide stroke order, directed stroke arcs, leaf IDs,
  sibling-family metadata, hierarchy, and occurrence numbers.
- Verified annotations may transfer a leaf template only when component ID and
  stroke count match exactly.
- A beam search chooses component-template combinations using whole-glyph
  symmetric reconstruction error. The held-out target annotation is never read
  while choosing.
- Centerlines can be fit coherently, constrained to the PDF skeleton graph, or
  constrained to a candidate corridor. No single snapping mode is universally
  best; they must remain competing hypotheses.
- The PDF ink partition is evaluated only after the choice is frozen.

## Failed or limited experiments

- Local per-component template selection can produce a globally inconsistent
  glyph. U+64CE T improved substantially only after joint whole-glyph search.
- An unrestricted skeleton route may leave a stroke at an intersection and
  return through a neighbouring component.
- A fixed-width hard corridor repairs that failure for some glyphs (U+671E T)
  but rejects legitimate large deformation in others (U+66DA N). It is a
  hypothesis, not a universal rule.
- A small score margin is evidence of ambiguity, not a calibrated probability.
- Adding skeleton coverage/collision terms to the old whole-glyph score improved
  the 60-case training split from 32 to 34 correct, but reduced the frozen
  42-case test split from 24 to 20. These terms remain diagnostics, not weights.
- Candidate-conditioned component closure was circular: each candidate first
  partitioned its own grading region and was then scored inside that region.
  It reached only 46/102 overall and 15/42 on the frozen test split, versus the
  whole-glyph baseline's 56/102 and 24/42. Future closure comparison must use a
  target window frozen jointly across all sibling candidates.
- In strict nine-fold annotation LOOCV, exact-ID verified templates initially
  selected 5/9 glyph candidates. Treating same Unicode across different IRG
  sources as the same absolute layout caused U+64CE G/T to swap. Restricting
  absolute transfer to the same Unicode *and* the same source raised selection
  to 6/9.
- Two exports still contained the sibling label `133` even though their chosen
  repository candidates use `1128`; the editor had corrected this only while
  displaying the files. Reference truth is now normalized by the exported
  candidate ID plus its complete directed stroke sequence before propagation.
  This fixes template coverage without editing the original JSON.
- A candidate containing an exact leaf already verified in another character
  of the same IRG source, when no sibling candidate contains that leaf, is now
  reported as unique same-source evidence. It would correct U+64EC T and
  U+65E8 T in the nine-case corpus, but on 20 previously labelled sources
  outside that annotation corpus it was correct only 15 times. Source
  conventions are not universal (for example, one source may use more than
  one grass-head form), so this signal is never a candidate decision by itself.
- Recursive terminal-stroke topology alone also selected 6/9, but made a
  complementary set of errors: it fixed U+64CE T and U+64EC T while breaking
  U+64CE G and U+64CF H, and still missed U+65E8 T. It is therefore reported
  as an experimental opinion and never overrides the verified-template result.
  A generic numeric topology gate was rejected: the gate found on the frozen
  60-case train split made 7 correct overrides there but only 3/7 on the
  independent 42-case test split. It must not be used for automatic writes.
- U+64CE T has a narrower auditable pattern: its optical/template winner is the
  candidate independently verified for G of the same Unicode, while the T PDF
  terminal-stroke topology supports the other candidate. The reverse G fold
  does not meet this condition. Reporting this as a cross-source divergence
  review recommendation would make the exploratory nine-fold review hints 9/9.
  A broader audit, however, found this cross-source divergence condition right
  on only 20/28 triggers across the 102-case corpus. The strict prediction thus
  remains 6/9; the 9/9 number is displayed only as an exploratory review-hint
  result and never as accuracy or permission to write.
- Applying the source-aware transferred centerlines as final geometry reduced
  mean component/stroke IoU from 0.73/0.55 to 0.70/0.52. Verified templates
  therefore remain candidate-selection evidence by default; applying their
  geometry is an explicit experimental mode until component-local warping is
  independently validated.

## Next algorithmic step

Decode the glyph jointly over `(stroke index, component instance, template,
snap hypothesis)`. A state may advance only by consuming the next directed
stroke. It may close a component only at a candidate-tree leaf boundary, and a
closed occurrence may not be re-entered. The objective should combine:

- missing/extra strokes and stroke-count agreement;
- pen-down/pen-up direction, tangent, turns, crossings, and connectivity;
- component-local reconstruction and whole-glyph reconstruction;
- component boundary/spacing consistency;
- ink ambiguity, fragmentation, and competing-hypothesis margin.

Automatic acceptance remains prohibited when an exact component has no
verified template, a new sibling/component is implicated, sequence grammar is
violated, or the best/runner-up margin is too small.

## Residual decoder and leakage boundary

The residual decoder makes each stroke decision against the ink that remains
after earlier strokes. It converts the skeleton to directed half-edges, treats
each junction cluster as a gate, permits reuse only in a small contact zone,
and permanently consumes the selected road outside that zone. A route that
leaves its allowed half-edge, re-enters a closed component occurrence, reverses
at a gate, or leaks into an unselected branch is rejected before learned costs
are considered.

Learned rules use schema version 2. Their key is
`source | exact leaf ID | leaf stroke ordinal | leaf stroke count | feature`,
so a G observation cannot silently train a T, J, or other source convention.
Each observation retains the selected route half-edges and the alternative
half-edges explicitly rejected at its junctions. An observation with a hard
constraint violation is listed as a review case and is not converted into a
negative training example. After excluding the current target, at least two
independent supporting cases are still required before a learned prior changes
a route score.

The residual leave-one-out boundary accepts only the target measurements,
candidate strokes, source/code point, rule model, and normative catalog. It has
no annotation or expected-answer parameter. The held-out truth may be opened
only after that prediction is frozen. Its report aggregates candidate
correctness, pen-down error, directed DTW, stroke/component IoU, forbidden
branch leakage, residual unexplained ink, safe/review counts, and named hard
violations. Any hard violation forces `needs-review`, irrespective of the
average score or a caller-provided optimistic status.
