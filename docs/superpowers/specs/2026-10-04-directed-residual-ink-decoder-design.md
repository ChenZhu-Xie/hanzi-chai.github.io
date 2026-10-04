# Directed Residual-Ink Stroke Decoder

## Status

Design approved in conversation on 2026-10-04. This document defines the
replacement for the current route-owner flood; it does not authorize automatic
database writes.

## Intent

Recover an ordered, directed stroke decomposition of each Unicode PDF source
glyph while preserving the repository's existing component tree and candidate
workflow. The decoder must learn from human-directed cyan stroke annotations,
but its decisions must remain inspectable as stroke-order, component, topology,
and residual-ink evidence.

Success means that the decoder can:

- place each pen-down point on the intended remaining stroke;
- traverse only the half-edges allowed by that directed stroke at a junction;
- assign the PDF ink around the selected route without leaking into another
  stroke;
- close leaf components in writing order;
- retain ambiguity instead of forcing an unsupported candidate or component;
- explain every decision in the review HTML.

The first mandatory regressions are the ordinary grass head, hand radical, and
old-head failures already observed by the human reviewer.

## Normative Evidence and Scope

The decoder distinguishes normative stroke semantics from glyph geometry.

- GF 0023-2020 defines stroke order as both stroke sequence and direction, and
  treats stroke count, components, separation/contact/intersection, and
  component position as character-form evidence. It is the primary normative
  source for covered G-source forms.
- GF 3002-1999 is a broader G-source fallback for characters outside the 8,105
  characters in GF 0023-2020.
- Taiwan Ministry of Education stroke rules and official stroke SVGs are the
  preferred normative source for covered T-source forms.
- H, J, K, KP/N, U, V, and unsupported forms use exact verified repository
  component templates and human truth first. A G-source order must not be
  silently projected onto them.

General rules such as top-to-bottom, left-to-right, horizontal-before-vertical,
and outside-before-inside are scoped constraints, not universal global sort
keys. They rank feasible strokes inside the applicable character structure or
component. An exact source-specific or component-specific sequence overrides a
generic rule.

Evidence priority is:

1. verified source-character stroke truth;
2. verified exact leaf-component template in the same source convention;
3. applicable source-region normative sequence;
4. candidate tree, stroke feature, direction, turns, and contact grammar;
5. generic structural stroke-order rules;
6. optical similarity and learned statistical priors.

Lower-priority evidence cannot overturn a contradiction at a higher level. It
can only choose between solutions still feasible at the higher levels.

## Existing Failure

The current pipeline first selects directed centerline routes, then
`geodesic_owners` seeds those routes and runs a multi-source shortest-path flood
over the entire skeleton graph. Every unselected skeleton branch is therefore
assigned to some stroke. This makes an ordinary grass-head horizontal own the
upper halves of the two verticals and lets a hand-radical vertical hook own part
of the following rising stroke.

The current beam state stores occupied centerline pixels, not the thick PDF ink
already explained by earlier strokes. Every new stroke is consequently chosen
against almost the same full glyph. Pen-down location is merely a weak global
ordinal tie-breaker, which allowed the two old-head horizontals to be swapped.

These are algorithmic failures. Retraining the current weights cannot enforce
the missing invariants.

## Considered Approaches

### 1. Retune the current global route and owner weights

This is small but rejected. A whole-skeleton Voronoi flood necessarily owns
unselected branches, regardless of training quality. Additional penalties can
move the error without removing it.

### 2. Solve one global graph partition using ILP/CRF-style optimization

This could express route exclusivity and whole-glyph reconstruction jointly.
It is harder to inspect, expensive to enumerate, and awkward when a later
stroke must reuse an earlier contact zone. It remains a possible future solver,
not the first implementation.

### 3. Sequential residual-ink decoding with global beam backtracking

This is the selected approach. It follows actual writing order, makes the next
pen-down conditional on what has already been explained, and gates every
junction explicitly. A beam preserves alternative histories so an early local
choice can be revised when later strokes become impossible.

## Architecture

The new decoder has six independently testable units.

### Source-aware stroke grammar

Compiles a candidate into ordered stroke expectations containing source,
component ID and occurrence, stroke feature, direction, turn sequence, contact
roles, and component-closure boundary. It attaches applicable normative rules
and records their provenance.

### Directed half-edge skeleton

Splits each skeleton junction into directed incoming and outgoing half-edges.
A route transition is legal only when its direction, turn, and expected contact
permit that exact outgoing half-edge. Non-selected exits are barriers, not
corridors that an ownership flood may later enter.

### Residual-ink ledger

Maintains three distinct states after each stroke:

- unexplained ink, still available for ordinary future strokes;
- reusable contact evidence at a narrow intersection or attachment zone;
- explained non-contact ink, unavailable to future stroke starts or routes.

This is a soft, topology-aware subtraction rather than destructive bitmap
erasure. It preserves enough evidence for a later crossing stroke without
letting that stroke re-enter an arbitrary earlier branch.

### Pen-down proposer

Enumerates feasible starts from residual skeleton endpoints, corners, and
junction half-edges. It first filters by expected component occurrence, stroke
feature, direction, and source-specific order. Within the remaining choices it
uses scoped top-to-bottom and left-to-right order, learned component-relative
start priors, and local residual support.

For an old-head horizontal, the first expected feature is horizontal and the
applicable structural rule is top-to-bottom. The upper short horizontal is
therefore selected before the lower long horizontal; the lower one becomes the
next horizontal after the first stroke and vertical have been consumed.

### Directed route and width solver

Routes the current stroke only through legal half-edges. At a cross, T, or L
junction it records the chosen exit and explicit forbidden exits. Stroke width
is then recovered by expansion normal to the selected centerline, clipped by
the PDF outline, medial boundaries, and junction gates. It never propagates
longitudinally along an unselected skeleton road.

### Sequential beam decoder

Each beam state contains the ordered routes, residual-ink ledger, consumed
half-edges, component closures, accumulated evidence, and failure reasons. A
state may advance only by consuming the next expected directed stroke. A closed
component occurrence cannot be re-entered except through an explicitly reusable
contact zone.

## Intersection and Display Semantics

An intersection has two representations:

- During inference, its narrow contact region may provide evidence to every
  stroke that legitimately traverses or attaches there.
- In the final mutually exclusive coloured display, the later-written stroke
  owns the visible overlap, matching painter's order.

Thus evidence sharing does not imply colour leakage or ownership sharing.

## Scoring and Hard Gates

Hard constraints reject a hypothesis before scoring when it:

- starts in explained non-contact ink;
- uses a forbidden junction exit;
- reverses the expected stroke direction;
- violates an exact verified stroke or component sequence;
- re-enters a closed component occurrence;
- leaves a required later stroke with no feasible route.

Scores rank only surviving hypotheses. Terms include pen-down and pen-up error,
directed trajectory distance, turn and contact-role agreement, stroke-region
precision/recall, residual unexplained ink, component closure, and competing
hypothesis margin. Whole-glyph coverage is not an unconditional reward.

Learned rules supply bounded priors. They cannot make an illegal route legal.

## Review and Safety

Every result reports:

- the normative and learned rules used for each stroke;
- pen-down candidates and why each was accepted or rejected;
- selected and forbidden half-edges at every junction;
- residual ink before and after the stroke;
- reusable contact zones;
- per-stroke region precision, recall, and leakage;
- best and runner-up histories and their margin.

A result remains `needs-review` when source-specific order is unavailable, a
new component or sibling is implicated, the residual cannot be fully explained,
the best hypotheses are close, or any hard constraint has no feasible path.

This decoder does not call `createGlyph`, `updateCharacter`, or any other write
API. Admin application remains a later, separately reviewed stage.

## Test Strategy

Implementation follows test-first development.

### Deterministic unit fixtures

- A selected horizontal through two vertical contacts may own the intersection
  disks but no vertical half-edge beyond them.
- A hand-radical vertical hook may contact the later rising stroke but may not
  own its diagonal exit.
- A reusable intersection remains feasible to a later stroke; final display
  ownership is assigned to that later stroke.
- Hard residual subtraction prevents a later start in explained non-contact
  ink.
- Scoped top-to-bottom order selects the upper old-head horizontal first.
- Source-specific order overrides a conflicting generic rule.

### Named regression fixtures

- U+66DA J and U+6726 J, component 228: horizontal, vertical, vertical, with no
  horizontal leakage along either vertical.
- U+6418 G, component 220: horizontal, vertical-hook, rising stroke, with the
  rising stroke retaining its full region.
- U+6418 G, component 439: upper short horizontal before the lower long
  horizontal.
- U+64CE T, component 486: the verified horizontal/vertical sequence is retained
  under T-source rules.

### Corpus evaluation

Run leave-one-out evaluation over every available human annotation. Report
candidate selection, mean pen-down error, directed trajectory DTW, forbidden
branch leakage, per-stroke and per-component IoU, residual unexplained ink, and
safe/needs-review coverage. The held-out annotation is never read during route
selection.

The new decoder is not accepted merely for improving an average. All named
topology regressions must pass, no hard constraint may be violated, and any
remaining ambiguous result must be downgraded to `needs-review`.

## Rollout

1. Introduce the half-edge graph, residual ledger, and deterministic fixtures
   behind an experimental decoder entry point.
2. Reproduce the three named failure families without learned priors.
3. Add source-aware normative adapters and compare them with cyan annotations.
4. Add bounded learned priors from the remaining training annotations.
5. Run strict leave-one-out evaluation and render the existing interactive HTML
   with new route, residual, and junction evidence.
6. Keep the current decoder available as a baseline until the human reviewer
   accepts the new visual results.

## Non-goals

- Inferring stroke order from PDF outline geometry alone.
- Treating candidate coordinates, scale, or occupied area as ground truth.
- Forcing every IRG source to use mainland G-source stroke order.
- Automatically creating a new component solely from an optical mismatch.
- Applying results to upstream data during this decoder-validation phase.

## References

- GF 0023-2020, `https://hudong.moe.gov.cn/jyb_sjzl/ziliao/A19/202103/W020210318300204215237.pdf`
- GF 3002-1999, `https://www.moe.gov.cn/jyb_sjzl/ziliao/A19/201001/W020150902457900316281.pdf`
- Taiwan Ministry of Education stroke-order principles,
  `https://stroke-order.learningweb.moe.edu.tw/page.jsp?ID=23`
- Taiwan Ministry of Education stroke SVG/data description,
  `https://stroke-order.learningweb.moe.edu.tw/page.jsp?ID=11`
