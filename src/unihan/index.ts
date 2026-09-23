import {
  type 基本字形数据,
  type 复合体数据,
  type 字形来源数据,
  type 字符数据,
  来源排序,
} from "hanzi-chai";

export const CJK_UNIFIED_START = 0x4e00;
export const CJK_UNIFIED_END = 0x9fff;
export const SUPPORTED_UNIHAN_VERSION = "18.0.0";
export const DEFAULT_MINIMUM_EVIDENCE = 3;
export const DEFAULT_MINIMUM_DOMINANCE = 2;
export const MINIMUM_UNCONTESTED_IDENTITY_EVIDENCE = 2;

export const IRG_PROPERTY_TO_SOURCE = new Map<string, string>([
  ["kIRG_GSource", "G"],
  ["kIRG_HSource", "H"],
  ["kIRG_JSource", "J"],
  ["kIRG_KSource", "K"],
  ["kIRG_KPSource", "N"],
  ["kIRG_MSource", "M"],
  ["kIRG_SSource", "S"],
  ["kIRG_TSource", "T"],
  ["kIRG_UKSource", "B"],
  ["kIRG_USource", "U"],
  ["kIRG_VSource", "V"],
]);

export const RECOMMENDATION_SAMPLE_RANGES = [
  { start: 0x4e00, end: 0x6400 },
  { start: 0x7a70, end: 0x7aca },
  { start: 0x7cf8, end: 0x7f35 },
  { start: 0x8fb6, end: 0x9090 },
  { start: 0x96e8, end: 0x9761 },
  { start: 0x8278, end: 0x827f },
] as const;

export type UnihanSourceMap = Map<number, string[]>;

export type UnihanAuditStatus =
  | "reviewed-reference"
  | "already-complete"
  | "safe-candidate"
  | "needs-review"
  | "existing-non-g-data"
  | "insufficient-evidence"
  | "unsupported";

export interface RecommendationEvidence {
  referenceId: number;
  replacementId?: number;
  count: number;
  reliable: boolean;
  reviewed?: boolean;
  /** Accepted with two identity examples because no sibling competes. */
  uncontestedIdentity?: boolean;
  alternatives: { id: number; count: number; examples: number[] }[];
  /** Nested evidence used to resolve this top-level reference recursively. */
  basis?: RecommendationEvidence[];
}

export interface UnihanGlyphProposal {
  source: string;
  existingGlyphId?: number;
  requiresChange: boolean;
  glyph: 基本字形数据;
  evidence: RecommendationEvidence[];
}

export interface UnihanUnresolvedSource {
  source: string;
  evidence: RecommendationEvidence[];
}

export type ManualSourceChoices = ReadonlyMap<number, number>;

export interface UnihanAuditItem {
  unicode: number;
  codepoint: string;
  character: string;
  expectedSources: string[];
  existingSources: string[];
  missingSources: string[];
  extraSources: string[];
  existingGlyphs: 字形来源数据[];
  status: UnihanAuditStatus;
  reason: string;
  proposals: UnihanGlyphProposal[];
  unresolved: UnihanUnresolvedSource[];
}

export interface UnihanAuditSummary {
  scanned: number;
  hasNonG: number;
  reviewedReferences: number;
  alreadyComplete: number;
  safeCandidates: number;
  needsReview: number;
  skippedExistingNonG: number;
  insufficientEvidence: number;
  unsupported: number;
}

export interface UnihanAudit {
  metadata: {
    unicodeVersion: typeof SUPPORTED_UNIHAN_VERSION;
    range: [string, string];
    sourceOrder: string;
    generatedAt: string;
    minimumEvidence: number;
    minimumDominance: number;
    minimumUncontestedIdentityEvidence: number;
    reviewedDecisions: ReviewedSourceDecision[];
  };
  summary: UnihanAuditSummary;
  items: UnihanAuditItem[];
}

type EvidenceExamples = Map<number, Set<number>>;

export type SourceEvidenceIndex = Map<string, Map<number, EvidenceExamples>>;

export type GlyphEvidenceIndex = Map<string, Map<number, EvidenceExamples>>;

export interface ReviewedSourceDecision {
  unicode: number;
  source: string;
  referenceId: number;
  replacementId: number;
  provenance?: "manual" | "reviewed-data";
}

export interface GlyphSiblingExample {
  unicode: number;
  source: string;
  parentGlyphId: number;
  position: number;
}

export interface GlyphSiblingCandidate {
  id: number;
  count: number;
  sources: string[];
  examples: GlyphSiblingExample[];
}

interface GlyphIndex {
  byId: Map<number, 基本字形数据>;
  byShape: Map<string, number>;
}

export function sortSources(values: Iterable<string>): string[] {
  return [...new Set(values)].sort((a, b) => {
    const left = 来源排序.indexOf(a);
    const right = 来源排序.indexOf(b);
    if (left === -1 && right === -1) return a.localeCompare(b);
    if (left === -1) return 1;
    if (right === -1) return -1;
    return left - right;
  });
}

export function parseUnihanIRGSources(
  text: string,
  from = CJK_UNIFIED_START,
  to = CJK_UNIFIED_END,
): UnihanSourceMap {
  const raw = new Map<number, Set<string>>();
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const [codepoint, property, value] = line.split("\t");
    if (!codepoint || !property || !value?.trim()) continue;
    const matched = /^U\+([0-9A-F]{4,6})$/i.exec(codepoint);
    if (!matched) continue;
    const source = IRG_PROPERTY_TO_SOURCE.get(property);
    if (!source) continue;
    const unicode = Number.parseInt(matched[1]!, 16);
    if (unicode < from || unicode > to) continue;
    const sources = raw.get(unicode) ?? new Set<string>();
    sources.add(source);
    raw.set(unicode, sources);
  }
  return new Map(
    [...raw]
      .sort(([a], [b]) => a - b)
      .map(([unicode, sources]) => [unicode, sortSources(sources)]),
  );
}

export function readUnihanVersion(text: string): string | undefined {
  return /^# Unicode Version ([0-9]+(?:\.[0-9]+){1,2})\s*$/m.exec(text)?.[1];
}

export function isRecommendationSample(unicode: number): boolean {
  return RECOMMENDATION_SAMPLE_RANGES.some(
    ({ start, end }) => unicode >= start && unicode <= end,
  );
}

export function buildSourceEvidenceIndex(
  characters: 字符数据[],
  glyphs: 基本字形数据[],
): SourceEvidenceIndex {
  const glyphById = new Map(glyphs.map((glyph) => [glyph.id, glyph]));
  const index: SourceEvidenceIndex = new Map();
  for (const character of characters) {
    if (!isRecommendationSample(character.unicode)) continue;
    const gEntry = character.glyphs.find((entry) =>
      entry.sources.includes("G"),
    );
    if (!gEntry) continue;
    const gGlyph = glyphById.get(gEntry.id);
    if (gGlyph?.type !== "compound") continue;

    const seenEvidenceBySource = new Map<string, Set<string>>();
    for (const targetEntry of character.glyphs) {
      const targetGlyph = glyphById.get(targetEntry.id);
      if (targetGlyph?.type !== "compound") continue;
      if (targetGlyph.operator !== gGlyph.operator) continue;
      if (targetGlyph.references.length !== gGlyph.references.length) continue;
      for (const source of targetEntry.sources) {
        if (source === "G") continue;
        const seenInCharacter =
          seenEvidenceBySource.get(source) ?? new Set<string>();
        seenEvidenceBySource.set(source, seenInCharacter);
        let sourceIndex = index.get(source);
        if (!sourceIndex) {
          sourceIndex = new Map();
          index.set(source, sourceIndex);
        }
        for (
          let position = 0;
          position < gGlyph.references.length;
          position++
        ) {
          const referenceId = gGlyph.references[position]!.id;
          const replacementId = targetGlyph.references[position]!.id;
          if (!glyphById.has(replacementId)) continue;
          const evidenceKey = `${referenceId}:${replacementId}`;
          if (seenInCharacter.has(evidenceKey)) continue;
          seenInCharacter.add(evidenceKey);
          let replacements = sourceIndex.get(referenceId);
          if (!replacements) {
            replacements = new Map();
            sourceIndex.set(referenceId, replacements);
          }
          const examples = replacements.get(replacementId) ?? new Set<number>();
          examples.add(character.unicode);
          replacements.set(replacementId, examples);
        }
      }
    }
  }
  return index;
}

/**
 * Add a maintainer decision at the smallest known differing subtree and also
 * propagate it down structurally compatible descendants. This makes one
 * reviewed sibling decision reusable without treating all forms from the same
 * IRG source as globally identical.
 */
export function augmentSourceEvidenceWithReviews(
  index: SourceEvidenceIndex,
  decisions: ReviewedSourceDecision[],
  glyphs: 基本字形数据[],
): SourceEvidenceIndex {
  const glyphById = new Map(glyphs.map((glyph) => [glyph.id, glyph]));
  const add = (
    source: string,
    referenceId: number,
    replacementId: number,
    unicode: number,
    depth = 0,
  ) => {
    if (depth > 10) return;
    let sourceIndex = index.get(source);
    if (!sourceIndex) {
      sourceIndex = new Map();
      index.set(source, sourceIndex);
    }
    let replacements = sourceIndex.get(referenceId);
    if (!replacements) {
      replacements = new Map();
      sourceIndex.set(referenceId, replacements);
    }
    const examples = replacements.get(replacementId) ?? new Set<number>();
    examples.add(unicode);
    replacements.set(replacementId, examples);

    const reference = glyphById.get(referenceId);
    const replacement = glyphById.get(replacementId);
    if (
      reference?.type !== "compound" ||
      replacement?.type !== "compound" ||
      reference.operator !== replacement.operator ||
      reference.references.length !== replacement.references.length
    ) {
      return;
    }
    for (let index = 0; index < reference.references.length; index++) {
      add(
        source,
        reference.references[index]!.id,
        replacement.references[index]!.id,
        unicode,
        depth + 1,
      );
    }
  };
  for (const decision of decisions) {
    if (
      !glyphById.has(decision.referenceId) ||
      !glyphById.has(decision.replacementId)
    ) {
      continue;
    }
    add(
      decision.source,
      decision.referenceId,
      decision.replacementId,
      decision.unicode,
    );
  }
  return index;
}

export function buildGlyphEvidenceIndex(
  characters: 字符数据[],
  glyphs: 基本字形数据[],
): GlyphEvidenceIndex {
  const glyphIds = new Set(glyphs.map(({ id }) => id));
  const index: GlyphEvidenceIndex = new Map();
  for (const character of characters) {
    if (!isRecommendationSample(character.unicode)) continue;
    const gEntry = character.glyphs.find((entry) =>
      entry.sources.includes("G"),
    );
    if (!gEntry || !glyphIds.has(gEntry.id)) continue;
    for (const targetEntry of character.glyphs) {
      if (!glyphIds.has(targetEntry.id)) continue;
      for (const source of targetEntry.sources) {
        if (source === "G") continue;
        let byReference = index.get(source);
        if (!byReference) {
          byReference = new Map();
          index.set(source, byReference);
        }
        let replacements = byReference.get(gEntry.id);
        if (!replacements) {
          replacements = new Map();
          byReference.set(gEntry.id, replacements);
        }
        const examples = replacements.get(targetEntry.id) ?? new Set<number>();
        examples.add(character.unicode);
        replacements.set(targetEntry.id, examples);
      }
    }
  }
  return index;
}

function addSiblingExample(
  raw: Map<number, Map<number, GlyphSiblingExample[]>>,
  left: number,
  right: number,
  example: GlyphSiblingExample,
) {
  if (left === right) return;
  const siblings = raw.get(left) ?? new Map<number, GlyphSiblingExample[]>();
  const examples = siblings.get(right) ?? [];
  if (
    !examples.some(
      (item) =>
        item.unicode === example.unicode &&
        item.source === example.source &&
        item.position === example.position,
    )
  ) {
    examples.push(example);
  }
  siblings.set(right, examples);
  raw.set(left, siblings);
}

/**
 * Derive candidate sibling glyphs only from the maintainer-reviewed ranges.
 * The relation is symmetric, but each example retains the source direction
 * that produced it.
 */
export function buildReviewedGlyphSiblingIndex(
  characters: 字符数据[],
  glyphs: 基本字形数据[],
): Map<number, GlyphSiblingCandidate[]> {
  const glyphById = new Map(glyphs.map((glyph) => [glyph.id, glyph]));
  const raw = new Map<number, Map<number, GlyphSiblingExample[]>>();
  for (const character of characters) {
    if (!isRecommendationSample(character.unicode)) continue;
    const gEntry = character.glyphs.find((entry) =>
      entry.sources.includes("G"),
    );
    const gGlyph = gEntry ? glyphById.get(gEntry.id) : undefined;
    if (!gEntry || !gGlyph) continue;
    for (const targetEntry of character.glyphs) {
      const targetGlyph = glyphById.get(targetEntry.id);
      if (!targetGlyph) continue;
      for (const source of targetEntry.sources) {
        if (source === "G") continue;
        if (gGlyph.type === "component" && targetGlyph.type === "component") {
          const example = {
            unicode: character.unicode,
            source,
            parentGlyphId: gEntry.id,
            position: -1,
          };
          addSiblingExample(raw, gEntry.id, targetEntry.id, example);
          addSiblingExample(raw, targetEntry.id, gEntry.id, example);
          continue;
        }
        if (
          gGlyph.type !== "compound" ||
          targetGlyph.type !== "compound" ||
          gGlyph.operator !== targetGlyph.operator ||
          gGlyph.references.length !== targetGlyph.references.length
        ) {
          continue;
        }
        for (
          let position = 0;
          position < gGlyph.references.length;
          position++
        ) {
          const left = gGlyph.references[position]!.id;
          const right = targetGlyph.references[position]!.id;
          const example = {
            unicode: character.unicode,
            source,
            parentGlyphId: targetEntry.id,
            position,
          };
          addSiblingExample(raw, left, right, example);
          addSiblingExample(raw, right, left, example);
        }
      }
    }
  }
  return new Map(
    [...raw].map(([id, siblings]) => [
      id,
      [...siblings]
        .map(([siblingId, examples]) => ({
          id: siblingId,
          count: new Set(examples.map(({ unicode }) => unicode)).size,
          sources: sortSources(examples.map(({ source }) => source)),
          examples: examples.sort(
            (a, b) => a.unicode - b.unicode || a.source.localeCompare(b.source),
          ),
        }))
        .sort((a, b) => b.count - a.count || a.id - b.id),
    ]),
  );
}

export function glyphShapeKey(glyph: 基本字形数据): string {
  const {
    id: _id,
    name: _name,
    gf0014_id: _gf0014,
    gf3001_id: _gf3001,
    ...shape
  } = glyph;
  return JSON.stringify(shape);
}

function buildGlyphIndex(glyphs: 基本字形数据[]): GlyphIndex {
  return {
    byId: new Map(glyphs.map((glyph) => [glyph.id, glyph])),
    byShape: new Map(glyphs.map((glyph) => [glyphShapeKey(glyph), glyph.id])),
  };
}

function makeRecommendationEvidence(
  referenceId: number,
  replacements: EvidenceExamples | undefined,
  minimumEvidence: number,
  minimumDominance: number,
  reviewedDecision?: ReviewedSourceDecision,
  allowUncontestedIdentity = false,
): RecommendationEvidence {
  const alternatives = [...(replacements ?? [])]
    .map(([id, examples]) => ({
      id,
      count: examples.size,
      examples: [...examples].sort((a, b) => a - b),
    }))
    .sort((a, b) => b.count - a.count || a.id - b.id);
  if (
    reviewedDecision &&
    !alternatives.some(({ id }) => id === reviewedDecision.replacementId)
  ) {
    alternatives.push({
      id: reviewedDecision.replacementId,
      count: 1,
      examples: [reviewedDecision.unicode],
    });
  }
  if (reviewedDecision) {
    const reviewedAlternative = alternatives.find(
      ({ id }) => id === reviewedDecision.replacementId,
    )!;
    return {
      referenceId,
      replacementId: reviewedDecision.replacementId,
      count: reviewedAlternative.count,
      reliable: true,
      reviewed: true,
      alternatives,
    };
  }
  const winner = alternatives[0];
  const runnerUp = alternatives[1];
  const meetsStandardThreshold =
    winner !== undefined &&
    winner.count >= minimumEvidence &&
    (runnerUp === undefined ||
      winner.count >= runnerUp.count * minimumDominance);
  const uncontestedIdentity =
    !meetsStandardThreshold &&
    allowUncontestedIdentity &&
    winner?.id === referenceId &&
    winner.count >= MINIMUM_UNCONTESTED_IDENTITY_EVIDENCE &&
    alternatives.length === 1;
  return {
    referenceId,
    replacementId: winner?.id,
    count: winner?.count ?? 0,
    reliable: meetsStandardThreshold || uncontestedIdentity,
    uncontestedIdentity: uncontestedIdentity || undefined,
    alternatives,
  };
}

function resolveReferenceRecursively(
  referenceId: number,
  source: string,
  glyphIndex: GlyphIndex,
  evidenceIndex: SourceEvidenceIndex,
  minimumEvidence: number,
  minimumDominance: number,
  reviewedDecisions: ReadonlyMap<number, ReviewedSourceDecision>,
  knownSiblingIds?: ReadonlySet<number>,
  depth = 0,
  seen = new Set<number>(),
): { id: number; evidence: RecommendationEvidence } | undefined {
  if (depth > 10 || seen.has(referenceId)) return undefined;
  const nextSeen = new Set(seen).add(referenceId);
  const direct = makeRecommendationEvidence(
    referenceId,
    evidenceIndex.get(source)?.get(referenceId),
    minimumEvidence,
    minimumDominance,
    reviewedDecisions.get(referenceId),
    knownSiblingIds !== undefined && !knownSiblingIds.has(referenceId),
  );
  if (
    direct.reliable &&
    direct.replacementId !== undefined &&
    glyphIndex.byId.has(direct.replacementId)
  ) {
    return { id: direct.replacementId, evidence: direct };
  }
  const glyph = glyphIndex.byId.get(referenceId);
  if (glyph?.type !== "compound") return undefined;
  const resolved = glyph.references.map((reference) =>
    resolveReferenceRecursively(
      reference.id,
      source,
      glyphIndex,
      evidenceIndex,
      minimumEvidence,
      minimumDominance,
      reviewedDecisions,
      knownSiblingIds,
      depth + 1,
      nextSeen,
    ),
  );
  if (resolved.some((result) => result === undefined)) return undefined;
  const candidate: 复合体数据 = {
    ...glyph,
    id: 0,
    references: glyph.references.map((reference, index) => ({
      ...reference,
      id: resolved[index]!.id,
    })),
  };
  const existingId = glyphIndex.byShape.get(glyphShapeKey(candidate));
  if (existingId === undefined) return undefined;
  const basis = [
    ...(direct.alternatives.length > 0 ? [direct] : []),
    ...resolved.map((result) => result!.evidence),
  ];
  const reliableCounts = basis
    .flatMap((item) => (item.basis ? item.basis : [item]))
    .filter((item) => item.reliable)
    .map((item) => item.count);
  const examples = new Set(
    basis.flatMap((item) =>
      item.alternatives
        .filter(({ id }) => id === item.replacementId)
        .flatMap((alternative) => alternative.examples),
    ),
  );
  return {
    id: existingId,
    evidence: {
      referenceId,
      replacementId: existingId,
      count: reliableCounts.length > 0 ? Math.min(...reliableCounts) : 0,
      reliable: true,
      reviewed: basis.some(
        (item) => item.reviewed || item.basis?.some((child) => child.reviewed),
      ),
      alternatives: [
        {
          id: existingId,
          count: reliableCounts.length > 0 ? Math.min(...reliableCounts) : 0,
          examples: [...examples].sort((left, right) => left - right),
        },
      ],
      basis,
    },
  };
}

export function recommendMissingSources(
  character: 字符数据,
  missingSources: string[],
  glyphs: 基本字形数据[],
  evidenceIndex: SourceEvidenceIndex,
  minimumEvidence = DEFAULT_MINIMUM_EVIDENCE,
  minimumDominance = DEFAULT_MINIMUM_DOMINANCE,
  existingGlyphIndex?: GlyphIndex,
  glyphEvidenceIndex = buildGlyphEvidenceIndex([character], glyphs),
  reviewedDecisions: ReviewedSourceDecision[] = [],
  knownSiblingIds?: ReadonlySet<number>,
): {
  proposals: UnihanGlyphProposal[];
  unresolvedSources: string[];
  unresolved: UnihanUnresolvedSource[];
} {
  const gEntry = character.glyphs.find((entry) => entry.sources.includes("G"));
  const glyphIndex = existingGlyphIndex ?? buildGlyphIndex(glyphs);
  const referenceGlyph = gEntry ? glyphIndex.byId.get(gEntry.id) : undefined;
  if (!gEntry || !referenceGlyph) {
    return {
      proposals: [],
      unresolvedSources: missingSources,
      unresolved: missingSources.map((source) => ({ source, evidence: [] })),
    };
  }
  const proposals: UnihanGlyphProposal[] = [];
  const unresolvedSources: string[] = [];
  const unresolved: UnihanUnresolvedSource[] = [];
  const reviewedBySource = new Map<
    string,
    Map<number, ReviewedSourceDecision>
  >();
  for (const decision of reviewedDecisions) {
    if (decision.unicode !== character.unicode) continue;
    const byReference = reviewedBySource.get(decision.source) ?? new Map();
    byReference.set(decision.referenceId, decision);
    reviewedBySource.set(decision.source, byReference);
  }

  for (const source of missingSources) {
    const sourceReviews = reviewedBySource.get(source) ?? new Map();
    if (referenceGlyph.type === "component") {
      const evidence = makeRecommendationEvidence(
        referenceGlyph.id,
        glyphEvidenceIndex.get(source)?.get(referenceGlyph.id),
        minimumEvidence,
        minimumDominance,
        sourceReviews.get(referenceGlyph.id),
        knownSiblingIds !== undefined &&
          !knownSiblingIds.has(referenceGlyph.id),
      );
      const winner = evidence.replacementId
        ? glyphIndex.byId.get(evidence.replacementId)
        : undefined;
      if (!evidence.reliable || winner?.type !== "component") {
        unresolvedSources.push(source);
        unresolved.push({ source, evidence: [evidence] });
        continue;
      }
      proposals.push({
        source,
        existingGlyphId: winner.id,
        requiresChange: winner.id !== referenceGlyph.id,
        glyph: { ...winner, id: 0 },
        evidence: [evidence],
      });
      continue;
    }

    const sourceIndex = evidenceIndex.get(source);
    const evidence: RecommendationEvidence[] = [];
    const replacements: number[] = [];
    let reliable = true;
    for (const reference of referenceGlyph.references) {
      const componentEvidence = makeRecommendationEvidence(
        reference.id,
        sourceIndex?.get(reference.id),
        minimumEvidence,
        minimumDominance,
        sourceReviews.get(reference.id),
        knownSiblingIds !== undefined && !knownSiblingIds.has(reference.id),
      );
      const recursive = componentEvidence.reliable
        ? undefined
        : resolveReferenceRecursively(
            reference.id,
            source,
            glyphIndex,
            evidenceIndex,
            minimumEvidence,
            minimumDominance,
            sourceReviews,
            knownSiblingIds,
          );
      if (recursive) {
        evidence.push(recursive.evidence);
        replacements.push(recursive.id);
      } else if (
        componentEvidence.reliable &&
        componentEvidence.replacementId !== undefined
      ) {
        evidence.push(componentEvidence);
        replacements.push(componentEvidence.replacementId);
      } else {
        evidence.push(componentEvidence);
        reliable = false;
      }
    }
    if (!reliable) {
      unresolvedSources.push(source);
      unresolved.push({ source, evidence });
      continue;
    }
    const glyph: 复合体数据 = {
      id: 0,
      type: "compound",
      operator: referenceGlyph.operator,
      references: referenceGlyph.references.map((reference, index) => ({
        ...reference,
        id: replacements[index]!,
      })),
      strokes: referenceGlyph.strokes,
      ambiguous: referenceGlyph.ambiguous,
    };
    const shapeKey = glyphShapeKey(glyph);
    const requiresChange = shapeKey !== glyphShapeKey(referenceGlyph);
    const existingGlyphId = requiresChange
      ? glyphIndex.byShape.get(shapeKey)
      : referenceGlyph.id;
    proposals.push({
      source,
      existingGlyphId,
      requiresChange,
      glyph,
      evidence,
    });
  }
  return { proposals, unresolvedSources, unresolved };
}

/**
 * Turn one maintainer-reviewed set of component choices into a proposal.
 * Every manual replacement must still be one of the alternatives produced by
 * the fresh audit, so a stale UI choice cannot silently bypass revalidation.
 */
export function resolveUnresolvedSourceProposal(
  character: 字符数据,
  glyphs: 基本字形数据[],
  unresolved: UnihanUnresolvedSource,
  choices: ManualSourceChoices,
): UnihanGlyphProposal | undefined {
  const glyphIndex = buildGlyphIndex(glyphs);
  const gEntry = character.glyphs.find((entry) => entry.sources.includes("G"));
  const referenceGlyph = gEntry ? glyphIndex.byId.get(gEntry.id) : undefined;
  if (!referenceGlyph) return undefined;

  const replacements = unresolved.evidence.map((evidence) => {
    const replacementId = evidence.reliable
      ? evidence.replacementId
      : choices.get(evidence.referenceId);
    if (
      replacementId === undefined ||
      !evidence.alternatives.some(({ id }) => id === replacementId) ||
      !glyphIndex.byId.has(replacementId)
    ) {
      return undefined;
    }
    return replacementId;
  });
  if (replacements.some((id) => id === undefined)) return undefined;

  if (referenceGlyph.type === "component") {
    const replacement = glyphIndex.byId.get(replacements[0]!);
    if (replacement?.type !== "component") return undefined;
    return {
      source: unresolved.source,
      existingGlyphId: replacement.id,
      requiresChange: replacement.id !== referenceGlyph.id,
      glyph: { ...replacement, id: 0 },
      evidence: unresolved.evidence,
    };
  }
  if (replacements.length !== referenceGlyph.references.length) {
    return undefined;
  }
  const glyph: 复合体数据 = {
    id: 0,
    type: "compound",
    operator: referenceGlyph.operator,
    references: referenceGlyph.references.map((reference, index) => ({
      ...reference,
      id: replacements[index]!,
    })),
    strokes: referenceGlyph.strokes,
    ambiguous: referenceGlyph.ambiguous,
  };
  const shapeKey = glyphShapeKey(glyph);
  const requiresChange = shapeKey !== glyphShapeKey(referenceGlyph);
  return {
    source: unresolved.source,
    existingGlyphId: requiresChange
      ? glyphIndex.byShape.get(shapeKey)
      : referenceGlyph.id,
    requiresChange,
    glyph,
    evidence: unresolved.evidence,
  };
}

function codepoint(unicode: number): string {
  return `U+${unicode.toString(16).toUpperCase().padStart(4, "0")}`;
}

export function auditUnihanSources(
  expectedByUnicode: UnihanSourceMap,
  characters: 字符数据[],
  glyphs: 基本字形数据[],
  options: {
    from?: number;
    to?: number;
    minimumEvidence?: number;
    minimumDominance?: number;
    reviewedDecisions?: ReviewedSourceDecision[];
  } = {},
): UnihanAudit {
  const from = options.from ?? CJK_UNIFIED_START;
  const to = options.to ?? CJK_UNIFIED_END;
  const minimumEvidence = options.minimumEvidence ?? DEFAULT_MINIMUM_EVIDENCE;
  const minimumDominance =
    options.minimumDominance ?? DEFAULT_MINIMUM_DOMINANCE;
  const charactersByUnicode = new Map(
    characters.map((character) => [character.unicode, character]),
  );
  const glyphIndex = buildGlyphIndex(glyphs);
  const evidenceIndex = buildSourceEvidenceIndex(characters, glyphs);
  augmentSourceEvidenceWithReviews(
    evidenceIndex,
    options.reviewedDecisions ?? [],
    glyphs,
  );
  const glyphEvidenceIndex = buildGlyphEvidenceIndex(characters, glyphs);
  const knownSiblingIndex = buildReviewedGlyphSiblingIndex(characters, glyphs);
  const knownSiblingIds = new Set<number>();
  const markKnownSiblingTree = (
    referenceId: number,
    replacementId: number,
    depth = 0,
  ) => {
    if (depth > 10 || referenceId === replacementId) return;
    knownSiblingIds.add(referenceId);
    knownSiblingIds.add(replacementId);
    const reference = glyphIndex.byId.get(referenceId);
    const replacement = glyphIndex.byId.get(replacementId);
    if (
      reference?.type !== "compound" ||
      replacement?.type !== "compound" ||
      reference.operator !== replacement.operator ||
      reference.references.length !== replacement.references.length
    ) {
      return;
    }
    for (let index = 0; index < reference.references.length; index++) {
      markKnownSiblingTree(
        reference.references[index]!.id,
        replacement.references[index]!.id,
        depth + 1,
      );
    }
  };
  for (const [referenceId, siblings] of knownSiblingIndex) {
    for (const sibling of siblings) {
      markKnownSiblingTree(referenceId, sibling.id);
    }
  }
  for (const decision of options.reviewedDecisions ?? []) {
    markKnownSiblingTree(decision.referenceId, decision.replacementId);
  }
  const summary: UnihanAuditSummary = {
    scanned: 0,
    hasNonG: 0,
    reviewedReferences: 0,
    alreadyComplete: 0,
    safeCandidates: 0,
    needsReview: 0,
    skippedExistingNonG: 0,
    insufficientEvidence: 0,
    unsupported: 0,
  };
  const items: UnihanAuditItem[] = [];

  for (const [unicode, expectedSources] of expectedByUnicode) {
    if (unicode < from || unicode > to) continue;
    summary.scanned++;
    if (!expectedSources.some((source) => source !== "G")) continue;
    summary.hasNonG++;
    const character = charactersByUnicode.get(unicode);
    const existingSources = sortSources(
      character?.glyphs.flatMap((glyph) => glyph.sources) ?? [],
    );
    const expectedSet = new Set(expectedSources);
    const existingSet = new Set(existingSources);
    const missingSources = expectedSources.filter(
      (source) => !existingSet.has(source),
    );
    const extraSources = existingSources.filter(
      (source) => !expectedSet.has(source),
    );
    let status: UnihanAuditStatus;
    let reason: string;
    let proposals: UnihanGlyphProposal[] = [];
    let unresolved: UnihanUnresolvedSource[] = [];

    const hasExistingNonGData =
      character !== undefined &&
      character.glyphs.length > 1 &&
      character.glyphs.some((entry) =>
        entry.sources.some((source) => source !== "G"),
      );

    if (!character) {
      status = "unsupported";
      reason = "当前 hanzi-chai 数据中没有这个字符。";
    } else if (isRecommendationSample(unicode)) {
      status = "reviewed-reference";
      reason = "属于维护者已完成区间；只作为推荐训练样本，不进入自动写入。";
    } else if (!expectedSet.has("G") || !existingSet.has("G")) {
      status = "needs-review";
      reason = "缺少可作为参考的 G 来源字形。";
    } else {
      const expectedNonG = expectedSources.filter((source) => source !== "G");
      const recommendation = recommendMissingSources(
        character,
        expectedNonG,
        glyphs,
        evidenceIndex,
        minimumEvidence,
        minimumDominance,
        glyphIndex,
        glyphEvidenceIndex,
        options.reviewedDecisions ?? [],
        knownSiblingIds,
      );
      proposals = recommendation.proposals;
      unresolved = recommendation.unresolved;
      if (hasExistingNonGData) {
        status = "existing-non-g-data";
        reason =
          "区间外已存在多个来源字形；展示完整 source 建议供复核，但不会自动覆盖。";
      } else if (
        expectedNonG.length > 0 &&
        recommendation.unresolvedSources.length === 0 &&
        proposals.length === expectedNonG.length &&
        extraSources.length === 0
      ) {
        if (
          proposals.some((proposal) => proposal.requiresChange) ||
          missingSources.length > 0
        ) {
          status = "safe-candidate";
          reason = `每个目标来源的完整拆分都可由已完成区间或当前字符的维护者确认推导；统计项通常至少有 ${minimumEvidence} 个独立字符证据，且第一名至少是第二名的 ${minimumDominance} 倍；无已知兄弟且无竞争者的原部件保持可由 ${MINIMUM_UNCONTESTED_IDENTITY_EVIDENCE} 个独立样本确认。`;
        } else {
          status = "already-complete";
          reason =
            "来源标签已完整，且已完成区间的统计证据推断其完整拆分与当前分组相同；这不等于已通过 PDF 人工视觉核验。";
        }
      } else {
        status = "insufficient-evidence";
        reason = `以下来源没有覆盖全部 component 的可靠证据：${recommendation.unresolvedSources.join(", ") || "未知"}。`;
      }
    }

    if (status === "reviewed-reference") summary.reviewedReferences++;
    if (status === "already-complete") summary.alreadyComplete++;
    if (status === "safe-candidate") summary.safeCandidates++;
    if (status === "existing-non-g-data") summary.skippedExistingNonG++;
    if (status === "insufficient-evidence") summary.insufficientEvidence++;
    if (status === "unsupported") summary.unsupported++;
    if (status === "needs-review") summary.needsReview++;
    items.push({
      unicode,
      codepoint: codepoint(unicode),
      character: String.fromCodePoint(unicode),
      expectedSources,
      existingSources,
      missingSources,
      extraSources,
      existingGlyphs: character?.glyphs ?? [],
      status,
      reason,
      proposals,
      unresolved,
    });
  }

  return {
    metadata: {
      unicodeVersion: SUPPORTED_UNIHAN_VERSION,
      range: [codepoint(from), codepoint(to)],
      sourceOrder: 来源排序,
      generatedAt: new Date().toISOString(),
      minimumEvidence,
      minimumDominance,
      minimumUncontestedIdentityEvidence:
        MINIMUM_UNCONTESTED_IDENTITY_EVIDENCE,
      reviewedDecisions: (options.reviewedDecisions ?? []).map((decision) => ({
        ...decision,
      })),
    },
    summary,
    items,
  };
}

export function applySourceAssignments(
  character: 字符数据,
  assignments: { source: string; id: number }[],
): 字符数据 {
  const glyphs = character.glyphs.map((glyph) => ({
    ...glyph,
    sources: [...glyph.sources],
  }));
  for (const { source, id } of assignments) {
    for (const glyph of glyphs) {
      glyph.sources = glyph.sources.filter((existing) => existing !== source);
    }
    const existing = glyphs.find((glyph) => glyph.id === id);
    if (existing) existing.sources = sortSources([...existing.sources, source]);
    else glyphs.push({ id, sources: [source] });
  }
  return {
    ...character,
    glyphs: glyphs.filter((glyph) => glyph.sources.length > 0),
  };
}
