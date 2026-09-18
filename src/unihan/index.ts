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
  | "already-complete"
  | "safe-candidate"
  | "needs-review"
  | "existing-non-g-data"
  | "insufficient-evidence"
  | "unsupported";

export interface RecommendationEvidence {
  referenceId: number;
  replacementId: number;
  count: number;
  alternatives: { id: number; count: number }[];
}

export interface UnihanGlyphProposal {
  source: string;
  existingGlyphId?: number;
  requiresChange: boolean;
  glyph: 复合体数据;
  evidence: RecommendationEvidence[];
}

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
}

export interface UnihanAuditSummary {
  scanned: number;
  hasNonG: number;
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
  };
  summary: UnihanAuditSummary;
  items: UnihanAuditItem[];
}

type SourceEvidenceIndex = Map<string, Map<number, Map<number, number>>>;

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

function isRecommendationSample(unicode: number): boolean {
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
    if (character.glyphs.length < 2) continue;
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
          replacements.set(
            replacementId,
            (replacements.get(replacementId) ?? 0) + 1,
          );
        }
      }
    }
  }
  return index;
}

export function glyphShapeKey(glyph: 复合体数据): string {
  return JSON.stringify({
    type: glyph.type,
    operator: glyph.operator,
    references: glyph.references,
    strokes: glyph.strokes,
    ambiguous: glyph.ambiguous,
  });
}

function buildGlyphIndex(glyphs: 基本字形数据[]): GlyphIndex {
  return {
    byId: new Map(glyphs.map((glyph) => [glyph.id, glyph])),
    byShape: new Map(
      glyphs
        .filter((glyph): glyph is 复合体数据 => glyph.type === "compound")
        .map((glyph) => [glyphShapeKey(glyph), glyph.id]),
    ),
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
): { proposals: UnihanGlyphProposal[]; unresolvedSources: string[] } {
  const gEntry = character.glyphs.find((entry) => entry.sources.includes("G"));
  const glyphIndex = existingGlyphIndex ?? buildGlyphIndex(glyphs);
  const referenceGlyph = gEntry ? glyphIndex.byId.get(gEntry.id) : undefined;
  if (!gEntry || !referenceGlyph || referenceGlyph.type !== "compound") {
    return { proposals: [], unresolvedSources: missingSources };
  }
  const proposals: UnihanGlyphProposal[] = [];
  const unresolvedSources: string[] = [];

  for (const source of missingSources) {
    const sourceIndex = evidenceIndex.get(source);
    const evidence: RecommendationEvidence[] = [];
    const replacements: number[] = [];
    let reliable = true;
    for (const reference of referenceGlyph.references) {
      const alternatives = [...(sourceIndex?.get(reference.id) ?? [])]
        .map(([id, count]) => ({ id, count }))
        .sort((a, b) => b.count - a.count || a.id - b.id);
      const winner = alternatives[0];
      const runnerUp = alternatives[1];
      if (
        !winner ||
        winner.count < minimumEvidence ||
        (runnerUp !== undefined &&
          winner.count < runnerUp.count * minimumDominance)
      ) {
        reliable = false;
        break;
      }
      replacements.push(winner.id);
      evidence.push({
        referenceId: reference.id,
        replacementId: winner.id,
        count: winner.count,
        alternatives,
      });
    }
    if (!reliable) {
      unresolvedSources.push(source);
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
  return { proposals, unresolvedSources };
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
  const glyphsById = new Map(glyphs.map((glyph) => [glyph.id, glyph]));
  const glyphIndex = buildGlyphIndex(glyphs);
  const evidenceIndex = buildSourceEvidenceIndex(characters, glyphs);
  const summary: UnihanAuditSummary = {
    scanned: 0,
    hasNonG: 0,
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

    const hasExistingNonGData =
      character !== undefined &&
      character.glyphs.length > 1 &&
      character.glyphs.some((entry) =>
        entry.sources.some((source) => source !== "G"),
      );

    if (!character) {
      status = "unsupported";
      reason = "当前 hanzi-chai 数据中没有这个字符。";
    } else if (hasExistingNonGData) {
      status = "existing-non-g-data";
      reason = "已存在多个字形和人工整理的非 G 来源数据，自动流程跳过。";
    } else if (!expectedSet.has("G") || !existingSet.has("G")) {
      status = "needs-review";
      reason = "缺少可作为参考的 G 来源字形。";
    } else if (
      character.glyphs.length !== 1 ||
      !character.glyphs[0]!.sources.includes("G")
    ) {
      status = "needs-review";
      reason = "当前字符不是含 G 来源的单一字形。";
    } else if (glyphsById.get(character.glyphs[0]!.id)?.type !== "compound") {
      status = "insufficient-evidence";
      reason = "G 字形不是可按 component 推断的复合体。";
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
      );
      proposals = recommendation.proposals;
      if (
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
          reason = `每个目标来源的每个 component 都有至少 ${minimumEvidence} 个独立字符的证据，且第一名至少是第二名的 ${minimumDominance} 倍。`;
        } else {
          status = "already-complete";
          reason = "来源已完整，且可靠证据表明非 G 来源与当前字形相同。";
        }
      } else {
        status = "insufficient-evidence";
        reason = `以下来源没有覆盖全部 component 的可靠证据：${recommendation.unresolvedSources.join(", ") || "未知"}。`;
      }
    }

    if (status === "already-complete") summary.alreadyComplete++;
    if (status === "safe-candidate") summary.safeCandidates++;
    if (status === "existing-non-g-data") summary.skippedExistingNonG++;
    if (status === "insufficient-evidence") summary.insufficientEvidence++;
    if (status === "unsupported") summary.unsupported++;
    if (
      status === "needs-review" ||
      status === "insufficient-evidence" ||
      status === "unsupported"
    ) {
      summary.needsReview++;
    }
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
