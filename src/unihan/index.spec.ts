import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { type 基本字形数据, type 字符数据, 来源排序 } from "hanzi-chai";
import {
  applySourceAssignments,
  auditUnihanSources,
  augmentSourceEvidenceWithReviews,
  buildGlyphEvidenceIndex,
  buildReviewedGlyphSiblingIndex,
  buildSourceEvidenceIndex,
  IRG_PROPERTY_TO_SOURCE,
  parseUnihanIRGSources,
  readUnihanVersion,
  recommendMissingSources,
  resolveUnresolvedSourceProposal,
  sortSources,
} from "./index";

describe("parseUnihanIRGSources", () => {
  test("parses one source", () => {
    const parsed = parseUnihanIRGSources("U+4E00\tkIRG_GSource\tG0-523B");
    expect(parsed.get(0x4e00)).toEqual(["G"]);
  });

  test("parses G + T", () => {
    const parsed = parseUnihanIRGSources(
      "U+4E00\tkIRG_TSource\tT1-4421\nU+4E00\tkIRG_GSource\tG0-523B",
    );
    expect(parsed.get(0x4e00)).toEqual(["G", "T"]);
  });

  test("uses repository source order for G + H + T + J + K", () => {
    const parsed = parseUnihanIRGSources(
      ["K", "J", "T", "H", "G"]
        .map((source) => `U+4E00\tkIRG_${source}Source\tvalue`)
        .join("\n"),
    );
    expect(parsed.get(0x4e00)).toEqual(["G", "H", "T", "J", "K"]);
  });

  test("maps KP to N and UK to B", () => {
    const parsed = parseUnihanIRGSources(
      "U+4E00\tkIRG_KPSource\tKP0-FCD6\nU+4E00\tkIRG_UKSource\tUK-0001",
    );
    expect(parsed.get(0x4e00)).toEqual(["N", "B"]);
  });

  test("reads and validates the release version metadata", () => {
    expect(readUnihanVersion("# Unicode Version 18.0.0\n")).toBe("18.0.0");
    expect(readUnihanVersion("# no version here\n")).toBeUndefined();
    expect(
      [...IRG_PROPERTY_TO_SOURCE.values()].every((source) =>
        来源排序.includes(source),
      ),
    ).toBe(true);
  });

  test("deduplicates multiple references for one property", () => {
    const parsed = parseUnihanIRGSources(
      "U+4E00\tkIRG_GSource\tG0-523B G1-0001\nU+4E00\tkIRG_GSource\tG2-0001",
    );
    expect(parsed.get(0x4e00)).toEqual(["G"]);
  });

  test("ignores unknown properties, out-of-range codepoints, malformed lines, and comments", () => {
    const parsed = parseUnihanIRGSources(
      [
        "# comment",
        "not a valid line",
        "U+4E00\tkDefinition\tone",
        "U+3400\tkIRG_GSource\tG0-0000",
        "U+4E00\tkIRG_GSource\tG0-523B",
      ].join("\n"),
    );
    expect([...parsed]).toEqual([[0x4e00, ["G"]]]);
  });
});

describe("applySourceAssignments", () => {
  test("is idempotent and moves a source to its proposed glyph", () => {
    const character: 字符数据 = {
      unicode: 0x4e00,
      glyphs: [{ id: 1, sources: ["G", "T", "J"] }],
    };
    const once = applySourceAssignments(character, [
      { source: "T", id: 1 },
      { source: "J", id: 2 },
    ]);
    const twice = applySourceAssignments(once, [
      { source: "T", id: 1 },
      { source: "J", id: 2 },
    ]);
    expect(once).toEqual({
      unicode: 0x4e00,
      glyphs: [
        { id: 1, sources: ["G", "T"] },
        { id: 2, sources: ["J"] },
      ],
    });
    expect(twice).toEqual(once);
    expect(sortSources(["B", "KP", "G", "T"])).toEqual(["G", "T", "B", "KP"]);
  });
});

describe("recommendation safety", () => {
  const glyphs: 基本字形数据[] = [
    {
      id: 1,
      type: "component",
      strokes: [],
      operator: undefined,
      references: undefined,
      ambiguous: false,
    },
    {
      id: 2,
      type: "component",
      strokes: [],
      operator: undefined,
      references: undefined,
      ambiguous: false,
    },
    {
      id: 3,
      type: "component",
      strokes: [],
      operator: undefined,
      references: undefined,
      ambiguous: false,
    },
    {
      id: 100,
      type: "compound",
      operator: "⿰",
      references: [{ id: 1 }, { id: 3 }],
      ambiguous: false,
    },
    {
      id: 101,
      type: "compound",
      operator: "⿰",
      references: [{ id: 2 }, { id: 3 }],
      ambiguous: false,
    },
    {
      id: 102,
      type: "compound",
      operator: "⿰",
      references: [
        { id: 1, xbegin: 10 },
        { id: 3, ybegin: 4 },
      ],
      ambiguous: false,
    },
  ];
  const reviewedSamples: 字符数据[] = [0x4e10, 0x4e11, 0x4e12].map(
    (unicode) => ({
      unicode,
      glyphs: [
        { id: 100, sources: ["G"] },
        { id: 101, sources: ["T"] },
      ],
    }),
  );
  const candidate: 字符数据 = {
    unicode: 0x6401,
    glyphs: [{ id: 102, sources: ["G", "T"] }],
  };

  test("uses only reviewed multi-glyph samples and preserves reference geometry", () => {
    const evidence = buildSourceEvidenceIndex(
      [...reviewedSamples, candidate],
      glyphs,
    );
    const result = recommendMissingSources(candidate, ["T"], glyphs, evidence);
    expect(result.unresolvedSources).toEqual([]);
    expect(result.proposals[0]?.requiresChange).toBe(true);
    expect(result.proposals[0]?.glyph.references).toEqual([
      { id: 2, xbegin: 10 },
      { id: 3, ybegin: 4 },
    ]);
  });

  test("does not treat an unreviewed single-glyph character as evidence", () => {
    const evidence = buildSourceEvidenceIndex([candidate], glyphs);
    const result = recommendMissingSources(candidate, ["T"], glyphs, evidence);
    expect(result.proposals).toEqual([]);
    expect(result.unresolvedSources).toEqual(["T"]);
  });

  test("turns a maintainer choice into a revalidated complete proposal", () => {
    const unresolved = {
      source: "T",
      evidence: [
        {
          referenceId: 1,
          replacementId: 1,
          count: 4,
          reliable: false,
          alternatives: [
            { id: 1, count: 4, examples: [0x4e10] },
            { id: 2, count: 4, examples: [0x4e11] },
          ],
        },
        {
          referenceId: 3,
          replacementId: 3,
          count: 5,
          reliable: true,
          alternatives: [{ id: 3, count: 5, examples: [0x4e12] }],
        },
      ],
    };
    const proposal = resolveUnresolvedSourceProposal(
      candidate,
      glyphs,
      unresolved,
      new Map([[1, 2]]),
    );
    expect(proposal?.source).toBe("T");
    expect(proposal?.glyph.references).toEqual([
      { id: 2, xbegin: 10 },
      { id: 3, ybegin: 4 },
    ]);
    expect(
      resolveUnresolvedSourceProposal(
        candidate,
        glyphs,
        unresolved,
        new Map([[1, 999]]),
      ),
    ).toBeUndefined();
  });

  test("propagates reviewed decisions to leaves and resolves nested references", () => {
    const nestedGlyphs: 基本字形数据[] = [
      ...glyphs,
      {
        id: 4,
        type: "component",
        strokes: [],
        operator: undefined,
        references: undefined,
        ambiguous: false,
      },
      {
        id: 10,
        type: "compound",
        operator: "⿰",
        references: [{ id: 1 }, { id: 2 }],
        ambiguous: false,
      },
      {
        id: 11,
        type: "compound",
        operator: "⿰",
        references: [{ id: 1 }, { id: 3 }],
        ambiguous: false,
      },
      {
        id: 200,
        type: "compound",
        operator: "⿱",
        references: [{ id: 10 }, { id: 4 }],
        ambiguous: false,
      },
    ];
    const decisions = [0x4e50, 0x4e51, 0x4e52].flatMap((unicode) => [
      { unicode, source: "T", referenceId: 1, replacementId: 1 },
      { unicode, source: "T", referenceId: 2, replacementId: 3 },
      { unicode, source: "T", referenceId: 4, replacementId: 4 },
    ]);
    const evidence = augmentSourceEvidenceWithReviews(
      new Map(),
      decisions,
      nestedGlyphs,
    );
    const nestedCandidate: 字符数据 = {
      unicode: 0x6403,
      glyphs: [{ id: 200, sources: ["G", "T"] }],
    };
    const result = recommendMissingSources(
      nestedCandidate,
      ["T"],
      nestedGlyphs,
      evidence,
    );
    expect(result.unresolvedSources).toEqual([]);
    expect(result.proposals[0]?.glyph.references).toEqual([
      { id: 11 },
      { id: 4 },
    ]);
    expect(result.proposals[0]?.evidence).toHaveLength(2);
    expect(result.proposals[0]?.evidence[0]).toMatchObject({
      referenceId: 10,
      replacementId: 11,
      reliable: true,
    });
    expect(result.proposals[0]?.evidence[0]?.basis).toBeDefined();

    const structuralEvidence = augmentSourceEvidenceWithReviews(
      new Map(),
      [
        {
          unicode: 0x4e60,
          source: "T",
          referenceId: 10,
          replacementId: 11,
        },
      ],
      nestedGlyphs,
    );
    expect(structuralEvidence.get("T")?.get(2)?.get(3)).toEqual(
      new Set([0x4e60]),
    );
  });

  test("uses reviewed same-glyph source assignments as identity evidence", () => {
    const reviewedIdentitySamples: 字符数据[] = [0x4e30, 0x4e31, 0x4e32].map(
      (unicode) => ({
        unicode,
        glyphs: [{ id: 100, sources: ["G", "T"] }],
      }),
    );
    const evidence = buildSourceEvidenceIndex(reviewedIdentitySamples, glyphs);
    const result = recommendMissingSources(candidate, ["T"], glyphs, evidence);
    expect(result.unresolvedSources).toEqual([]);
    expect(result.proposals[0]?.requiresChange).toBe(false);
    expect(result.proposals[0]?.evidence.map((item) => item.count)).toEqual([
      3, 3,
    ]);
  });

  test("recommends a reviewed sibling for an indecomposable component", () => {
    const reviewedComponents: 字符数据[] = [0x4e40, 0x4e41, 0x4e42].map(
      (unicode) => ({
        unicode,
        glyphs: [
          { id: 1, sources: ["G"] },
          { id: 2, sources: ["T"] },
        ],
      }),
    );
    const leafCandidate: 字符数据 = {
      unicode: 0x6402,
      glyphs: [{ id: 1, sources: ["G", "T"] }],
    };
    const result = recommendMissingSources(
      leafCandidate,
      ["T"],
      glyphs,
      buildSourceEvidenceIndex(reviewedComponents, glyphs),
      3,
      2,
      undefined,
      buildGlyphEvidenceIndex(reviewedComponents, glyphs),
    );
    expect(result.unresolvedSources).toEqual([]);
    expect(result.proposals[0]?.existingGlyphId).toBe(2);
    expect(result.proposals[0]?.requiresChange).toBe(true);
  });

  test("derives sibling candidates only from reviewed source variants", () => {
    const index = buildReviewedGlyphSiblingIndex(
      [...reviewedSamples, candidate],
      glyphs,
    );
    expect(index.get(1)?.[0]).toMatchObject({
      id: 2,
      count: 3,
      sources: ["T"],
    });
    expect(index.get(2)?.[0]).toMatchObject({ id: 1, count: 3 });
  });

  test("counts repeated component replacements once per source character", () => {
    const repeatedGlyphs: 基本字形数据[] = [
      ...glyphs,
      {
        id: 103,
        type: "compound",
        operator: "⿰",
        references: [{ id: 1 }, { id: 1 }],
        ambiguous: false,
      },
      {
        id: 104,
        type: "compound",
        operator: "⿰",
        references: [{ id: 2 }, { id: 2 }],
        ambiguous: false,
      },
    ];
    const evidence = buildSourceEvidenceIndex(
      [
        {
          unicode: 0x4e20,
          glyphs: [
            { id: 103, sources: ["G"] },
            { id: 104, sources: ["T"] },
          ],
        },
      ],
      repeatedGlyphs,
    );
    const candidateWithRepeatedParts: 字符数据 = {
      unicode: 0x4e21,
      glyphs: [{ id: 103, sources: ["G", "T"] }],
    };
    const result = recommendMissingSources(
      candidateWithRepeatedParts,
      ["T"],
      repeatedGlyphs,
      evidence,
      2,
    );
    expect(result.unresolvedSources).toEqual(["T"]);
  });

  test("ignores evidence whose source glyph has a different structure", () => {
    const incompatibleGlyphs: 基本字形数据[] = [
      ...glyphs,
      {
        id: 105,
        type: "compound",
        operator: "⿱",
        references: [{ id: 2 }, { id: 3 }],
        ambiguous: false,
      },
    ];
    const evidence = buildSourceEvidenceIndex(
      [
        {
          unicode: 0x4e22,
          glyphs: [
            { id: 100, sources: ["G"] },
            { id: 105, sources: ["T"] },
          ],
        },
      ],
      incompatibleGlyphs,
    );
    expect(evidence.get("T")).toBeUndefined();
  });

  test("marks a candidate safe but skips existing multi-glyph data", () => {
    const sources = new Map([
      [candidate.unicode, ["G", "T"]],
      [reviewedSamples[0]!.unicode, ["G", "T"]],
    ]);
    const audit = auditUnihanSources(
      sources,
      [...reviewedSamples, candidate],
      glyphs,
    );
    expect(
      audit.items.find((item) => item.unicode === candidate.unicode)?.status,
    ).toBe("safe-candidate");
    expect(
      audit.items.find((item) => item.unicode === reviewedSamples[0]!.unicode)
        ?.status,
    ).toBe("reviewed-reference");
    expect(
      audit.summary.reviewedReferences +
        audit.summary.alreadyComplete +
        audit.summary.safeCandidates +
        audit.summary.needsReview +
        audit.summary.skippedExistingNonG +
        audit.summary.insufficientEvidence +
        audit.summary.unsupported,
    ).toBe(audit.summary.scanned);
  });
});

const realUnihanPath = ".local/unicode/Unihan_IRGSources.txt";
(existsSync(realUnihanPath) ? test : test.skip)(
  "parses Unicode 18.0 real data",
  () => {
    const parsed = parseUnihanIRGSources(readFileSync(realUnihanPath, "utf8"));
    expect(parsed.get(0x4e00)).toEqual(["G", "H", "T", "J", "K", "N", "V"]);
    expect(parsed.get(0x9fff)).toEqual(["G", "M", "U"]);
    expect(parsed.size).toBeGreaterThan(20_000);
  },
);
