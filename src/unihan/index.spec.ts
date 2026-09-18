import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { type 基本字形数据, type 字符数据, 来源排序 } from "hanzi-chai";
import {
  applySourceAssignments,
  auditUnihanSources,
  buildSourceEvidenceIndex,
  IRG_PROPERTY_TO_SOURCE,
  parseUnihanIRGSources,
  readUnihanVersion,
  recommendMissingSources,
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
    unicode: 0x4e13,
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
    ).toBe("existing-non-g-data");
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
