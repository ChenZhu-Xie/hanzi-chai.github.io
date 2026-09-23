import { expect, test } from "bun:test";
import type { 基本字形数据, 字符数据 } from "hanzi-chai";
import { recommendMissingSources } from "./index";
import { REVIEWED_SOURCE_DECISIONS } from "./reviewed-decisions";

test("resolves the reviewed T-source 尼 variant for U+6635", () => {
  const glyphs: 基本字形数据[] = [5925, 58234].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const result = recommendMissingSources(
    { unicode: 0x6635, glyphs: [{ id: 5925, sources: ["G", "T"] }] },
    ["T"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(result.proposals[0]).toMatchObject({
    source: "T",
    existingGlyphId: 58234,
    requiresChange: true,
  });
});

test("resolves the reviewed U+65E8 匕 source groups", () => {
  const glyphs: 基本字形数据[] = [133, 1128].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const character: 字符数据 = {
    unicode: 0x65e8,
    glyphs: [{ id: 133, sources: ["G", "H", "T", "J", "K", "N", "V"] }],
  };
  const result = recommendMissingSources(
    character,
    ["H", "T", "J", "K", "N", "V"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 1128, T: 1128, J: 133, K: 133, N: 133, V: 133 });
});

test("keeps every U+657C source on 喜 5766", () => {
  const glyphs: 基本字形数据[] = [5766, 127051].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const result = recommendMissingSources(
    {
      unicode: 0x657c,
      glyphs: [{ id: 5766, sources: ["G", "H", "T", "J", "K", "N"] }],
    },
    ["H", "T", "J", "K", "N"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 5766, T: 5766, J: 5766, K: 5766, N: 5766 });
});

test("resolves the three reviewed U+6EED 畢 source groups", () => {
  const glyphs: 基本字形数据[] = [1104, 1186, 4646].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const result = recommendMissingSources(
    {
      unicode: 0x6eed,
      glyphs: [{ id: 1104, sources: ["G", "H", "T", "J", "K", "N"] }],
    },
    ["H", "T", "J", "K", "N"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 1186, T: 4646, J: 1104, K: 1104, N: 1104 });
});

test("keeps every U+69B4 source on 留 6821 with top 6822", () => {
  const glyphs: 基本字形数据[] = [6822, 5127, 727].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push(
    {
      id: 6821,
      type: "compound",
      operator: "⿱",
      references: [{ id: 6822 }, { id: 727 }],
      ambiguous: false,
    },
    {
      id: 53980,
      type: "compound",
      operator: "⿱",
      references: [{ id: 5127 }, { id: 727 }],
      ambiguous: false,
    },
  );
  const result = recommendMissingSources(
    {
      unicode: 0x69b4,
      glyphs: [{ id: 6821, sources: ["G", "H", "T", "J", "K", "N", "V"] }],
    },
    ["H", "T", "J", "K", "N", "V"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 6821, T: 6821, J: 6821, K: 6821, N: 6821, V: 6821 });
});

test("resolves the two reviewed U+6962 酋 source groups", () => {
  const glyphs: 基本字形数据[] = [1099, 126948].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const result = recommendMissingSources(
    {
      unicode: 0x6962,
      glyphs: [{ id: 1099, sources: ["G", "H", "T", "J", "K", "N", "V"] }],
    },
    ["H", "T", "J", "K", "N", "V"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 1099, T: 1099, J: 126948, K: 126948, N: 126948, V: 1099 });
});

test("keeps every U+6E9C source on the same 水 and 留 components", () => {
  const glyphs: 基本字形数据[] = [350, 6821].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 20494,
    type: "compound",
    operator: "⿰",
    references: [{ id: 350 }, { id: 6821 }],
    ambiguous: false,
  });
  const result = recommendMissingSources(
    {
      unicode: 0x6e9c,
      glyphs: [{ id: 20494, sources: ["G", "H", "T", "J", "K", "N"] }],
    },
    ["H", "T", "J", "K", "N"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 20494, T: 20494, J: 20494, K: 20494, N: 20494 });
});

test("keeps all U+6C15 sources on the same 气 and 撇 components", () => {
  const glyphs: 基本字形数据[] = [536, 1166, 5].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 19882,
    type: "compound",
    operator: "⿹",
    references: [{ id: 536 }, { id: 5 }],
    ambiguous: false,
  });
  const result = recommendMissingSources(
    {
      unicode: 0x6c15,
      glyphs: [{ id: 19882, sources: ["G", "H", "T"] }],
    },
    ["H", "T"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({ H: 19882, T: 19882 });
});

test("splits U+6709 月 into the reviewed T and non-T variants", () => {
  const glyphs: 基本字形数据[] = [72, 504, 569].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push(
    {
      id: 6445,
      type: "compound",
      operator: "⿸",
      references: [{ id: 72 }, { id: 504 }],
      ambiguous: false,
    },
    {
      id: 130670,
      type: "compound",
      operator: "⿸",
      references: [{ id: 72 }, { id: 569 }],
      ambiguous: false,
    },
  );
  const sources = ["H", "T", "J", "K", "N", "V"];
  const evidence = new Map(
    sources.map((source) => [
      source,
      new Map([[72, new Map([[72, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x6709,
      glyphs: [{ id: 6445, sources: ["G", ...sources] }],
    },
    sources,
    glyphs,
    evidence,
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId }) => [
        source,
        existingGlyphId,
      ]),
    ),
  ).toEqual({
    H: 6445,
    T: 130670,
    J: 6445,
    K: 6445,
    N: 6445,
    V: 6445,
  });
});
