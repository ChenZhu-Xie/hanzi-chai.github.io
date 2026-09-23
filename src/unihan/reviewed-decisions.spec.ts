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

test("splits U+6752 刃 into J 395 and T 397", () => {
  const glyphs: 基本字形数据[] = [448, 395, 397].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 18722,
    type: "compound",
    operator: "⿰",
    references: [{ id: 448 }, { id: 395 }],
    ambiguous: false,
  });
  const evidence = new Map(
    ["T", "J"].map((source) => [
      source,
      new Map([[448, new Map([[448, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x6752,
      glyphs: [{ id: 18722, sources: ["G", "T", "J"] }],
    },
    ["T", "J"],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    T: { existingGlyphId: undefined, references: [448, 397] },
    J: { existingGlyphId: 18722, references: [448, 395] },
  });
});

test("splits U+671E 月 into vertical G H N and left-falling T J K U", () => {
  const glyphs: 基本字形数据[] = [4276, 504, 569].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 18683,
    type: "compound",
    operator: "⿱",
    references: [{ id: 4276 }, { id: 504 }],
    ambiguous: false,
  });
  const sources = ["H", "T", "J", "K", "N", "U"];
  const evidence = new Map(
    sources.map((source) => [
      source,
      new Map([[4276, new Map([[4276, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x671e,
      glyphs: [{ id: 18683, sources: ["G", ...sources] }],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    H: { existingGlyphId: 18683, references: [4276, 504] },
    T: { existingGlyphId: undefined, references: [4276, 569] },
    J: { existingGlyphId: undefined, references: [4276, 569] },
    K: { existingGlyphId: undefined, references: [4276, 569] },
    N: { existingGlyphId: 18683, references: [4276, 504] },
    U: { existingGlyphId: undefined, references: [4276, 569] },
  });
});

test("splits U+6485 厥 into KP 127248 and all other sources 5384", () => {
  const components: 基本字形数据[] = [220, 71, 5385, 127247].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  const glyphs: 基本字形数据[] = [
    ...components,
    {
      id: 5384,
      type: "compound",
      operator: "⿸",
      references: [{ id: 71 }, { id: 5385 }],
      ambiguous: false,
    },
    {
      id: 127248,
      type: "compound",
      operator: "⿸",
      references: [{ id: 71 }, { id: 127247 }],
      ambiguous: false,
    },
    {
      id: 18088,
      type: "compound",
      operator: "⿰",
      references: [{ id: 220 }, { id: 5384 }],
      ambiguous: false,
    },
  ];
  const sources = ["H", "T", "J", "K", "N", "V"];
  const evidence = new Map(
    sources.map((source) => [
      source,
      new Map([[220, new Map([[220, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x6485,
      glyphs: [{ id: 18088, sources: ["G", ...sources] }],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    H: { existingGlyphId: 18088, references: [220, 5384] },
    T: { existingGlyphId: 18088, references: [220, 5384] },
    J: { existingGlyphId: 18088, references: [220, 5384] },
    K: { existingGlyphId: 18088, references: [220, 5384] },
    N: { existingGlyphId: undefined, references: [220, 127248] },
    V: { existingGlyphId: 18088, references: [220, 5384] },
  });
});

test("splits U+808E 月 into vertical 504, left-falling 569, and dotted 579", () => {
  const glyphs: 基本字形数据[] = [155, 504, 569, 579].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 24813,
    type: "compound",
    operator: "⿱",
    references: [{ id: 155 }, { id: 504 }],
    ambiguous: false,
  });
  const sources = ["T", "J", "K", "N"];
  const evidence = new Map(
    sources.map((source) => [
      source,
      new Map([[155, new Map([[155, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x808e,
      glyphs: [{ id: 24813, sources: ["G", ...sources] }],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    T: { existingGlyphId: undefined, references: [155, 579] },
    J: { existingGlyphId: undefined, references: [155, 569] },
    K: { existingGlyphId: undefined, references: [155, 569] },
    N: { existingGlyphId: 24813, references: [155, 504] },
  });
});

test("splits U+7527 月 into vertical G J and left-falling T", () => {
  const glyphs: 基本字形数据[] = [8688, 504, 569].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 22146,
    type: "compound",
    operator: "⿱",
    references: [{ id: 8688 }, { id: 504 }],
    ambiguous: false,
  });
  const evidence = new Map(
    ["T", "J"].map((source) => [
      source,
      new Map([[8688, new Map([[8688, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x7527,
      glyphs: [{ id: 22146, sources: ["G", "T", "J"] }],
    },
    ["T", "J"],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    T: { existingGlyphId: undefined, references: [8688, 569] },
    J: { existingGlyphId: 22146, references: [8688, 504] },
  });
});

test("splits U+67ED right component into KP 4929 and H T J K 4930", () => {
  const glyphs: 基本字形数据[] = [448, 4929, 4930].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 18856,
    type: "compound",
    operator: "⿰",
    references: [{ id: 448 }, { id: 4929 }],
    ambiguous: false,
  });
  const sources = ["H", "T", "J", "K", "N"];
  const evidence = new Map(
    sources.map((source) => [
      source,
      new Map([[448, new Map([[448, new Set([1, 2, 3])]])]]),
    ]),
  );
  const result = recommendMissingSources(
    {
      unicode: 0x67ed,
      glyphs: [{ id: 18856, sources: ["G", ...sources] }],
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
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    H: { existingGlyphId: undefined, references: [448, 4930] },
    T: { existingGlyphId: undefined, references: [448, 4930] },
    J: { existingGlyphId: undefined, references: [448, 4930] },
    K: { existingGlyphId: undefined, references: [448, 4930] },
    N: { existingGlyphId: 18856, references: [448, 4929] },
  });
});

test("uses grass sibling 486 in H/T 擎 and keeps 228 in K", () => {
  const component = (id: number): 基本字形数据 => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  });
  const glyphs: 基本字形数据[] = [228, 486, 4274, 545, 529].map(component);
  glyphs.push(
    {
      id: 6275,
      type: "compound",
      operator: "⿱",
      references: [{ id: 228 }, { id: 4274 }],
      ambiguous: false,
    },
    {
      id: 25376,
      type: "compound",
      operator: "⿱",
      references: [{ id: 486 }, { id: 4274 }],
      ambiguous: false,
    },
    {
      id: 6274,
      type: "compound",
      operator: "⿰",
      references: [{ id: 6275 }, { id: 545 }],
      ambiguous: false,
    },
    {
      id: 127089,
      type: "compound",
      operator: "⿰",
      references: [{ id: 25376 }, { id: 545 }],
      ambiguous: false,
    },
    {
      id: 18168,
      type: "compound",
      operator: "⿱",
      references: [{ id: 6274 }, { id: 529 }],
      ambiguous: false,
    },
  );

  const result = recommendMissingSources(
    {
      unicode: 0x64ce,
      glyphs: [{ id: 18168, sources: ["G", "H", "T", "K"] }],
    },
    ["H", "T", "K"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(result.proposals).toHaveLength(3);
  expect(
    Object.fromEntries(
      result.proposals.map(({ source, existingGlyphId, glyph }) => [
        source,
        {
          existingGlyphId,
          references:
            glyph.type === "compound"
              ? glyph.references.map(({ id }) => id)
              : [],
        },
      ]),
    ),
  ).toEqual({
    H: { existingGlyphId: undefined, references: [127089, 529] },
    T: { existingGlyphId: undefined, references: [127089, 529] },
    K: { existingGlyphId: 18168, references: [6274, 529] },
  });
});

test("propagates the reviewed H-source 敬 sibling into U+64CF 擏", () => {
  const glyphs: 基本字形数据[] = [220, 6274, 127089].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 18169,
    type: "compound",
    operator: "⿰",
    references: [{ id: 220 }, { id: 6274 }],
    ambiguous: false,
  });

  const result = recommendMissingSources(
    { unicode: 0x64cf, glyphs: [{ id: 18169, sources: ["G", "H"] }] },
    ["H"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual([]);
  expect(result.proposals[0]?.glyph).toMatchObject({
    type: "compound",
    operator: "⿰",
    references: [{ id: 220 }, { id: 127089 }],
  });
});

test("keeps the reviewed J-source grass top in U+82C6 苆 on 228 without assuming its lower component", () => {
  const glyphs: 基本字形数据[] = [228, 437, 486, 8920].map((id) => ({
    id,
    type: "component",
    strokes: [],
    operator: undefined,
    references: undefined,
    ambiguous: false,
  }));
  glyphs.push({
    id: 25313,
    type: "compound",
    operator: "⿱",
    references: [{ id: 228 }, { id: 8920 }],
    ambiguous: false,
  });

  const result = recommendMissingSources(
    { unicode: 0x82c6, glyphs: [{ id: 25313, sources: ["G", "J"] }] },
    ["J"],
    glyphs,
    new Map(),
    3,
    2,
    undefined,
    undefined,
    REVIEWED_SOURCE_DECISIONS,
  );

  expect(result.unresolvedSources).toEqual(["J"]);
  expect(result.proposals).toEqual([]);
  expect(result.unresolved[0]).toMatchObject({
    source: "J",
    evidence: [
      {
        referenceId: 228,
        replacementId: 228,
        reliable: true,
        reviewed: true,
      },
      {
        referenceId: 8920,
        reliable: false,
      },
    ],
  });
});
