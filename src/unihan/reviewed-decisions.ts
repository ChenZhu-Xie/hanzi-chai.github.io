import type { ReviewedSourceDecision } from "./index";

/**
 * Confirmed source forms whose required sibling component does not exist in
 * the repository yet. They are intentionally excluded from automatic apply
 * until the new component receives a real database ID.
 */
export const REVIEWED_MISSING_SIBLING_DECISIONS = [
  {
    unicode: 0x9aa8,
    sources: ["H", "T", "J", "K", "N"],
    referenceId: 752,
    topology: "inner-horizontal-right-of-middle-vertical",
  },
  {
    unicode: 0x9aa8,
    sources: ["V"],
    referenceId: 752,
    topology: "inner-horizontal-left-of-middle-vertical",
  },
] as const;

/**
 * Source-specific component decisions checked by a maintainer against the
 * Unicode 18.0 U4E00 chart. Keep these at the smallest differing subtree: a
 * decision about 粦's lower-right leaf can then support every structurally
 * compatible parent without declaring an entire IRG source globally equal.
 */
export const REVIEWED_SOURCE_DECISIONS: ReviewedSourceDecision[] = [
  // U+64CE 擎: H uses 敬 127089, whose smallest differing subtree is grass
  // sibling 486 rather than G's 228. The lower 手 remains 529.
  ...(["H", "T"] as const).flatMap((source) =>
    ([
      [6274, 127089],
      [529, 529],
    ] as const).map(([referenceId, replacementId]) => ({
      unicode: 0x64ce,
      source,
      referenceId,
      replacementId,
    })),
  ),
  ...([
    [6274, 6274],
    [529, 529],
  ] as const).map(([referenceId, replacementId]) => ({
    unicode: 0x64ce,
    source: "K" as const,
    referenceId,
    replacementId,
  })),

  // U+64CF 擏: H independently validates propagation from U+64CE. The left
  // 手 stays 220 while the right 敬 changes from 6274 to 127089.
  ...([
    [220, 220],
    [6274, 127089],
  ] as const).map(([referenceId, replacementId]) => ({
    unicode: 0x64cf,
    source: "H" as const,
    referenceId,
    replacementId,
  })),

  // U+7CA6 粦: G/H/T/KP(N) keep 4161 (leaf 255); J/K use 126982 (leaf 438).
  { unicode: 0x7ca6, source: "H", referenceId: 4161, replacementId: 4161 },
  { unicode: 0x7ca6, source: "T", referenceId: 4161, replacementId: 4161 },
  {
    unicode: 0x7ca6,
    source: "J",
    referenceId: 4161,
    replacementId: 126982,
  },
  {
    unicode: 0x7ca6,
    source: "K",
    referenceId: 4161,
    replacementId: 126982,
  },
  { unicode: 0x7ca6, source: "N", referenceId: 4161, replacementId: 4161 },

  // U+65B4 斴: G/T use 4726 (ultimately leaf 255); J/K use 127075 (leaf 438).
  { unicode: 0x65b4, source: "T", referenceId: 4726, replacementId: 4726 },
  {
    unicode: 0x65b4,
    source: "J",
    referenceId: 4726,
    replacementId: 127075,
  },
  {
    unicode: 0x65b4,
    source: "K",
    referenceId: 4726,
    replacementId: 127075,
  },

  // U+649B 撛: G/T/V use 4726; J/K use 127075.
  { unicode: 0x649b, source: "T", referenceId: 4726, replacementId: 4726 },
  {
    unicode: 0x649b,
    source: "J",
    referenceId: 4726,
    replacementId: 127075,
  },
  {
    unicode: 0x649b,
    source: "K",
    referenceId: 4726,
    replacementId: 127075,
  },
  { unicode: 0x649b, source: "V", referenceId: 4726, replacementId: 4726 },

  // U+660E 明: V keeps the same 月 component (569) as the other forms.
  { unicode: 0x660e, source: "V", referenceId: 569, replacementId: 569 },

  // U+6BA6 殦: H keeps the same 鳥 component (4199) as the other forms.
  { unicode: 0x6ba6, source: "H", referenceId: 4199, replacementId: 4199 },

  // U+6770 杰: H keeps the same bottom-four-dots component (605).
  { unicode: 0x6770, source: "H", referenceId: 605, replacementId: 605 },

  // U+6B3D 欽: the chart's final rising stroke is a conditional rendering
  // detail. The current model keeps every source on the existing 金 (4174).
  ...(["H", "T", "J", "K", "N", "V"] as const).map((source) => ({
    unicode: 0x6b3d,
    source,
    referenceId: 4174,
    replacementId: 4174,
  })),

  // U+6F76 潶: source-font differences in 三点水 and the four bottom dots
  // do not create siblings while their stroke categories stay unchanged.
  ...(["H", "T", "J", "K", "N"] as const).flatMap((source) => [
    { unicode: 0x6f76, source, referenceId: 350, replacementId: 350 },
    { unicode: 0x6f76, source, referenceId: 605, replacementId: 605 },
  ]),
  // Every source's 黑 was visually confirmed as 4213 = ⿱(1057, 605).
  ...(["H", "T", "J", "K", "N"] as const).map((source) => ({
    unicode: 0x6f76,
    source,
    referenceId: 4213,
    replacementId: 4213,
  })),

  // U+65E6 旦: every source keeps the semantic 日 component (506), not the
  // visually similar compound 冒字头 (4098).
  ...(["H", "T", "J", "K", "N", "V"] as const).map((source) => ({
    unicode: 0x65e6,
    source,
    referenceId: 506,
    replacementId: 506,
  })),

  // High-yield identity decisions visually confirmed by the maintainer.
  // U+6A22 樢: G/T/J/K/KP(N) all keep 鳥 (4199).
  ...(["T", "J", "K", "N"] as const).map((source) => ({
    unicode: 0x6a22,
    source,
    referenceId: 4199,
    replacementId: 4199,
  })),
  { unicode: 0x6536, source: "V", referenceId: 545, replacementId: 545 },
  { unicode: 0x6599, source: "V", referenceId: 933, replacementId: 933 },
  { unicode: 0x887e, source: "N", referenceId: 921, replacementId: 921 },
  { unicode: 0x73e4, source: "K", referenceId: 890, replacementId: 890 },

  // U+6637 昷: G/T/J/K/KP(N) all keep the bottom 皿 component (744).
  ...(["T", "J", "K", "N"] as const).map((source) => ({
    unicode: 0x6637,
    source,
    referenceId: 744,
    replacementId: 744,
  })),

  // U+6903 椃: H uses 虎 126882 = ⿸(866, 128), not G's 4109.
  {
    unicode: 0x6903,
    source: "H",
    referenceId: 4109,
    replacementId: 126882,
  },

  // U+6424 搤: G/H/T use 益 5644; J/K/KP(N) use 127533.
  ...(["H", "T"] as const).map((source) => ({
    unicode: 0x6424,
    source,
    referenceId: 5644,
    replacementId: 5644,
  })),
  ...(["J", "K", "N"] as const).map((source) => ({
    unicode: 0x6424,
    source,
    referenceId: 5644,
    replacementId: 127533,
  })),

  // U+6635 昵: T uses 尼 58234, whose 匕 starts with a horizontal stroke.
  {
    unicode: 0x6635,
    source: "T",
    referenceId: 5925,
    replacementId: 58234,
  },

  // U+65E8 旨: H/T use 匕 1128; G/J/K/KP(N)/V keep 133.
  ...(["H", "T"] as const).map((source) => ({
    unicode: 0x65e8,
    source,
    referenceId: 133,
    replacementId: 1128,
  })),
  ...(["J", "K", "N", "V"] as const).map((source) => ({
    unicode: 0x65e8,
    source,
    referenceId: 133,
    replacementId: 133,
  })),

  // U+657C 敼: all six chart forms keep 喜 5766, whose middle subtree uses
  // component 345 (点 + 撇 + 横), including KP(N); do not infer 127051.
  ...(["H", "T", "J", "K", "N"] as const).map((source) => ({
    unicode: 0x657c,
    source,
    referenceId: 5766,
    replacementId: 5766,
  })),

  // U+6EED 滭: the three reviewed 畢 variants are separated by stroke
  // continuity, the split/continuous middle horizontal, and the relative
  // length of the final two horizontals. G/J/K/KP(N) use 1104, H uses 1186,
  // and T uses 4646.
  {
    unicode: 0x6eed,
    source: "H",
    referenceId: 1104,
    replacementId: 1186,
  },
  {
    unicode: 0x6eed,
    source: "T",
    referenceId: 1104,
    replacementId: 4646,
  },
  ...(["J", "K", "N"] as const).map((source) => ({
    unicode: 0x6eed,
    source,
    referenceId: 1104,
    replacementId: 1104,
  })),

  // U+69B4 榴: every source keeps 留 6821 = ⿱(6822, 727); equivalently,
  // every 留 has the reviewed top 6822 rather than 53980's top 5127.
  ...(["H", "T", "J", "K", "N", "V"] as const).flatMap((source) => [
    {
      unicode: 0x69b4,
      source,
      referenceId: 6822,
      replacementId: 6822,
    },
    {
      unicode: 0x69b4,
      source,
      referenceId: 727,
      replacementId: 727,
    },
  ]),

  // U+6962 楢: G/H/T/V use the integral 酋 component 1099; J/K/KP(N)
  // use 126948 = ⿱(117, 977), with an independent 八 above 酉.
  ...(["H", "T", "V"] as const).map((source) => ({
    unicode: 0x6962,
    source,
    referenceId: 1099,
    replacementId: 1099,
  })),
  ...(["J", "K", "N"] as const).map((source) => ({
    unicode: 0x6962,
    source,
    referenceId: 1099,
    replacementId: 126948,
  })),

  // U+6E9C 溜: all six sources use the same decomposition ⿰(350, 6821).
  // Source-font details in 三点水 do not create siblings, and every 留 is
  // the reviewed 6821 = ⿱(6822, 727).
  ...(["H", "T", "J", "K", "N"] as const).flatMap((source) => [
    {
      unicode: 0x6e9c,
      source,
      referenceId: 350,
      replacementId: 350,
    },
    {
      unicode: 0x6e9c,
      source,
      referenceId: 6821,
      replacementId: 6821,
    },
  ]),

  // U+6C15 氕: G/H/T all use ⿹(536, 5). T keeps the curved 横斜钩
  // 气 component 536 rather than the right-angled 横折弯钩 variant 1166.
  ...(["H", "T"] as const).flatMap((source) => [
    {
      unicode: 0x6c15,
      source,
      referenceId: 536,
      replacementId: 536,
    },
    {
      unicode: 0x6c15,
      source,
      referenceId: 5,
      replacementId: 5,
    },
  ]),

  // U+6709 有: the lower 月 is 569 (a left-falling first stroke) only in T.
  // G/H/J/K/KP(N)/V keep component 504 with a vertical first stroke.
  ...(["H", "J", "K", "N", "V"] as const).map((source) => ({
    unicode: 0x6709,
    source,
    referenceId: 504,
    replacementId: 504,
  })),
  {
    unicode: 0x6709,
    source: "T",
    referenceId: 504,
    replacementId: 569,
  },

  // These J-source 月 forms keep 569 with two inner horizontal strokes. They
  // are not the two-dot sibling 126894 used by some compressed parents.
  ...([0x6714, 0x6715, 0x6717, 0x6ed5, 0x80ba, 0x9a30] as const).map(
    (unicode) => ({
      unicode,
      source: "J" as const,
      referenceId: 569,
      replacementId: 569,
    }),
  ),

  // U+6752 杒: J keeps G's 刃 component 395, whose third stroke falls down
  // and left; T uses sibling 397, whose third stroke extends down and right.
  {
    unicode: 0x6752,
    source: "J",
    referenceId: 395,
    replacementId: 395,
  },
  {
    unicode: 0x6752,
    source: "T",
    referenceId: 395,
    replacementId: 397,
  },

  // U+671E 朞: G/H/KP(N) use 月 504 with a vertical first stroke;
  // T/J/K/U use sibling 569 with a left-falling first stroke.
  ...(["H", "N"] as const).map((source) => ({
    unicode: 0x671e,
    source,
    referenceId: 504,
    replacementId: 504,
  })),
  ...(["T", "J", "K", "U"] as const).map((source) => ({
    unicode: 0x671e,
    source,
    referenceId: 504,
    replacementId: 569,
  })),

  // U+6485 撅: KP(N) uses 厥 127248, whose inner left child contains the
  // outward-opening 八 117. G/H/T/J/K/V use 5384 with inner child 934.
  ...(["H", "T", "J", "K", "V"] as const).map((source) => ({
    unicode: 0x6485,
    source,
    referenceId: 5384,
    replacementId: 5384,
  })),
  {
    unicode: 0x6485,
    source: "N",
    referenceId: 5384,
    replacementId: 127248,
  },

  // U+808E 肎: G/KP(N) use 月 504 (竖 first stroke); J/K use 569 (撇).
  // T uses the existing third sibling 579 = 撇、横折钩、点、横, where the
  // PDF's inner 点/捺 and 提 are covered by 点=捺 and 横=提.
  {
    unicode: 0x808e,
    source: "N",
    referenceId: 504,
    replacementId: 504,
  },
  ...(["J", "K"] as const).map((source) => ({
    unicode: 0x808e,
    source,
    referenceId: 504,
    replacementId: 569,
  })),
  {
    unicode: 0x808e,
    source: "T",
    referenceId: 504,
    replacementId: 579,
  },

  // U+9AA8 骨: T uses 月 sibling 579 (撇 + inner 点/捺 + 横). Its upper
  // component is reviewed separately because the inner folded horizontal is
  // topologically reversed from component 752.
  {
    unicode: 0x9aa8,
    source: "T",
    referenceId: 504,
    replacementId: 579,
  },

  // U+7527 甧: G/J use 月 504 with a vertical first stroke; T uses 569
  // with a left-falling first stroke.
  {
    unicode: 0x7527,
    source: "J",
    referenceId: 504,
    replacementId: 504,
  },
  {
    unicode: 0x7527,
    source: "T",
    referenceId: 504,
    replacementId: 569,
  },

  // U+67ED 柭: G/KP(N) use 4929 = ⿸(247, 188), whose inner first stroke
  // starts with a short horizontal segment. H/T/J/K use 4930 = ⿸(247, 118)
  // with a separate 撇 and no horizontal lead-in.
  {
    unicode: 0x67ed,
    source: "N",
    referenceId: 4929,
    replacementId: 4929,
  },
  ...(["H", "T", "J", "K"] as const).map((source) => ({
    unicode: 0x67ed,
    source,
    referenceId: 4929,
    replacementId: 4930,
  })),

  // U+82C6 苆: the J-source grass top has one horizontal crossing both
  // verticals, so it keeps component 228 rather than siblings 437 or 486.
  {
    unicode: 0x82c6,
    source: "J",
    referenceId: 228,
    replacementId: 228,
  },

  // U+659B 斛: H/T/J/K use 角 sibling 4110, whose inner vertical stops at the
  // lower horizontal; 1019's inner vertical continues below it.
  ...(["H", "T", "J", "K"] as const).map((source) => ({
    unicode: 0x659b,
    source,
    referenceId: 1019,
    replacementId: 4110,
  })),

  // U+6690 暐: J uses 韋 126927, whose bottom is 438. Its bottom horizontal
  // protrudes left of the left vertical; 4179 -> 255 starts flush there.
  {
    unicode: 0x6690,
    source: "J",
    referenceId: 4179,
    replacementId: 126927,
  },

  // U+7740 着: the J-source upper component is 62022 = ⿱(929, 5), not
  // component 928. On its last horizontal the central vertical and the
  // separate left-falling stroke cross at two horizontally offset points;
  // the vertical crossing is close to a right angle. The overall curved
  // outline is deliberately not evidence because both candidates have one.
  {
    unicode: 0x7740,
    source: "J",
    referenceId: 928,
    replacementId: 62022,
  },

  // U+72D1 狑: H/T use 令 sibling 775 (横 + 横撇/点), J/K use 774
  // (横 + 横折钩/竖), and KP(N) keeps 779 (点 + 横撇/点).
  ...(["H", "T"] as const).map((source) => ({
    unicode: 0x72d1,
    source,
    referenceId: 779,
    replacementId: 775,
  })),
  ...(["J", "K"] as const).map((source) => ({
    unicode: 0x72d1,
    source,
    referenceId: 779,
    replacementId: 774,
  })),
  {
    unicode: 0x72d1,
    source: "N",
    referenceId: 779,
    replacementId: 779,
  },

  // 蒙 family: the reviewed J forms of 曚/朦/檬 keep 6054, while the two
  // reviewed KP(N) forms use 48986. Component 1035 inside 6054 remains atomic:
  // rewriting it as ⿱(1,982) changes repository layout geometry.
  ...([0x66da, 0x6726, 0x6aac] as const).map((unicode) => ({
    unicode,
    source: "J" as const,
    referenceId: 6054,
    replacementId: 6054,
  })),
  ...([0x66da, 0x6726] as const).map((unicode) => ({
    unicode,
    source: "N" as const,
    referenceId: 6054,
    replacementId: 48986,
  })),

  // Recursive topology review batch. These decisions were made at the lowest
  // differing subtree, not from whole-glyph optical similarity:
  // - 搘/擬: 匕's first stroke is 横 (1128), not 平撇 (133).
  // - 摡: the J form keeps the complete 白-over-匕 subtree in 46403.
  // - 摸: J keeps grass sibling 228 inside 7695.
  // - 敚: T keeps the 倒八 component 1028, not 正八-over-737 (13645).
  { unicode: 0x6418, source: "T", referenceId: 9313, replacementId: 127551 },
  { unicode: 0x6461, source: "J", referenceId: 7368, replacementId: 46403 },
  { unicode: 0x6478, source: "J", referenceId: 7695, replacementId: 7695 },
  { unicode: 0x64ec, source: "T", referenceId: 5201, replacementId: 128254 },
  { unicode: 0x655a, source: "T", referenceId: 1028, replacementId: 1028 },
].map((decision) => ({ ...decision, provenance: "manual" }));
