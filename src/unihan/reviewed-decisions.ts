import type { ReviewedSourceDecision } from "./index";

/**
 * Source-specific component decisions checked by a maintainer against the
 * Unicode 18.0 U4E00 chart. Keep these at the smallest differing subtree: a
 * decision about 粦's lower-right leaf can then support every structurally
 * compatible parent without declaring an entire IRG source globally equal.
 */
export const REVIEWED_SOURCE_DECISIONS: ReviewedSourceDecision[] = [
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
].map((decision) => ({ ...decision, provenance: "manual" }));
