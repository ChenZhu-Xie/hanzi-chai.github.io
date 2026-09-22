import type { ReviewedSourceDecision } from "./index";

/**
 * Source-specific component decisions checked by a maintainer against the
 * Unicode 18.0 U4E00 chart. Keep these at the smallest differing subtree: a
 * decision about 粦's lower-right leaf can then support every structurally
 * compatible parent without declaring an entire IRG source globally equal.
 */
export const REVIEWED_SOURCE_DECISIONS: ReviewedSourceDecision[] = [
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
].map((decision) => ({ ...decision, provenance: "manual" }));
