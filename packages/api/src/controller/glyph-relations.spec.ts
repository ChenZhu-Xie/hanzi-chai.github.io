import { describe, expect, test } from "bun:test";
import { Err } from "../error/error";
import { normalizeGlyphRelation } from "./glyph-relations";

describe("normalizeGlyphRelation", () => {
  test("canonicalizes a symmetric relationship and deduplicates sources", () => {
    expect(
      normalizeGlyphRelation({
        leftId: 20,
        rightId: 10,
        kind: "manual",
        status: "confirmed",
        provenance: "manual",
        sources: ["T", "G", "T"],
        evidence: [],
      }),
    ).toEqual({
      leftId: 10,
      rightId: 20,
      kind: "manual",
      status: "confirmed",
      provenance: "manual",
      sources: ["G", "T"],
      evidence: [],
    });
  });

  test("rejects a self relationship and unknown enum values", () => {
    const self = normalizeGlyphRelation({
      leftId: 10,
      rightId: 10,
      kind: "manual",
      status: "confirmed",
      provenance: "manual",
      sources: [],
      evidence: [],
    });
    const unknown = normalizeGlyphRelation({
      leftId: 10,
      rightId: 20,
      kind: "same-everywhere",
      status: "confirmed",
      provenance: "manual",
      sources: [],
      evidence: [],
    });
    expect(self).toBeInstanceOf(Err);
    expect(unknown).toBeInstanceOf(Err);
  });

  test("rejects malformed evidence instead of persisting opaque JSON", () => {
    expect(
      normalizeGlyphRelation({
        leftId: 10,
        rightId: 20,
        kind: "visual-sibling",
        status: "candidate",
        provenance: "algorithm",
        sources: ["J"],
        evidence: [{ unicode: 0x4e00, source: "J" }],
      }),
    ).toBeInstanceOf(Err);
  });
});
