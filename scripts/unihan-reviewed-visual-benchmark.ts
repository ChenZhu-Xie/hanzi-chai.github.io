import { parseArgs } from "node:util";
import type { 基本字形数据, 字符数据 } from "hanzi-chai";
import { 字形库 } from "hanzi-chai";
import { glyphToSvgMarkup } from "../src/components/glyph-svg";
import {
  augmentSourceEvidenceWithReviews,
  buildGlyphEvidenceIndex,
  buildSourceEvidenceIndex,
  glyphShapeKey,
  parseUnihanIRGSources,
  recommendMissingSources,
} from "../src/unihan";
import { REVIEWED_SOURCE_DECISIONS } from "../src/unihan/reviewed-decisions";

const { values } = parseArgs({
  options: {
    characters: { type: "string" },
    glyphs: { type: "string" },
    unihan: { type: "string" },
    output: { type: "string" },
  },
});
if (!values.characters || !values.glyphs || !values.unihan || !values.output) {
  throw new Error("--characters, --glyphs, --unihan and --output are required");
}

const characters = (await Bun.file(values.characters).json()) as 字符数据[];
const glyphs = (await Bun.file(values.glyphs).json()) as 基本字形数据[];
const expectedByUnicode = parseUnihanIRGSources(
  await Bun.file(values.unihan).text(),
);
const charactersByUnicode = new Map(
  characters.map((character) => [character.unicode, character]),
);
const glyphById = new Map(glyphs.map((glyph) => [glyph.id, glyph]));
const shapeToId = new Map(
  glyphs.map((glyph) => [glyphShapeKey(glyph), glyph.id]),
);
const evidence = buildSourceEvidenceIndex(characters, glyphs);
augmentSourceEvidenceWithReviews(evidence, REVIEWED_SOURCE_DECISIONS, glyphs);
const glyphEvidence = buildGlyphEvidenceIndex(characters, glyphs);

const parent = new Map<number, number>();
const find = (id: number): number => {
  const current = parent.get(id) ?? id;
  if (current === id) {
    parent.set(id, id);
    return id;
  }
  const root = find(current);
  parent.set(id, root);
  return root;
};
const union = (left: number, right: number) => {
  const leftRoot = find(left);
  const rightRoot = find(right);
  if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
};
for (const decision of REVIEWED_SOURCE_DECISIONS) {
  if (decision.referenceId !== decision.replacementId) {
    union(decision.referenceId, decision.replacementId);
  }
}
const familyByRoot = new Map<number, number[]>();
for (const id of parent.keys()) {
  const root = find(id);
  const family = familyByRoot.get(root) ?? [];
  family.push(id);
  familyByRoot.set(root, family);
}
for (const family of familyByRoot.values()) family.sort((a, b) => a - b);
const familyKey = (id: number) =>
  familyByRoot.get(find(id))?.join("/") ?? `${id}`;

// These visually difficult families are held out in full. No character using
// one of them can leak into calibration through another source or parent.
const heldOutSeeds = new Set([504, 1104, 4929, 5384, 5644, 5925]);
const heldOutFamilies = new Set([...heldOutSeeds].map(familyKey));
const reviewedByUnicode = Map.groupBy(
  REVIEWED_SOURCE_DECISIONS,
  ({ unicode }) => unicode,
);
const syntheticGlyphs: 基本字形数据[] = [];
let nextSyntheticId = 900_000;
const rows: {
  unicode: number;
  split: "train" | "test";
  familyKeys: string[];
  sourceGlyphs: Record<string, number>;
}[] = [];

for (const [unicode, decisions] of reviewedByUnicode) {
  const character = charactersByUnicode.get(unicode);
  const expectedSources = expectedByUnicode.get(unicode);
  const gEntry = character?.glyphs.find(({ sources }) => sources.includes("G"));
  if (!character || !expectedSources || !gEntry) continue;
  const reviewedSources = [
    ...new Set(
      decisions.map(({ source }) => source).filter((source) => source !== "G"),
    ),
  ];
  const recommendation = recommendMissingSources(
    character,
    reviewedSources,
    glyphs,
    evidence,
    3,
    2,
    undefined,
    glyphEvidence,
    REVIEWED_SOURCE_DECISIONS,
  );
  const sourceGlyphs: Record<string, number> = { G: gEntry.id };
  for (const proposal of recommendation.proposals) {
    const shape = glyphShapeKey(proposal.glyph);
    let id = proposal.existingGlyphId ?? shapeToId.get(shape);
    if (id === undefined) {
      id = nextSyntheticId++;
      const synthetic = { ...proposal.glyph, id } as 基本字形数据;
      syntheticGlyphs.push(synthetic);
      glyphById.set(id, synthetic);
      shapeToId.set(shape, id);
    }
    sourceGlyphs[proposal.source] = id;
  }
  const candidateIds = [...new Set(Object.values(sourceGlyphs))];
  if (candidateIds.length < 2) continue;
  const familyKeys = [
    ...new Set(
      decisions
        .filter(
          ({ referenceId, replacementId }) => referenceId !== replacementId,
        )
        .map(({ referenceId }) => familyKey(referenceId)),
    ),
  ].sort();
  if (familyKeys.length === 0) continue;
  rows.push({
    unicode,
    split: familyKeys.some((key) => heldOutFamilies.has(key))
      ? "test"
      : "train",
    familyKeys,
    sourceGlyphs,
  });
}

const library = new 字形库([...glyphs, ...syntheticGlyphs]);
const renderedRows = rows.flatMap((row) => {
  const candidateIds = [...new Set(Object.values(row.sourceGlyphs))];
  const rendered = candidateIds.flatMap((id) => {
    const glyph = library.获取字形(id);
    return glyph ? [[id, glyph] as const] : [];
  });
  if (rendered.length !== candidateIds.length) return [];
  return [
    {
      ...row,
      candidates: Object.fromEntries(
        rendered.map(([id, glyph]) => [id, glyph.图形盒子.获取笔画列表()]),
      ),
      candidateSvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [id, glyphToSvgMarkup(glyph.图形盒子)]),
      ),
    },
  ];
});

const payload = {
  metadata: {
    provenance: "maintainer-reviewed Unicode 18.0 U4E00 decisions",
    splitStrategy: "whole sibling-family holdout",
    heldOutFamilies: [...heldOutFamilies].sort(),
    trainRows: renderedRows.filter(({ split }) => split === "train").length,
    testRows: renderedRows.filter(({ split }) => split === "test").length,
  },
  rows: renderedRows,
};
await Bun.write(values.output, `${JSON.stringify(payload, null, 2)}\n`);
console.log(JSON.stringify(payload.metadata, null, 2));
