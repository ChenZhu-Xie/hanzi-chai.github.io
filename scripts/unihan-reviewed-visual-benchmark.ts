import { parseArgs } from "node:util";
import type { 基本字形数据, 字符数据 } from "hanzi-chai";
import { 字形库 } from "hanzi-chai";
import {
  glyphLeafReviewColor,
  glyphLeafStrokeIds,
  glyphToSvgMarkup,
} from "../src/components/glyph-svg";
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

interface HierarchyStrokeOwner {
  leafId: number;
  occurrenceKey: string;
  hierarchyIds: number[];
}

interface HierarchyOccurrence {
  leafId: number;
  occurrence: number;
  strokeIndices: number[];
  hierarchy: {
    id: number;
    type: "component" | "compound" | "glyph";
    label: string;
    familyKey: string;
  }[];
}

const hierarchyNode = (
  id: number,
  rootId: number,
): HierarchyOccurrence["hierarchy"][number] => {
  const glyph = glyphById.get(id);
  if (!glyph) throw new Error(`字形 ${id} 不存在`);
  const type = id === rootId ? "glyph" : glyph.type;
  const label =
    glyph.type === "component" ? (glyph.name ?? "末级部件") : glyph.operator;
  return { id, type, label, familyKey: familyKey(id) };
};

const hierarchyStrokeOwners = (
  id: number,
  rootId: number,
  path: string,
  ancestors: number[] = [],
  seen = new Set<number>(),
): HierarchyStrokeOwner[] => {
  if (seen.has(id)) throw new Error(`字形 ${id} 存在循环引用`);
  const glyph = glyphById.get(id);
  if (!glyph) throw new Error(`字形 ${id} 不存在`);
  const hierarchyIds = [id, ...ancestors];
  if (glyph.type === "component") {
    return glyph.strokes.map(() => ({
      leafId: id,
      occurrenceKey: path,
      hierarchyIds,
    }));
  }
  const nextSeen = new Set(seen).add(id);
  const parts = glyph.references.map(({ id: referenceId }, index) =>
    hierarchyStrokeOwners(
      referenceId,
      rootId,
      `${path}/${index}:${referenceId}`,
      hierarchyIds,
      nextSeen,
    ),
  );
  if (!glyph.strokes?.length) return parts.flat();
  return glyph.strokes.flatMap(({ index, from, to }) => {
    const part = parts[index] ?? [];
    return part.slice(from ?? 0, (to ?? part.length - 1) + 1);
  });
};

const glyphHierarchyOccurrences = (id: number): HierarchyOccurrence[] => {
  const owners = hierarchyStrokeOwners(id, id, `${id}`);
  const byKey = new Map<string, HierarchyOccurrence>();
  const countByLeaf = new Map<number, number>();
  for (const [strokeIndex, owner] of owners.entries()) {
    let occurrence = byKey.get(owner.occurrenceKey);
    if (!occurrence) {
      const occurrenceIndex = countByLeaf.get(owner.leafId) ?? 0;
      countByLeaf.set(owner.leafId, occurrenceIndex + 1);
      occurrence = {
        leafId: owner.leafId,
        occurrence: occurrenceIndex,
        strokeIndices: [],
        hierarchy: owner.hierarchyIds.map((nodeId) =>
          hierarchyNode(nodeId, id),
        ),
      };
      byKey.set(owner.occurrenceKey, occurrence);
    }
    const resolvedOccurrence = occurrence;
    if (!resolvedOccurrence) throw new Error(`字形 ${id} 的层级实例生成失败`);
    resolvedOccurrence.strokeIndices.push(strokeIndex);
  }
  return [...byKey.values()];
};

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
  const leafIds = Object.fromEntries(
    rendered.map(([id]) => [id, glyphLeafStrokeIds(id, glyphById)]),
  );
  const leafSets = Object.values(leafIds).map((ids) => new Set(ids));
  const allLeafIds = new Set(Object.values(leafIds).flat());
  const commonLeafIds = new Set(
    [...allLeafIds].filter((id) => leafSets.every((set) => set.has(id))),
  );
  const focusLeafIds = [...allLeafIds]
    .filter((id) => !commonLeafIds.has(id))
    .sort((left, right) => left - right);
  const focus = new Set(focusLeafIds);
  const colorByFamily = new Map<string, string>();
  const colorFor = (leafId: number) => {
    const key = familyKey(leafId);
    let color = colorByFamily.get(key);
    if (!color) {
      color = glyphLeafReviewColor(colorByFamily.size);
      colorByFamily.set(key, color);
    }
    return color;
  };
  for (const id of Object.values(leafIds).flat()) colorFor(id);
  return [
    {
      ...row,
      focusLeafIds,
      candidateLeafIds: leafIds,
      candidates: Object.fromEntries(
        rendered.map(([id, glyph]) => [id, glyph.图形盒子.获取笔画列表()]),
      ),
      candidateScoringSvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [id, glyphToSvgMarkup(glyph.图形盒子)]),
      ),
      candidateSvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [
          id,
          glyphToSvgMarkup(glyph.图形盒子, false, {
            strokeWidthScale: 0.5,
            showStrokeBoundaryPoints: true,
            strokeColors: leafIds[id]!.map(colorFor),
          }),
        ]),
      ),
      candidateLeafColors: Object.fromEntries(
        [...allLeafIds].map((id) => [id, colorFor(id)]),
      ),
      candidateLeafSvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [
          id,
          glyphHierarchyOccurrences(id).map((leaf) => ({
            ...leaf,
            familyKey: familyKey(leaf.leafId),
            color: colorFor(leaf.leafId),
            svg: glyphToSvgMarkup(glyph.图形盒子, false, {
              strokeWidthScale: 0.5,
              strokeColors: leafIds[id]!.map(() => colorFor(leaf.leafId)),
              strokeVisibility: leafIds[id]!.map((_leafId, strokeIndex) =>
                leaf.strokeIndices.includes(strokeIndex),
              ),
            }),
          })),
        ]),
      ),
      candidateFocusSvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [
          id,
          glyphToSvgMarkup(glyph.图形盒子, false, {
            strokeColors: leafIds[id]!.map((leafId) =>
              focus.has(leafId) ? "#f59e0b" : "black",
            ),
          }),
        ]),
      ),
      candidateTopologySvgs: Object.fromEntries(
        rendered.map(([id, glyph]) => [
          id,
          glyphToSvgMarkup(glyph.图形盒子, false, {
            strokeColors: leafIds[id]!.map(() => "#f59e0b"),
            strokeVisibility: leafIds[id]!.map((leafId) => focus.has(leafId)),
          }),
        ]),
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
