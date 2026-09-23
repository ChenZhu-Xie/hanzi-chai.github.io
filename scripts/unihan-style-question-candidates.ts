import { parseArgs } from "node:util";
import type { 基本字形数据, 矢量笔画数据 } from "hanzi-chai";
import { 图形盒子 } from "hanzi-chai";
import {
  glyphLeafStrokeIds,
  glyphToSvgMarkup,
} from "../src/components/glyph-svg";

const { values } = parseArgs({
  options: {
    input: { type: "string" },
    output: { type: "string" },
    glyphs: { type: "string" },
    definitions: { type: "string" },
  },
});
if (!values.input || !values.output) {
  throw new Error("--input and --output are required");
}

const payload = (await Bun.file(values.input).json()) as {
  rows: Array<{
    candidates: Record<string, 矢量笔画数据[]>;
    candidateSvgs: Record<string, string>;
    candidateFocusSvgs?: Record<string, string>;
    candidateTopologySvgs?: Record<string, string>;
    candidateFocusIds?: Record<string, number[]>;
    candidateFocusTopology?: Record<
      string,
      Array<{
        id: number;
        leafIds: number[];
        strokeFeatures: string[];
        hasVerticalStroke: boolean;
        hasFallingStroke: boolean;
        verticalLeafIds: number[];
        fallingLeafIds: number[];
        hasSeparateVerticalAndFallingLeaves: boolean;
      }>
    >;
  }>;
};
const definitions = values.definitions
  ? ((await Bun.file(values.definitions).json()) as {
      syntheticGlyphs?: 基本字形数据[];
      siblingFamilies?: number[][];
    })
  : {};
const baseGlyphs = values.glyphs
  ? ((await Bun.file(values.glyphs).json()) as 基本字形数据[])
  : [];
const glyphById = new Map(
  [...baseGlyphs, ...(definitions.syntheticGlyphs ?? [])].map((glyph) => [
    glyph.id,
    glyph,
  ]),
);
const palette = [
  "#2563eb",
  "#16a34a",
  "#ea580c",
  "#7c3aed",
  "#0891b2",
  "#a16207",
];
const familyById = new Map<number, string>();
const colorByFamily = new Map<string, string>();
for (const [index, family] of (definitions.siblingFamilies ?? []).entries()) {
  const key = family
    .slice()
    .sort((a, b) => a - b)
    .join("/");
  for (const id of family) familyById.set(id, key);
  colorByFamily.set(key, palette[index % palette.length]!);
}
const colorFor = (id: number) => {
  const family = familyById.get(id) ?? `${id}`;
  let color = colorByFamily.get(family);
  if (!color) {
    color = palette[colorByFamily.size % palette.length]!;
    colorByFamily.set(family, color);
  }
  return color;
};
const familyDescendants = (id: number, seen = new Set<number>()): number[] => {
  if (seen.has(id)) return [];
  const nextSeen = new Set(seen).add(id);
  const glyph = glyphById.get(id);
  return [
    ...(familyById.has(id) ? [id] : []),
    ...(glyph?.type === "compound"
      ? glyph.references.flatMap((reference) =>
          familyDescendants(reference.id, nextSeen),
        )
      : []),
  ];
};

for (const row of payload.rows) {
  row.candidateFocusIds = Object.fromEntries(
    Object.keys(row.candidates).map((id) => [
      id,
      [...new Set(familyDescendants(Number(id)))],
    ]),
  );
  row.candidateFocusTopology = Object.fromEntries(
    Object.keys(row.candidates).map((id) => [
      id,
      familyDescendants(Number(id)).map((focusId) => {
        const leafIds = [...new Set(glyphLeafStrokeIds(focusId, glyphById))];
        const leafFeatures = new Map(
          leafIds.map((leafId) => {
            const leaf = glyphById.get(leafId);
            return [
              leafId,
              leaf?.type === "component"
                ? leaf.strokes.map((stroke) => stroke.feature)
                : [],
            ] as const;
          }),
        );
        const strokeFeatures = leafIds.flatMap(
          (leafId) => leafFeatures.get(leafId) ?? [],
        );
        const verticalLeafIds = leafIds.filter((leafId) =>
          leafFeatures.get(leafId)?.includes("竖"),
        );
        const fallingLeafIds = leafIds.filter((leafId) =>
          leafFeatures.get(leafId)?.includes("撇"),
        );
        return {
          id: focusId,
          leafIds,
          strokeFeatures,
          hasVerticalStroke: strokeFeatures.includes("竖"),
          hasFallingStroke: strokeFeatures.includes("撇"),
          verticalLeafIds,
          fallingLeafIds,
          hasSeparateVerticalAndFallingLeaves: verticalLeafIds.some(
            (verticalId) =>
              fallingLeafIds.some((fallingId) => fallingId !== verticalId),
          ),
        };
      }),
    ]),
  );
  row.candidateSvgs = Object.fromEntries(
    Object.entries(row.candidates).map(([id, strokes]) => [
      id,
      glyphToSvgMarkup(图形盒子.从笔画列表构建(strokes), false, {
        strokeWidthScale: 0.5,
        showStrokePoints: true,
        strokeColors:
          glyphById.size > 0
            ? glyphLeafStrokeIds(Number(id), glyphById).map(colorFor)
            : undefined,
      }),
    ]),
  );
  row.candidateFocusSvgs = Object.fromEntries(
    Object.entries(row.candidates).map(([id, strokes]) => {
      const glyphId = Number(id);
      const targetLeaves = new Set(
        familyDescendants(glyphId).flatMap((focusId) =>
          glyphLeafStrokeIds(focusId, glyphById),
        ),
      );
      return [
        id,
        glyphToSvgMarkup(图形盒子.从笔画列表构建(strokes), false, {
          strokeWidthScale: 0.5,
          showStrokePoints: true,
          strokeColors: glyphLeafStrokeIds(glyphId, glyphById).map((leafId) =>
            targetLeaves.has(leafId) ? "#2563eb" : "black",
          ),
        }),
      ];
    }),
  );
  row.candidateTopologySvgs = Object.fromEntries(
    Object.entries(row.candidates).map(([id, strokes]) => {
      const glyphId = Number(id);
      const targetLeaves = new Set(
        familyDescendants(glyphId).flatMap((focusId) =>
          glyphLeafStrokeIds(focusId, glyphById),
        ),
      );
      return [
        id,
        glyphToSvgMarkup(图形盒子.从笔画列表构建(strokes), false, {
          strokeWidthScale: 0.5,
          showStrokePoints: false,
          // Remove non-target paths from the SVG DOM. A transparent SVG path
          // can retain black RGB under alpha and reappear when rasterized.
          strokeColors: glyphLeafStrokeIds(glyphId, glyphById).map(
            () => "#2563eb",
          ),
          strokeVisibility: glyphLeafStrokeIds(glyphId, glyphById).map(
            (leafId) => targetLeaves.has(leafId),
          ),
        }),
      ];
    }),
  );
}

await Bun.write(values.output, `${JSON.stringify(payload, null, 2)}\n`);
