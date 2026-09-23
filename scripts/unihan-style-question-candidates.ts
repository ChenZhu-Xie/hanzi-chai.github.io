import { parseArgs } from "node:util";
import type { 矢量笔画数据 } from "hanzi-chai";
import type { 基本字形数据 } from "hanzi-chai";
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
  const key = family.slice().sort((a, b) => a - b).join("/");
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

for (const row of payload.rows) {
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
}

await Bun.write(values.output, `${JSON.stringify(payload, null, 2)}\n`);
