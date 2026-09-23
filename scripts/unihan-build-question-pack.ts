import { parseArgs } from "node:util";
import type { 基本字形数据 } from "hanzi-chai";
import { 字形库 } from "hanzi-chai";
import { glyphToSvgMarkup } from "../src/components/glyph-svg";

const { values } = parseArgs({
  options: {
    glyphs: { type: "string" },
    definitions: { type: "string" },
    unicode: { type: "string" },
    sources: { type: "string" },
    candidates: { type: "string" },
    outputCandidates: { type: "string" },
    outputResults: { type: "string" },
  },
});
const required = [
  "glyphs",
  "definitions",
  "unicode",
  "sources",
  "candidates",
  "outputCandidates",
  "outputResults",
] as const;
for (const key of required) {
  if (!values[key]) throw new Error(`--${key} is required`);
}

const glyphs = (await Bun.file(values.glyphs!).json()) as 基本字形数据[];
const definitions = (await Bun.file(values.definitions!).json()) as {
  syntheticGlyphs?: 基本字形数据[];
};
const allGlyphs = [...glyphs, ...(definitions.syntheticGlyphs ?? [])];
const library = new 字形库(allGlyphs);
const candidateIds = values.candidates!.split(",").map(Number);
const sources = values.sources!.split(",");
const unicode = Number.parseInt(values.unicode!.replace(/^U\+/i, ""), 16);
const rendered = candidateIds.map((id) => {
  const glyph = library.获取字形(id);
  if (!glyph) throw new Error(`candidate glyph ${id} does not exist`);
  return [id, glyph.图形盒子] as const;
});

const candidatePayload = {
  rows: [
    {
      unicode,
      sourceGlyphs: Object.fromEntries(sources.map((source) => [source, candidateIds[0]])),
      candidates: Object.fromEntries(
        rendered.map(([id, box]) => [id, box.获取笔画列表()]),
      ),
      candidateSvgs: Object.fromEntries(
        rendered.map(([id, box]) => [id, glyphToSvgMarkup(box)]),
      ),
    },
  ],
};
const resultPayload = {
  results: sources.map((source) => ({
    unicode,
    source,
    expectedGlyphId: candidateIds[0],
    predictedGlyphId: candidateIds[0],
    correct: true,
    candidates: candidateIds.map((id) => ({
      id,
      distance: 0,
      visualDistance: 0,
      topologyDistance: 0,
    })),
  })),
};

await Bun.write(
  values.outputCandidates!,
  `${JSON.stringify(candidatePayload, null, 2)}\n`,
);
await Bun.write(
  values.outputResults!,
  `${JSON.stringify(resultPayload, null, 2)}\n`,
);
