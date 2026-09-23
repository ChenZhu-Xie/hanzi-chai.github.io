import { parseArgs } from "node:util";
import { type 基本字形数据, 字形库, type 字符数据 } from "hanzi-chai";
import { glyphToSvgMarkup } from "../src/components/glyph-svg";
import { isRecommendationSample } from "../src/unihan";

const { values } = parseArgs({
  options: {
    characters: { type: "string" },
    glyphs: { type: "string" },
    output: { type: "string" },
  },
});
if (!values.characters || !values.glyphs || !values.output) {
  throw new Error("--characters, --glyphs and --output are required");
}

const characters = (await Bun.file(values.characters).json()) as 字符数据[];
const glyphs = (await Bun.file(values.glyphs).json()) as 基本字形数据[];
const library = new 字形库(glyphs);
const rows = [];

for (const character of characters) {
  if (!isRecommendationSample(character.unicode)) continue;
  const ids = [...new Set(character.glyphs.map(({ id }) => id))];
  if (ids.length < 2) continue;
  const candidates = Object.fromEntries(
    ids.flatMap((id) => {
      const glyph = library.获取字形(id);
      return glyph ? [[id, glyph.图形盒子.获取笔画列表()] as const] : [];
    }),
  );
  if (Object.keys(candidates).length !== ids.length) continue;
  const candidateSvgs = Object.fromEntries(
    ids.flatMap((id) => {
      const glyph = library.获取字形(id);
      return glyph ? [[id, glyphToSvgMarkup(glyph.图形盒子)] as const] : [];
    }),
  );
  rows.push({
    unicode: character.unicode,
    sourceGlyphs: Object.fromEntries(
      character.glyphs.flatMap(({ id, sources }) =>
        sources.map((source) => [source, id]),
      ),
    ),
    candidates,
    candidateSvgs,
  });
}

await Bun.write(values.output, `${JSON.stringify({ rows }, null, 2)}\n`);
console.log(`exported ${rows.length} reviewed multi-glyph characters`);
