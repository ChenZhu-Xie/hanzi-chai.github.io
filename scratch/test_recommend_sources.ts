import {
  augmentSourceEvidenceWithReviews,
  buildGlyphEvidenceIndex,
  buildSourceEvidenceIndex,
  recommendMissingSources,
} from "../src/unihan";
import { REVIEWED_SOURCE_DECISIONS } from "../src/unihan/reviewed-decisions";
import type { 字符数据, 基本字形数据 } from "hanzi-chai";

const characters = (await Bun.file(".local/latest-characters.json").json()) as 字符数据[];
const glyphs = (await Bun.file(".local/latest-glyphs.json").json()) as 基本字形数据[];

const evidence = buildSourceEvidenceIndex(characters, glyphs);
augmentSourceEvidenceWithReviews(evidence, REVIEWED_SOURCE_DECISIONS, glyphs);
const glyphEvidence = buildGlyphEvidenceIndex(characters, glyphs);

const testCases = [
  { unicode: 0x6962, source: "J", name: "楢" },
  { unicode: 0x6478, source: "H", name: "摸" },
  { unicode: 0x7ca6, source: "J", name: "粦" },
  { unicode: 0x7740, source: "J", name: "着" },
];

for (const tc of testCases) {
  const character = characters.find((c) => c.unicode === tc.unicode);
  if (!character) {
    console.log(`Character ${tc.unicode} not found`);
    continue;
  }
  const rec = recommendMissingSources(
    character,
    [tc.source],
    glyphs,
    evidence,
    3,
    2,
    undefined,
    glyphEvidence,
    REVIEWED_SOURCE_DECISIONS,
  );
  console.log(`\n=================== ${tc.name} (0x${tc.unicode.toString(16)}) ${tc.source} ===================`);
  console.log("Base glyphs:", character.glyphs);
  console.log("Proposals count:", rec.proposals.length);
  for (const p of rec.proposals) {
    console.log(`  Source: ${p.source}, existingGlyphId: ${p.existingGlyphId}`);
    console.log("  Glyph:", JSON.stringify(p.glyph));
  }
}
