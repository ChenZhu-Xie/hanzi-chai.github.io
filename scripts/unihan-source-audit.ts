import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { 基本字形数据, 字符数据 } from "hanzi-chai";
import {
  auditUnihanSources,
  parseUnihanIRGSources,
  readUnihanVersion,
  SUPPORTED_UNIHAN_VERSION,
} from "../src/unihan";
import { REVIEWED_SOURCE_DECISIONS } from "../src/unihan/reviewed-decisions";

const [unihanPath, charactersPath, glyphsPath, outputPath] =
  process.argv.slice(2);
if (!unihanPath || !charactersPath || !glyphsPath || !outputPath) {
  console.error(
    "Usage: bun scripts/unihan-source-audit.ts <Unihan_IRGSources.txt> <characters.json> <glyphs.json> <output.json>",
  );
  process.exit(2);
}

const unihanText = readFileSync(unihanPath, "utf8");
const version = readUnihanVersion(unihanText);
if (version !== SUPPORTED_UNIHAN_VERSION) {
  console.error(
    `Expected Unicode ${SUPPORTED_UNIHAN_VERSION}, received ${version ?? "unknown"}`,
  );
  process.exit(2);
}
const sources = parseUnihanIRGSources(unihanText);
const characters = JSON.parse(
  readFileSync(charactersPath, "utf8"),
) as 字符数据[];
const glyphs = JSON.parse(readFileSync(glyphsPath, "utf8")) as 基本字形数据[];
const audit = auditUnihanSources(sources, characters, glyphs, {
  reviewedDecisions: REVIEWED_SOURCE_DECISIONS,
});

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(audit, null, 2)}\n`);
console.log(JSON.stringify(audit.summary, null, 2));
console.log(`Wrote ${outputPath}`);
