import { parseArgs } from "node:util";
import type { 基本字形数据, 字符数据 } from "hanzi-chai";
import {
  auditUnihanSources,
  parseUnihanIRGSources,
  type ReviewedSourceDecision,
} from "../src/unihan";
import { REVIEWED_SOURCE_DECISIONS } from "../src/unihan/reviewed-decisions";

const { values } = parseArgs({
  options: {
    characters: { type: "string" },
    glyphs: { type: "string" },
    unihan: { type: "string" },
    decisions: { type: "string" },
    output: { type: "string" },
  },
});
if (!values.characters || !values.glyphs || !values.unihan || !values.output) {
  throw new Error(
    "--characters, --glyphs, --unihan and --output are required",
  );
}

const characters = (await Bun.file(values.characters).json()) as 字符数据[];
const glyphs = (await Bun.file(values.glyphs).json()) as 基本字形数据[];
const unihan = parseUnihanIRGSources(await Bun.file(values.unihan).text());
const reviewedDecisions = values.decisions
  ? ((await Bun.file(values.decisions).json()) as ReviewedSourceDecision[])
  : REVIEWED_SOURCE_DECISIONS;
const audit = auditUnihanSources(unihan, characters, glyphs, {
  reviewedDecisions,
});
await Bun.write(values.output, `${JSON.stringify(audit, null, 2)}\n`);
console.log(JSON.stringify(audit.summary, null, 2));
