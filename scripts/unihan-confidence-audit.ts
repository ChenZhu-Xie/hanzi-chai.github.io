import { parseArgs } from "node:util";
import {
  glyphShapeKey,
  type UnihanAudit,
  type UnihanAuditItem,
} from "../src/unihan";

interface PdfRow {
  sameEdges: string[];
  differentEdges?: string[];
}

interface PdfEvidence {
  metadata: unknown;
  rows: Record<string, PdfRow>;
}

const { values } = parseArgs({
  options: {
    audit: { type: "string" },
    pdf: { type: "string" },
    output: { type: "string" },
  },
});
if (!values.audit || !values.pdf || !values.output) {
  throw new Error("--audit, --pdf and --output are required");
}

const audit = (await Bun.file(values.audit).json()) as UnihanAudit;
const pdf = (await Bun.file(values.pdf).json()) as PdfEvidence;

function pair(left: string, right: string) {
  return [left, right].sort().join("/");
}

function sourceGroups(item: UnihanAuditItem): Map<string, string> {
  const result = new Map<string, string>();
  for (const entry of item.existingGlyphs) {
    if (entry.sources.includes("G")) result.set("G", `id:${entry.id}`);
  }
  for (const proposal of item.proposals) {
    result.set(
      proposal.source,
      proposal.existingGlyphId === undefined
        ? `shape:${glyphShapeKey(proposal.glyph)}`
        : `id:${proposal.existingGlyphId}`,
    );
  }
  return result;
}

function connected(members: string[], edges: Set<string>): boolean {
  if (members.length < 2) return true;
  const visited = new Set([members[0]!]);
  while (true) {
    const before = visited.size;
    for (const left of members) {
      for (const right of members) {
        if (visited.has(left) && edges.has(pair(left, right))) {
          visited.add(right);
        }
      }
    }
    if (visited.size === before) return visited.size === members.length;
  }
}

const rows = audit.items
  .filter((item) => item.status === "safe-candidate")
  .map((item) => {
    const sourceToGroup = sourceGroups(item);
    const row = pdf.rows[item.codepoint];
    const sameEdges = new Set(row?.sameEdges ?? []);
    const differentEdges = new Set(row?.differentEdges ?? []);
    const groups = new Map<string, string[]>();
    for (const source of item.expectedSources) {
      const group = sourceToGroup.get(source);
      if (!group) continue;
      const members = groups.get(group) ?? [];
      members.push(source);
      groups.set(group, members);
    }
    const predictedGroups = [...groups.values()];
    const withinGroupsConfirmed = predictedGroups.every((members) =>
      connected(members, sameEdges),
    );
    let betweenGroupsConfirmed = true;
    for (let left = 0; left < predictedGroups.length; left++) {
      for (let right = left + 1; right < predictedGroups.length; right++) {
        if (
          !predictedGroups[left]!.some((leftSource) =>
            predictedGroups[right]!.some((rightSource) =>
              differentEdges.has(pair(leftSource, rightSource)),
            ),
          )
        ) {
          betweenGroupsConfirmed = false;
        }
      }
    }
    const contradictions = [
      ...sameEdges,
    ].filter((edge) => {
      const [left, right] = edge.split("/");
      return sourceToGroup.get(left!) !== sourceToGroup.get(right!);
    });
    contradictions.push(
      ...[...differentEdges].filter((edge) => {
        const [left, right] = edge.split("/");
        return sourceToGroup.get(left!) === sourceToGroup.get(right!);
      }),
    );
    return {
      codepoint: item.codepoint,
      character: item.character,
      predictedGroups,
      withinGroupsConfirmed,
      betweenGroupsConfirmed,
      fullyDoubleConfirmed:
        Boolean(row) &&
        withinGroupsConfirmed &&
        betweenGroupsConfirmed &&
        contradictions.length === 0,
      contradictions,
      sameEdges: [...sameEdges],
      differentEdges: [...differentEdges],
    };
  });

const summary = {
  safeCandidates: rows.length,
  withPdfRow: rows.filter((row) => row.sameEdges.length > 0 || row.differentEdges.length > 0).length,
  withinGroupsConfirmed: rows.filter((row) => row.withinGroupsConfirmed).length,
  betweenGroupsConfirmed: rows.filter((row) => row.betweenGroupsConfirmed).length,
  fullyDoubleConfirmed: rows.filter((row) => row.fullyDoubleConfirmed).length,
  contradictions: rows.filter((row) => row.contradictions.length > 0).length,
};
const payload = {
  policy:
    "Auto-confirm only when reviewed-range decomposition evidence and calibrated PDF same/different edges cover every proposed source group. Missing image evidence is not treated as agreement.",
  summary,
  rows,
};
await Bun.write(values.output, `${JSON.stringify(payload, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
