import type { 字形关系数据 } from "hanzi-chai";
import type { IRequest } from "itty-router";
import type { Env } from "../dto/context";
import { Err, ErrCode } from "../error/error";

const table = "glyph_relations";
const kinds = new Set(["source-variant", "visual-sibling", "manual"]);
const statuses = new Set(["candidate", "confirmed", "rejected"]);
const provenances = new Set(["reviewed-range", "manual", "algorithm"]);

interface 字形关系模型 {
  id: number;
  left_id: number;
  right_id: number;
  kind: 字形关系数据["kind"];
  status: 字形关系数据["status"];
  provenance: 字形关系数据["provenance"];
  sources: string;
  evidence: string;
}

function 转数据(row: 字形关系模型): 字形关系数据 {
  return {
    id: row.id,
    leftId: row.left_id,
    rightId: row.right_id,
    kind: row.kind,
    status: row.status,
    provenance: row.provenance,
    sources: JSON.parse(row.sources) as string[],
    evidence: JSON.parse(row.evidence) as 字形关系数据["evidence"],
  };
}

export function normalizeGlyphRelation(value: unknown): 字形关系数据 | Err {
  if (typeof value !== "object" || value === null) {
    return new Err(ErrCode.ParamInvalid, "字形关系格式不正确");
  }
  const input = value as Partial<字形关系数据>;
  const validEvidence =
    Array.isArray(input.evidence) &&
    input.evidence.every(
      (entry) =>
        typeof entry === "object" &&
        entry !== null &&
        Number.isInteger(entry.unicode) &&
        entry.unicode >= 0 &&
        typeof entry.source === "string" &&
        entry.source.length > 0 &&
        Number.isInteger(entry.parentGlyphId) &&
        entry.parentGlyphId >= 0 &&
        Number.isInteger(entry.position) &&
        entry.position >= 0,
    );
  if (
    !Number.isInteger(input.leftId) ||
    !Number.isInteger(input.rightId) ||
    input.leftId! < 0 ||
    input.rightId! < 0 ||
    input.leftId === input.rightId ||
    !input.kind ||
    !kinds.has(input.kind) ||
    !input.status ||
    !statuses.has(input.status) ||
    !input.provenance ||
    !provenances.has(input.provenance) ||
    !Array.isArray(input.sources) ||
    !input.sources.every((source) => typeof source === "string") ||
    !validEvidence
  ) {
    return new Err(ErrCode.ParamInvalid, "字形关系字段不正确");
  }
  const rawLeftId = input.leftId as number;
  const rawRightId = input.rightId as number;
  const [leftId, rightId] =
    rawLeftId < rawRightId ? [rawLeftId, rawRightId] : [rawRightId, rawLeftId];
  return {
    leftId,
    rightId,
    kind: input.kind,
    status: input.status,
    provenance: input.provenance,
    sources: [...new Set(input.sources)].sort(),
    evidence: input.evidence,
  };
}

export async function List(request: IRequest, env: Env) {
  const glyphId = Number.parseInt(
    new URL(request.url).searchParams.get("glyph") ?? "",
    10,
  );
  const query = Number.isInteger(glyphId)
    ? env.CHAI.prepare(
        `SELECT * FROM ${table} WHERE left_id=? OR right_id=? ORDER BY id`,
      ).bind(glyphId, glyphId)
    : env.CHAI.prepare(`SELECT * FROM ${table} ORDER BY id`);
  try {
    const { results } = await query.all<字形关系模型>();
    return results.map(转数据);
  } catch (error) {
    return new Err(
      ErrCode.DataQueryFailed,
      `查询字形关系失败（${(error as Error).message}）`,
    );
  }
}

export async function Create(request: IRequest, env: Env) {
  let body: unknown;
  try {
    body = await request.json();
  } catch (error) {
    return new Err(ErrCode.ParamInvalid, (error as Error).message);
  }
  const relation = normalizeGlyphRelation(body);
  if (relation instanceof Err) return relation;
  const glyphRows = await env.CHAI.prepare(
    "SELECT id FROM glyphs WHERE id IN (?, ?)",
  )
    .bind(relation.leftId, relation.rightId)
    .all<{ id: number }>();
  if (glyphRows.results.length !== 2) {
    return new Err(ErrCode.RecordNotFound, "关系中的字形不存在");
  }
  try {
    await env.CHAI.prepare(
      `INSERT INTO ${table} (left_id, right_id, kind, status, provenance, sources, evidence)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(left_id, right_id, kind) DO UPDATE SET
         status=excluded.status,
         provenance=excluded.provenance,
         sources=excluded.sources,
         evidence=excluded.evidence`,
    )
      .bind(
        relation.leftId,
        relation.rightId,
        relation.kind,
        relation.status,
        relation.provenance,
        JSON.stringify(relation.sources),
        JSON.stringify(relation.evidence),
      )
      .run();
    const row = await env.CHAI.prepare(
      `SELECT * FROM ${table} WHERE left_id=? AND right_id=? AND kind=?`,
    )
      .bind(relation.leftId, relation.rightId, relation.kind)
      .first<字形关系模型>();
    if (!row) return new Err(ErrCode.DataCreateFailed, "创建字形关系失败");
    return 转数据(row);
  } catch (error) {
    return new Err(
      ErrCode.DataCreateFailed,
      `创建字形关系失败（${(error as Error).message}）`,
    );
  }
}

export async function Delete(request: IRequest, env: Env) {
  const id = Number.parseInt(request.params.id, 10);
  if (!Number.isInteger(id)) {
    return new Err(ErrCode.ParamInvalid, "字形关系 ID 不正确");
  }
  try {
    await env.CHAI.prepare(`DELETE FROM ${table} WHERE id=?`).bind(id).run();
    return true;
  } catch (error) {
    return new Err(
      ErrCode.DataDeleteFailed,
      `删除字形关系失败（${(error as Error).message}）`,
    );
  }
}
