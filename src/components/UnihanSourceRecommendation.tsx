import {
  Alert,
  Button,
  Card,
  Flex,
  Input,
  Popconfirm,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useAtom } from "jotai";
import { isEqual } from "lodash-es";
import { useMemo, useState } from "react";
import {
  createGlyph,
  getCharacter,
  listCharacters,
  listGlyphs,
  removeGlyph,
  updateCharacter,
} from "~/api";
import { 可编辑字形列表原子, 可编辑字符列表原子 } from "~/atoms";
import {
  applySourceAssignments,
  auditUnihanSources,
  DEFAULT_MINIMUM_EVIDENCE,
  glyphShapeKey,
  parseUnihanIRGSources,
  readUnihanVersion,
  resolveUnresolvedSourceProposal,
  sortSources,
  SUPPORTED_UNIHAN_VERSION,
  type UnihanAudit,
  type UnihanAuditItem,
  type UnihanAuditStatus,
  type UnihanSourceMap,
} from "~/unihan";
import { REVIEWED_SOURCE_DECISIONS } from "~/unihan/reviewed-decisions";

const MAX_APPLY_BATCH = 20;

const statusLabels: Record<UnihanAuditStatus, string> = {
  "reviewed-reference": "已完成训练样本",
  "already-complete": "现有分组符合统计推断",
  "safe-candidate": "自动候选（待确认）",
  "needs-review": "需要人工检查",
  "existing-non-g-data": "已有非 G 数据，跳过",
  "insufficient-evidence": "证据不足",
  unsupported: "不支持",
};

const statusColors: Record<UnihanAuditStatus, string> = {
  "reviewed-reference": "cyan",
  "already-complete": "green",
  "safe-candidate": "blue",
  "needs-review": "orange",
  "existing-non-g-data": "purple",
  "insufficient-evidence": "gold",
  unsupported: "default",
};

function parseHex(value: string): number | undefined {
  const normalized = value.trim().replace(/^U\+/i, "");
  if (!/^[0-9A-F]+$/i.test(normalized)) return undefined;
  return Number.parseInt(normalized, 16);
}

function isApiError(value: unknown): value is { err: string; msg: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "err" in value &&
    "msg" in value
  );
}

function downloadAudit(audit: UnihanAudit) {
  const blob = new Blob([`${JSON.stringify(audit, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "unihan-source-audit.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function proposalGroups(item: UnihanAuditItem) {
  const groups = new Map<
    string,
    { id: number | "新建"; sources: string[]; decomposition?: string }
  >();
  const gEntry = item.existingGlyphs.find((entry) =>
    entry.sources.includes("G"),
  );
  if (gEntry) {
    groups.set(`id:${gEntry.id}`, { id: gEntry.id, sources: ["G"] });
  }
  for (const proposal of item.proposals) {
    const id = proposal.existingGlyphId ?? "新建";
    const key =
      proposal.existingGlyphId === undefined
        ? `shape:${glyphShapeKey(proposal.glyph)}`
        : `id:${proposal.existingGlyphId}`;
    const decomposition =
      proposal.glyph.type === "compound"
        ? `${proposal.glyph.operator}(${proposal.glyph.references.map(({ id }) => id).join(",")})`
        : `部件 ${proposal.existingGlyphId ?? proposal.glyph.id}`;
    const group = groups.get(key) ?? { id, sources: [], decomposition };
    group.sources.push(proposal.source);
    groups.set(key, group);
  }
  return [...groups.values()];
}

export default function UnihanSourceRecommendation() {
  const [characters, setCharacters] = useAtom(可编辑字符列表原子);
  const [glyphs, setGlyphs] = useAtom(可编辑字形列表原子);
  const [sources, setSources] = useState<UnihanSourceMap>(new Map());
  const [filename, setFilename] = useState("");
  const [fromText, setFromText] = useState("4E00");
  const [toText, setToText] = useState("9FFF");
  const [audit, setAudit] = useState<UnihanAudit>();
  const [selected, setSelected] = useState<React.Key[]>([]);
  const [manualChoices, setManualChoices] = useState<Record<string, number>>(
    {},
  );
  const [auditing, setAuditing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [feedback, setFeedback] = useState<{
    type: "success" | "info" | "error";
    text: string;
  }>();
  const from = parseHex(fromText);
  const to = parseHex(toText);
  const validRange = from !== undefined && to !== undefined && from <= to;
  const glyphById = useMemo(
    () => new Map(glyphs.map((glyph) => [glyph.id, glyph])),
    [glyphs],
  );

  const choiceKey = (unicode: number, source: string, referenceId: number) =>
    `${unicode}:${source}:${referenceId}`;

  const formatGlyphTree = (id: number, depth = 2): string => {
    const glyph = glyphById.get(id);
    if (!glyph || glyph.type === "component" || depth === 0) return `${id}`;
    return `${id}=${glyph.operator}(${glyph.references
      .map((reference) => formatGlyphTree(reference.id, depth - 1))
      .join(",")})`;
  };

  const withManualResolution = (
    item: UnihanAuditItem,
    currentCharacters = characters,
    currentGlyphs = glyphs,
  ): UnihanAuditItem => {
    if (item.status !== "insufficient-evidence") return item;
    const character = currentCharacters.find(
      (candidate) => candidate.unicode === item.unicode,
    );
    if (!character) return item;
    const resolved = item.unresolved.flatMap((unresolved) => {
      const choices = new Map<number, number>();
      for (const evidence of unresolved.evidence) {
        const selectedId =
          manualChoices[
            choiceKey(item.unicode, unresolved.source, evidence.referenceId)
          ];
        if (selectedId !== undefined) {
          choices.set(evidence.referenceId, selectedId);
        }
      }
      const proposal = resolveUnresolvedSourceProposal(
        character,
        currentGlyphs,
        unresolved,
        choices,
      );
      return proposal ? [proposal] : [];
    });
    if (resolved.length !== item.unresolved.length) return item;
    return {
      ...item,
      status: "safe-candidate",
      reason:
        "所有统计未决部件均已由维护者选择；写入前仍会对最新远端数据和候选集合重新验证。",
      proposals: [...item.proposals, ...resolved],
      // Keep the original evidence visible so the maintainer can revise a
      // choice before selecting the row for apply.
      unresolved: item.unresolved,
    };
  };

  const effectiveItems = useMemo(
    () => audit?.items.map((item) => withManualResolution(item)) ?? [],
    [audit, characters, glyphs, manualChoices],
  );
  const effectiveSummary = useMemo(() => {
    if (!audit) return undefined;
    const manuallyResolved = effectiveItems.filter(
      (item, index) =>
        item.status === "safe-candidate" &&
        audit.items[index]?.status === "insufficient-evidence",
    ).length;
    return {
      ...audit.summary,
      safeCandidates: audit.summary.safeCandidates + manuallyResolved,
      insufficientEvidence:
        audit.summary.insufficientEvidence - manuallyResolved,
    };
  }, [audit, effectiveItems]);

  const runAudit = (currentCharacters = characters, currentGlyphs = glyphs) => {
    if (!validRange || from === undefined || to === undefined) {
      setFeedback({ type: "error", text: "请输入有效的 Unicode 范围。" });
      return;
    }
    setAuditing(true);
    window.setTimeout(() => {
      const result = auditUnihanSources(
        sources,
        currentCharacters,
        currentGlyphs,
        {
          from,
          to,
          reviewedDecisions: REVIEWED_SOURCE_DECISIONS,
        },
      );
      setAudit(result);
      setSelected([]);
      setManualChoices({});
      setAuditing(false);
    }, 0);
  };

  const selectedSafe = useMemo(
    () =>
      effectiveItems.filter(
        (item) =>
          item.status === "safe-candidate" && selected.includes(item.unicode),
      ),
    [effectiveItems, selected],
  );

  const applySelected = async () => {
    if (!audit || selectedSafe.length === 0) return;
    if (selectedSafe.length > MAX_APPLY_BATCH) {
      setFeedback({
        type: "error",
        text: `单批最多写入 ${MAX_APPLY_BATCH} 个候选，请缩小选择范围。`,
      });
      return;
    }
    setApplying(true);
    try {
      const [latestCharacters, latestGlyphs] = await Promise.all([
        listCharacters(),
        listGlyphs(),
      ]);
      if (isApiError(latestCharacters)) {
        throw new Error(`写入前刷新远端字符失败：${latestCharacters.msg}`);
      }
      if (!Array.isArray(latestCharacters)) {
        throw new Error("写入前刷新远端字符失败：响应格式不正确");
      }
      if (isApiError(latestGlyphs)) {
        throw new Error(`写入前刷新远端字形失败：${latestGlyphs.msg}`);
      }
      if (!Array.isArray(latestGlyphs)) {
        throw new Error("写入前刷新远端字形失败：响应格式不正确");
      }
      let currentCharacters = latestCharacters;
      const currentGlyphs = latestGlyphs;
      const freshAudit = auditUnihanSources(
        sources,
        currentCharacters,
        currentGlyphs,
        { from, to, reviewedDecisions: REVIEWED_SOURCE_DECISIONS },
      );
      const freshItems = new Map(
        freshAudit.items.map((item) => [item.unicode, item]),
      );
      const glyphIdByShape = new Map(
        currentGlyphs.map((glyph) => [glyphShapeKey(glyph), glyph.id]),
      );
      let updated = 0;
      let failed = 0;
      let skipped = 0;
      const errors: string[] = [];

      const rollbackCreatedGlyphs = async (ids: number[]) => {
        for (const id of [...ids].reverse()) {
          try {
            const response = await removeGlyph(id);
            if (isApiError(response)) {
              errors.push(`回滚字形 ${id} 失败：${response.msg}`);
              continue;
            }
            if (response !== true) {
              errors.push(`回滚字形 ${id} 失败：响应格式不正确`);
              continue;
            }
            const index = currentGlyphs.findIndex((glyph) => glyph.id === id);
            if (index !== -1) currentGlyphs.splice(index, 1);
            for (const [shape, glyphId] of glyphIdByShape) {
              if (glyphId === id) glyphIdByShape.delete(shape);
            }
          } catch (error) {
            errors.push(
              `回滚字形 ${id} 结果未知：${error instanceof Error ? error.message : String(error)}`,
            );
          }
        }
      };

      for (const selectedItem of selectedSafe) {
        const freshItem = freshItems.get(selectedItem.unicode);
        const item = freshItem
          ? withManualResolution(freshItem, currentCharacters, currentGlyphs)
          : undefined;
        const character = currentCharacters.find(
          (candidate) => candidate.unicode === selectedItem.unicode,
        );
        if (item?.status !== "safe-candidate" || !character) {
          skipped++;
          continue;
        }

        const assignments: { source: string; id: number }[] = [];
        const createdIds: number[] = [];
        try {
          for (const proposal of item.proposals) {
            const shapeKey = glyphShapeKey(proposal.glyph);
            let id = proposal.existingGlyphId ?? glyphIdByShape.get(shapeKey);
            if (id === undefined) {
              const response = await createGlyph(proposal.glyph);
              if (isApiError(response)) {
                throw new Error(`创建字形失败：${response.msg}`);
              }
              if (!Number.isInteger(response)) {
                throw new Error("创建字形失败：响应格式不正确");
              }
              id = response;
              createdIds.push(id);
              currentGlyphs.push({ ...proposal.glyph, id });
              glyphIdByShape.set(shapeKey, id);
            }
            assignments.push({ source: proposal.source, id });
          }

          const updatedCharacter = applySourceAssignments(
            character,
            assignments,
          );
          const latestCharacter = await getCharacter(character.unicode);
          if (isApiError(latestCharacter)) {
            throw new Error(`更新前复核字符失败：${latestCharacter.msg}`);
          }
          if (
            typeof latestCharacter !== "object" ||
            latestCharacter === null ||
            !isEqual(latestCharacter, character)
          ) {
            throw new Error("远端字符已变化，已取消本项写入");
          }
          const response = await updateCharacter(updatedCharacter);
          if (isApiError(response)) {
            throw new Error(`更新字符失败：${response.msg}`);
          }
          if (response !== true) {
            throw new Error("更新字符失败：响应格式不正确");
          }
          currentCharacters = currentCharacters.map((candidate) =>
            candidate.unicode === updatedCharacter.unicode
              ? updatedCharacter
              : candidate,
          );
          updated++;
        } catch (error) {
          failed++;
          errors.push(
            `${item.codepoint} ${error instanceof Error ? error.message : String(error)}`,
          );
          await rollbackCreatedGlyphs(createdIds);
        }
      }

      setCharacters(currentCharacters);
      setGlyphs(currentGlyphs);
      runAudit(currentCharacters, currentGlyphs);
      setFeedback({
        type: errors.length > 0 ? "error" : "info",
        text: `写入 ${updated}；失败 ${failed}；状态变化后跳过 ${skipped}。${errors[0] ?? ""}`,
      });
    } catch (error) {
      setFeedback({
        type: "error",
        text: `写入流程中断；请重新 dry-run 核对远端状态。${error instanceof Error ? error.message : String(error)}`,
      });
    } finally {
      setApplying(false);
    }
  };

  const columns: ColumnsType<UnihanAuditItem> = [
    {
      title: "字符",
      dataIndex: "character",
      width: 72,
      render: (value: string) => (
        <Typography.Text className="text-2xl">{value}</Typography.Text>
      ),
    },
    { title: "码位", dataIndex: "codepoint", width: 96 },
    {
      title: "Unihan 来源",
      dataIndex: "expectedSources",
      width: 130,
      render: (value: string[]) => value.join(" "),
    },
    {
      title: "现有完整分组",
      dataIndex: "existingGlyphs",
      width: 190,
      render: (value: UnihanAuditItem["existingGlyphs"]) =>
        value.length > 0 ? (
          <Flex vertical gap={2}>
            {value.map((entry) => (
              <Typography.Text key={entry.id}>
                {entry.sources.join(" ") || "无来源"} → {entry.id}
              </Typography.Text>
            ))}
          </Flex>
        ) : (
          "—"
        ),
    },
    {
      title: "来源标签缺失",
      dataIndex: "missingSources",
      width: 110,
      render: (value: string[]) => value.join(" ") || "—",
    },
    {
      title: "建议重分来源",
      width: 120,
      render: (_, item) =>
        sortSources(
          item.proposals
            .filter((proposal) => proposal.requiresChange)
            .map((proposal) => proposal.source),
        ).join(" ") || "—",
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 150,
      render: (value: UnihanAuditStatus, item) => (
        <Tag color={statusColors[value]}>
          {value === "safe-candidate" && item.unresolved.length > 0
            ? "人工选择完整（待勾选）"
            : statusLabels[value]}
        </Tag>
      ),
    },
    {
      title: "建议完整分组 / 未决部件",
      width: 420,
      render: (_, item) => {
        const groups = proposalGroups(item);
        return groups.length > 0 || item.unresolved.length > 0 ? (
          <Flex vertical gap={2}>
            {groups.map((group) => (
              <Typography.Text key={`${group.id}:${group.sources.join("")}`}>
                {sortSources(group.sources).join(" ")} → {group.id}
                {group.decomposition ? ` ${group.decomposition}` : ""}
              </Typography.Text>
            ))}
            {item.unresolved.map(({ source, evidence }) => (
              <Flex vertical gap={2} key={source}>
                <Typography.Text type="warning">
                  {source}：统计未决，请逐层确认兄弟部件
                </Typography.Text>
                {evidence.length === 0 ? (
                  <Typography.Text type="secondary">
                    没有已完成区间证据
                  </Typography.Text>
                ) : (
                  evidence.map((entry) => {
                    const key = choiceKey(
                      item.unicode,
                      source,
                      entry.referenceId,
                    );
                    return (
                      <Flex gap="small" align="center" key={entry.referenceId}>
                        <Typography.Text>
                          G 部件 {formatGlyphTree(entry.referenceId)} →
                        </Typography.Text>
                        {entry.reliable && entry.replacementId !== undefined ? (
                          <Typography.Text>
                            {formatGlyphTree(entry.replacementId)}（
                            {entry.reviewed ? "人工已确认" : "统计已定"}）
                          </Typography.Text>
                        ) : entry.alternatives.length > 0 ? (
                          <Select
                            aria-label={`${item.codepoint} ${source} 部件 ${entry.referenceId} 的人工选择`}
                            placeholder="选择视觉匹配项"
                            value={manualChoices[key]}
                            className="min-w-72"
                            options={entry.alternatives.map((alternative) => ({
                              value: alternative.id,
                              label: `${formatGlyphTree(alternative.id)}（${alternative.count} 个已完成父字）`,
                            }))}
                            allowClear
                            onChange={(value) => {
                              setManualChoices((current) => {
                                const next = { ...current };
                                if (value === undefined) delete next[key];
                                else next[key] = value;
                                return next;
                              });
                              setSelected((current) =>
                                current.filter(
                                  (value) => value !== item.unicode,
                                ),
                              );
                            }}
                          />
                        ) : (
                          <Typography.Text type="secondary">
                            无候选
                          </Typography.Text>
                        )}
                      </Flex>
                    );
                  })
                )}
              </Flex>
            ))}
            <Typography.Text type="secondary">{item.reason}</Typography.Text>
          </Flex>
        ) : (
          item.reason
        );
      },
    },
  ];

  return (
    <Card
      size="small"
      title="Unicode 18.0 非 G 来源候选（先审计，人工核对后写入）"
      className="m-4 mb-0 shrink-0"
    >
      <Flex vertical gap="middle">
        <Alert
          type="info"
          showIcon
          message="Unihan_IRGSources.txt 决定 source membership；PDF 只用于视觉核对。Dry-run 不会发起写请求。"
          description={`审计还会使用 ${REVIEWED_SOURCE_DECISIONS.length} 条已逐项视觉确认的最小部件决策；它们不会单独触发写入。当前不因字体细节新建兄弟部件：横／提、点／捺、竖／竖钩、竖弯钩／竖提可留作将来的条件变体。`}
        />
        {feedback && (
          <Alert
            type={feedback.type}
            showIcon
            closable
            message={feedback.text}
            onClose={() => setFeedback(undefined)}
          />
        )}
        <Flex gap="small" align="center" wrap>
          <Upload
            accept=".txt,text/plain"
            maxCount={1}
            showUploadList={false}
            beforeUpload={async (file) => {
              const text = await file.text();
              const version = readUnihanVersion(text);
              if (version !== SUPPORTED_UNIHAN_VERSION) {
                setSources(new Map());
                setFilename("");
                setAudit(undefined);
                setSelected([]);
                setManualChoices({});
                setFeedback({
                  type: "error",
                  text: `仅接受 Unicode ${SUPPORTED_UNIHAN_VERSION}；当前文件版本为 ${version ?? "未知"}。`,
                });
                return false;
              }
              const parsed = parseUnihanIRGSources(text);
              setSources(parsed);
              setFilename(file.name);
              setAudit(undefined);
              setSelected([]);
              setManualChoices({});
              setFeedback({
                type: "success",
                text: `读取 ${parsed.size} 个 U+4E00..U+9FFF 来源记录。`,
              });
              return false;
            }}
          >
            <Button>{filename || "载入 Unihan_IRGSources.txt"}</Button>
          </Upload>
          <span>从</span>
          <Input
            aria-label="起始码位"
            className="w-28!"
            prefix="U+"
            value={fromText}
            onChange={(event) => setFromText(event.target.value)}
          />
          <span>到</span>
          <Input
            aria-label="结束码位"
            className="w-28!"
            prefix="U+"
            value={toText}
            onChange={(event) => setToText(event.target.value)}
          />
          <Button
            type="primary"
            disabled={
              sources.size === 0 || characters.length === 0 || !validRange
            }
            loading={auditing}
            onClick={() => runAudit()}
          >
            Dry-run 审计
          </Button>
          <Button
            disabled={!audit}
            onClick={() => audit && downloadAudit(audit)}
          >
            下载 JSON
          </Button>
          <Popconfirm
            title={`写入已人工核对的 ${selectedSafe.length} 个候选？`}
            description={`将逐个重新审计并在失败时回滚新建字形；每批最多 ${MAX_APPLY_BATCH} 个。`}
            okText="写入选中项"
            cancelText="取消"
            onConfirm={applySelected}
          >
            <Button
              danger
              disabled={
                selectedSafe.length === 0 ||
                selectedSafe.length > MAX_APPLY_BATCH
              }
              loading={applying}
            >
              应用已核对候选（{selectedSafe.length}）
            </Button>
          </Popconfirm>
        </Flex>

        {audit && (
          <>
            <Space size="large" wrap>
              <Statistic title="扫描" value={effectiveSummary?.scanned} />
              <Statistic title="含非 G" value={effectiveSummary?.hasNonG} />
              <Statistic
                title="已完成训练样本"
                value={effectiveSummary?.reviewedReferences}
              />
              <Statistic
                title="现有分组符合统计推断"
                value={effectiveSummary?.alreadyComplete}
              />
              <Statistic
                title={`自动候选（≥${DEFAULT_MINIMUM_EVIDENCE} 个独立样本）`}
                value={effectiveSummary?.safeCandidates}
              />
              <Statistic
                title="缺少 G 参考，需检查"
                value={effectiveSummary?.needsReview}
              />
              <Statistic
                title="证据不足"
                value={effectiveSummary?.insufficientEvidence}
              />
              <Statistic
                title="当前不支持"
                value={effectiveSummary?.unsupported}
              />
              <Statistic
                title="已有非 G，跳过"
                value={effectiveSummary?.skippedExistingNonG}
              />
            </Space>
            <Table<UnihanAuditItem>
              aria-label="Unihan 来源审计结果"
              rowKey="unicode"
              size="small"
              columns={columns}
              dataSource={effectiveItems}
              pagination={{ defaultPageSize: 20, showSizeChanger: true }}
              scroll={{ x: 1000, y: 360 }}
              rowSelection={{
                hideSelectAll: true,
                selectedRowKeys: selected,
                onChange: setSelected,
                getCheckboxProps: (item) => ({
                  disabled: item.status !== "safe-candidate",
                  title:
                    item.status === "safe-candidate"
                      ? "选择已人工核对的候选"
                      : statusLabels[item.status],
                }),
              }}
            />
          </>
        )}
      </Flex>
    </Card>
  );
}
