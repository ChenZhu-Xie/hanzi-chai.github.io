import {
  Alert,
  Button,
  Card,
  Flex,
  Input,
  Popconfirm,
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
  SUPPORTED_UNIHAN_VERSION,
  type UnihanAudit,
  type UnihanAuditItem,
  type UnihanAuditStatus,
  type UnihanSourceMap,
} from "~/unihan";

const MAX_APPLY_BATCH = 20;

const statusLabels: Record<UnihanAuditStatus, string> = {
  "already-complete": "已完整",
  "safe-candidate": "自动候选（待确认）",
  "needs-review": "需要人工检查",
  "existing-non-g-data": "已有非 G 数据，跳过",
  "insufficient-evidence": "证据不足",
  unsupported: "不支持",
};

const statusColors: Record<UnihanAuditStatus, string> = {
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

export default function UnihanSourceRecommendation() {
  const [characters, setCharacters] = useAtom(可编辑字符列表原子);
  const [glyphs, setGlyphs] = useAtom(可编辑字形列表原子);
  const [sources, setSources] = useState<UnihanSourceMap>(new Map());
  const [filename, setFilename] = useState("");
  const [fromText, setFromText] = useState("4E00");
  const [toText, setToText] = useState("9FFF");
  const [audit, setAudit] = useState<UnihanAudit>();
  const [selected, setSelected] = useState<React.Key[]>([]);
  const [auditing, setAuditing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [feedback, setFeedback] = useState<{
    type: "success" | "info" | "error";
    text: string;
  }>();
  const from = parseHex(fromText);
  const to = parseHex(toText);
  const validRange = from !== undefined && to !== undefined && from <= to;

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
        },
      );
      setAudit(result);
      setSelected([]);
      setAuditing(false);
    }, 0);
  };

  const selectedSafe = useMemo(
    () =>
      audit?.items.filter(
        (item) =>
          item.status === "safe-candidate" && selected.includes(item.unicode),
      ) ?? [],
    [audit, selected],
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
        { from, to },
      );
      const freshItems = new Map(
        freshAudit.items.map((item) => [item.unicode, item]),
      );
      const glyphIdByShape = new Map(
        currentGlyphs
          .filter((glyph) => glyph.type === "compound")
          .map((glyph) => [glyphShapeKey(glyph), glyph.id]),
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
        const item = freshItems.get(selectedItem.unicode);
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
            throw new Error(
              `更新前复核字符失败：${latestCharacter.msg}`,
            );
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
      title: "现有来源",
      dataIndex: "existingSources",
      width: 110,
      render: (value: string[]) => value.join(" ") || "—",
    },
    {
      title: "缺失来源",
      dataIndex: "missingSources",
      width: 110,
      render: (value: string[]) => value.join(" ") || "—",
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 150,
      render: (value: UnihanAuditStatus) => (
        <Tag color={statusColors[value]}>{statusLabels[value]}</Tag>
      ),
    },
    {
      title: "Proposal / 原因",
      render: (_, item) =>
        item.proposals.length > 0 ? (
          <Flex vertical gap={2}>
            {item.proposals.map((proposal) => (
              <Typography.Text key={proposal.source}>
                {proposal.source} → 字形 {proposal.existingGlyphId ?? "新建"}；
                {proposal.evidence
                  .map(
                    (entry) =>
                      `${entry.referenceId}→${entry.replacementId}(${entry.count})`,
                  )
                  .join("、")}
              </Typography.Text>
            ))}
          </Flex>
        ) : (
          item.reason
        ),
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
              <Statistic title="扫描" value={audit.summary.scanned} />
              <Statistic title="含非 G" value={audit.summary.hasNonG} />
              <Statistic title="已完整" value={audit.summary.alreadyComplete} />
              <Statistic
                title={`自动候选（≥${DEFAULT_MINIMUM_EVIDENCE} 个独立样本）`}
                value={audit.summary.safeCandidates}
              />
              <Statistic title="需检查" value={audit.summary.needsReview} />
              <Statistic
                title="已有非 G，跳过"
                value={audit.summary.skippedExistingNonG}
              />
            </Space>
            <Table<UnihanAuditItem>
              aria-label="Unihan 来源审计结果"
              rowKey="unicode"
              size="small"
              columns={columns}
              dataSource={audit.items}
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
