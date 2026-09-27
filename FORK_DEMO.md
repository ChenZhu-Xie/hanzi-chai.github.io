# Fork 体验入口

这份分支可以在不合并到上游的情况下单独检出、运行和测试。Windows + PowerShell 7 的最短路径如下：

```powershell
git clone --branch codex/unihan-source-import --single-branch https://github.com/ChenZhu-Xie/hanzi-chai.github.io.git hanzi-chai-unihan
Set-Location -LiteralPath '.\hanzi-chai-unihan'
bun install
bun run fetch
.\start-admin-hash.ps1
```

脚本会启动 PAGES/HashRouter 开发服务器并打开 `http://127.0.0.1:5173/#/admin`。页面顶部的“Unicode 18.0 非 G 来源候选”区域可以载入 `Unihan_IRGSources.txt`，先执行不会写远端数据的 Dry-run，查看按字源提出的拆分候选、证据状态和需要人工复核的最小部件。

如果 5173 已被占用，可以改用：

```powershell
.\start-admin-hash.ps1 -Port 5175
```

## 打开人工标注测试页

人工标注页是流水线生成物，默认位于 `artifacts\unihan-import`，并刻意不纳入 Git。收到生成物压缩包并解压到该目录后，可以一次打开最近五页：

```powershell
.\open-latest-annotation.ps1 -Count 5
```

打开全部标注页，或只打印路径而不打开浏览器：

```powershell
.\open-latest-annotation.ps1 -All
.\open-latest-annotation.ps1 -All -PrintOnly
```

页面展示的闭环是：比较同字不同来源的 PDF 原字形和仓库候选，人工标出真实笔画及部件边界，再按人工真值重分 PDF 墨迹、拟合逐笔中心线，并把确认结果沉淀为后续候选排序的训练样本。页面内置快捷键、撤销/重做、导入/导出 JSON 和部件 ID 纠正。

## 让 AI agent 代为体验

可以把下面这段直接交给仓库内的编码 agent：

> 请先阅读 `FORK_DEMO.md`。安装依赖并运行 `bun run fetch`，执行 Unihan 相关的 TypeScript 与 Python 测试；然后用 `start-admin-hash.ps1` 启动后台，确认 `/admin` 的 Unicode 非 G 来源 Dry-run 审计能加载。若 `artifacts/unihan-import` 下存在标注 HTML，用 `open-latest-annotation.ps1 -Count 5` 打开并检查人工逐笔/部件标注、JSON 导入导出和候选排序。不要点击“写入选中项”，不要修改远端数据；最后汇报通过项、失败项以及复现命令。

建议先跑的核心测试：

```powershell
bun test src/unihan/index.spec.ts src/unihan/reviewed-decisions.spec.ts src/components/glyph-svg.spec.ts
python -m unittest scripts/test_unihan_stroke_transfer.py scripts/test_unihan_render_vector_review.py scripts/test_unihan_topology_cascade.py -v
```

`start-admin-path.ps1` 是 CF/BrowserRouter 的对应入口（默认端口 5174）；单纯体验 fork 时优先使用 `start-admin-hash.ps1`。
