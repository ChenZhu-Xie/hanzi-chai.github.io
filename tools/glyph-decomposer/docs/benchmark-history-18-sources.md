# 18 字源算法阶段回溯

本报告在相同 PDF、candidate catalog、glyph data 与 18 个固定字源上，
从 primitive fitter 初版开始逐阶段重跑。目的不是寻找“测试全绿”的 commit，
而是同时观察覆盖率、路径复用、残余骨架、人工真值 Chamfer 与耗时。

## 口径

- 12 个已有人工有向笔画真值的字源：报告 fits、residual、repeated、Chamfer、RMSE。
- 6 个 upstream 已拆分但没有人工有向真值的 blind 字源：报告 fits、residual、
  repeated、RMSE，不把 Chamfer 纳入成绩。
- `e9b79f0` 以前尚无 `--blind-cases` 入口。对这些阶段，使用 `e9b79f0`
  已冻结的完整 blind fit 生成仅含 case metadata 和等量轨迹的合成 annotation。
  历史 runner 在加载 annotation 前已完成并冻结拟合，因此这些文件只负责定位 case，
  不参与路线选择；其合成 Chamfer 不使用。
- `complete` 表示成功拟合数恰好等于候选笔画数；0/N 视为整字失败。
- `sum repeated`、`sum residual` 只对该阶段成功完成的 case 求和；阶段间若覆盖率不同，
  不能单独用这两个总数判断优劣。

## 阶段总览

| 阶段 | 主要变化 | complete | 0/N | 成功的人工样本平均 Chamfer | sum repeated | sum residual | 四组耗时（秒） |
|---|---|---:|---|---:|---:|---:|---|
| `49fdf7d` | deterministic primitive fitter 初版 | 18/18 | 无 | 1.038 | 329 | 4221 | 168 / 196 / 250 / 382 |
| `4fa1ac7` | 拒绝严重弯折的捺候选 | 18/18 | 无 | 1.006 | 206 | 4159 | 166 / 194 / 250 / 403 |
| `e519aea` | 缓存、IDS 可视化阶段的代表 | 18/18 | 无 | 1.006 | 206 | 4159 | 155 / 161 / 243 / 364 |
| `e9b79f0` | 安全 blind audit 与更严格路径互斥 | 14/18 | 4 | 1.032 | 21 | 3120 | 136 / 174 / 231 / 365 |
| `ea3f9aa` | 全局重拟合冲突 leaf route | 14/18 | 4 | 1.023 | 15 | 2929 | 237 / 176 / 500 / 442 |
| `dec0b63` | 优先完整 primitive route | 11/18 | 7 | 1.306 | 14 | 1425 | 226 / 249 / 596 / 439 |
| `78b6748` | 保留 semantic hook terminal | 10/18 | 8 | 2.466 | 11 | 1352 | 221 / 185 / 600 / 445 |
| archived incident | strict/relaxed fallback、reserve、beam 等未提交实验 | 17/18 | 1 | 1.725 | 144 | 2508 | 411 / 245 / 784 / 451 |

## 逐阶段失败集合

### `49fdf7d`、`4fa1ac7`、`e519aea`

均为 18/18，无 0/N。

### `e9b79f0`、`ea3f9aa`

- U+6418-G-17973
- U+64CF-H-900001
- U+66DA-J-18633
- U+6726-J-18691

### `dec0b63`

- U+6418-G-17973
- U+6424-J-900005
- U+6485-J-18088
- U+64CF-H-900001
- U+66DA-J-18633
- U+66DA-N-57149
- U+6726-J-18691

### `78b6748`

- U+6418-G-17973
- U+6424-J-900005
- U+6485-J-18088
- U+64CF-H-900001
- U+64EC-T-900021
- U+66DA-J-18633
- U+66DA-N-57149
- U+6726-J-18691

### archived incident

- U+66DA-J-18633

## 关键转折

### 1. `49fdf7d` → `4fa1ac7`：无覆盖率代价的明确改善

U+64CF-H 从 residual 272、repeated 131、Chamfer 1.162，改善为
residual 210、repeated 8、Chamfer 0.779；其余固定样本结果不变。

### 2. `4fa1ac7` → `e519aea`：该样本集上算法结果等价

`959ae1d`、`77db48c`、`e519aea` 主要引入 PDF 索引、缓存与 IDS 可视化。
在这 18 个固定字源上，`4fa1ac7` 与 `e519aea` 的逐项指标一致；性能有所改善。

### 3. `e519aea` → `e9b79f0`：从“总能给答案”切到“宁可失败”

覆盖率从 18/18 降为 14/18，但 sum repeated 从 206 降到 21。
这不是普通小回归，而是策略边界变化：更严格的路径互斥消除了大量共享墨迹，
同时没有 fallback，导致四个整字成为 0/N。

### 4. `e9b79f0` → `ea3f9aa`：质量改善但没有救回失败 case

覆盖率保持 14/18；sum repeated 21→15，sum residual 3120→2929，
人工样本平均 Chamfer 1.032→1.023。全局重拟合是净质量增益，但不是覆盖率修复。

### 5. `ea3f9aa` → `dec0b63`：第二次覆盖率断崖

新增三个 0/N：U+6424-J、U+6485-J、U+66DA-N，覆盖率 14→11。
“完整路径优先”与全局 no-reuse/contact signature 组合后，局部没有合法候选会使整字失败。

### 6. `dec0b63` → `78b6748`：hook 修复再次回归

U+64EC-T 从 17/17 变成 0/17；U+808E-T 虽仍为 6/6，Chamfer 从约
1.962 恶化到 6.339、residual 从 208 增至 371。
局部观测 hook 被置于完整主干之前，是这一步的核心风险。

### 7. archived incident：覆盖率回升，但质量与复杂度失控

未提交实验把覆盖率救到 17/18，但 sum repeated 回升到 144，成功人工样本平均
Chamfer 1.725，且 U+66DA-J 仍为 0/17。最慢组从纯净 `78b6748` 的约 600 秒
升至约 784 秒。它证明 fallback 必须存在，也证明不能把 relaxed 路线直接混入主搜索。

## 当前判断

- 最好的“覆盖率基线”是 `4fa1ac7` / `e519aea`：18/18，但共享路径偏多。
- 最好的“严格互斥基线”是 `ea3f9aa`：重复少、质量较稳，但只有 14/18。
- `dec0b63` 和 `78b6748` 都不是合适的继续开发基点；它们分别新增覆盖率断崖。
- 下一轮应以 `ea3f9aa` 的严格结果作为主路径，并把 `e519aea` 的成功路线只作为
  明确标记、可审计的 fallback 候选池；不能让 fallback 与 strict 路线同权竞争。

## 本地 HTML

以下 artifact 不提交 Git：

- `artifacts/deterministic-decomposer/primitive-fit-stage-49fdf7d-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-stage-4fa1ac7-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-stage-e519aea-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-stage-e9b79f0-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-stage-ea3f9aa-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-stage-dec0b63-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-baseline-78b6748-18.html`
- `artifacts/deterministic-decomposer/primitive-fit-diagnosis-current-18.html`
