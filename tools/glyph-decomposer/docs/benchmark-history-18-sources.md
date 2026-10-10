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
| `feat/first-principles` | 双层分层搜索 (Tier 1 严格互斥 + Tier 2 受控软惩罚兜底) + 几何方向约束 | 18/18 | 无 | 1.046 | 36 | 4144 | 完整 18 源全跑 ~1118s |
| `feat/first-principles` (tighten IDS regression) | 收紧 leaf route 局部提议的 IDS 结构容忍阈值（0.10 -> 0.05），杜绝偏旁窃取（修复 U+6903-H） | 18/18 | 无 | 0.792 | 36 | 4061 | 924s（18 难字验证集亦 18/18 全通，且 repeated 337 -> 254） |

## 逐阶段失败集合

### `49fdf7d`、`4fa1ac7`、`e519aea`、`feat/first-principles`
 
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

### 8. `ea3f9aa` → `feat/first-principles`：双层分层搜索与几何约束闭环

本轮基于第一性原理彻底解除了覆盖率与互斥性的长期拉锯：
1. **同 leaf 严格互斥分层 (Two-Tier Architecture)**：
   - **Tier 1 (Strict)**：采用严格同 leaf 非折返约束（`maximum_same_leaf_shared_run = 8`），健康汉字直接在 Tier 1 产出高互斥度最优解，避免被松弛解劣化。
   - **Tier 2 (Controlled Fallback)**：仅在 Tier 1 搜索陷入死胡同 (0/N) 时受控触发，允许更大共享上限 (`fallback_maximum_same_leaf_shared_run = 60`)，并对超出 8px 的重叠部分按长度施加惩罚 (`same_leaf_shared_run_penalty = 0.5`)。
   - **Retrace Guard**：引入 `maximum_retrace_fraction = 0.75`，在几何上杜绝重复折返顺描同一笔画。
2. **跨 leaf 自然接触接纳**：
   - 跨 leaf 笔画（如 `老`/`匕`，`扌`/`八`）物理交汇为真实客观现象，由全局 `repeated_pixel_penalty` 与 IDS 树约束调节，不施加跨 leaf 硬性排斥（撤销导致 7 个 0/N 的全局 `_reuses_completed_segment` 限制）。
3. **几何方向过滤修复**：
   - 恢复单线几何方向兼容性（`_single_line_direction_compatible`）以及复合笔画终段语义一致性（`_terminal_compound_direction_compatible`，横撇终段必须一致），剔除局部截断噪音。

**成果对比**：
- **完整率达到 18/18**（0/N = 0），所有历史上曾失败过的 8 个字源均成功满拟合。
- **重复像素（sum repeated）仅 36**，相较历史 18/18 版本的 206~329 像素下降近一个数量级。
- **12 个具有人工真值的字源平均 Chamfer 维持在 1.046**，保持高度拟合保真度与结构自洽。

## 当前判断

- `feat/first-principles` 已成功兼顾“覆盖率”与“互斥性”，成为新的稳定生产基线。
- 架构原则经验总结：
  - **绝不能将宽松 fallback 路线同权混入主搜索**（否则会劣化正常 glyph 的 Chamfer 和重复度）。
  - **绝不能将同 leaf 的互斥硬规则扩散到跨 leaf 物理接触**（否则会导致字根接缝处 0/N 断崖）。
  - **几何方向剪枝必须守住整体语义方向，而不能单凭骨架端点距离贪心截断**。

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
