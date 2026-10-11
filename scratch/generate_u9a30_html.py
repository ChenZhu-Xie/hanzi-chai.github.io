import json
from pathlib import Path

data = json.loads(Path("scratch/u9a30_annotation_data.json").read_text(encoding="utf-8"))

claimed_total = data["claimed_total"]
total_pts = data["total_pts"]
cov_pct = f"{claimed_total / total_pts * 100:.1f}%"
unclaimed_count = len(data["unclaimed_indices"])
red_pct = f"{(633 - unclaimed_count) / 633 * 100:.0f}%"

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>U+9A30 騰 · K 视觉逐笔时空切片标注与算法病灶剖析</title>
  <style>
    :root {{
      --bg-dark: #090d16;
      --bg-card: rgba(17, 24, 39, 0.75);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-active: rgba(56, 189, 248, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --danger: #f43f5e;
      --danger-glow: rgba(244, 63, 94, 0.25);
      --success: #10b981;
      --warning: #f59e0b;
      --radius: 12px;
      --radius-sm: 8px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-dark) 50%, #030712 100%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      line-height: 1.5;
    }}

    header {{
      padding: 18px 32px;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(9, 13, 22, 0.8);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}

    .header-left {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .glyph-badge {{
      width: 48px;
      height: 48px;
      border-radius: var(--radius-sm);
      background: linear-gradient(135deg, #7c3aed, #06b6d4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 26px;
      font-weight: 700;
      color: white;
      box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }}

    .title-group h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .title-group p {{
      font-size: 13px;
      color: var(--text-muted);
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background: rgba(56, 189, 248, 0.12);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.25);
    }}

    .tag-danger {{
      background: rgba(244, 63, 94, 0.12);
      color: #f43f5e;
      border-color: rgba(244, 63, 94, 0.25);
    }}

    .tag-success {{
      background: rgba(16, 185, 129, 0.12);
      color: #10b981;
      border-color: rgba(16, 185, 129, 0.25);
    }}

    .header-right {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .stat-pill {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      padding: 4px 12px;
      background: rgba(255, 255, 255, 0.04);
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
    }}

    .stat-pill .val {{
      font-size: 16px;
      font-weight: 700;
      color: var(--accent);
    }}

    .stat-pill .lbl {{
      font-size: 10px;
      color: var(--text-dim);
      text-transform: uppercase;
    }}

    /* Main Container */
    main {{
      flex: 1;
      padding: 24px 32px;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 560px 1fr;
      gap: 24px;
      align-items: start;
    }}

    @media (max-width: 1100px) {{
      main {{
        grid-template-columns: 1fr;
      }}
    }}

    /* Left Panel: Viewer */
    .viewer-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
      position: sticky;
      top: 96px;
    }}

    .canvas-wrapper {{
      position: relative;
      width: 100%;
      aspect-ratio: 1;
      background: #030712;
      border-radius: var(--radius-sm);
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: inset 0 0 40px rgba(0,0,0,0.8);
    }}

    svg.main-canvas {{
      width: 100%;
      height: 100%;
      transform-origin: center center;
      transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .canvas-overlay-ui {{
      position: absolute;
      top: 12px;
      left: 12px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }}

    .canvas-btn {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 6px 10px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      backdrop-filter: blur(8px);
    }}

    .canvas-btn:hover {{
      background: rgba(30, 41, 59, 0.95);
      border-color: var(--accent);
      color: white;
    }}

    .canvas-btn.active {{
      background: rgba(56, 189, 248, 0.18);
      border-color: var(--accent);
      color: var(--accent);
    }}

    .canvas-btn.danger.active {{
      background: rgba(244, 63, 94, 0.2);
      border-color: var(--danger);
      color: #fda4af;
    }}

    .layer-switches {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--text-muted);
      padding: 0 4px;
    }}

    .switch-label {{
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }}

    .switch-label input {{
      accent-color: var(--accent);
      cursor: pointer;
    }}

    /* Scrubber Section */
    .scrubber-card {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .scrubber-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .playback-btns {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .p-btn {{
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      width: 34px;
      height: 34px;
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.15s ease;
    }}

    .p-btn:hover {{
      background: rgba(255, 255, 255, 0.12);
      border-color: var(--accent);
      color: white;
    }}

    .p-btn.play {{
      background: linear-gradient(135deg, #7c3aed, #2563eb);
      border-color: rgba(255, 255, 255, 0.2);
      color: white;
      width: 40px;
      font-size: 15px;
    }}

    .p-btn.play:hover {{
      filter: brightness(1.15);
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .scrubber-info {{
      font-size: 12px;
      font-weight: 600;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .slider-track-wrap {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    input[type="range"].time-slider {{
      -webkit-appearance: none;
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: #1e293b;
      outline: none;
      cursor: pointer;
      position: relative;
    }}

    input[type="range"].time-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #38bdf8;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
      cursor: pointer;
      border: 2px solid white;
      transition: transform 0.1s ease;
    }}

    input[type="range"].time-slider::-webkit-slider-thumb:hover {{
      transform: scale(1.2);
    }}

    /* Keyframe Pills Track (21 pills) */
    .keyframe-pills {{
      display: grid;
      grid-template-columns: repeat(21, 1fr);
      gap: 2px;
      margin-top: 4px;
    }}

    .k-pill {{
      padding: 4px 0;
      text-align: center;
      font-size: 9px;
      font-weight: 600;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-dim);
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.12s ease;
      user-select: none;
    }}

    .k-pill:hover {{
      background: rgba(255, 255, 255, 0.1);
      color: var(--text-main);
    }}

    .k-pill.active {{
      background: var(--accent);
      color: #030712;
      font-weight: 700;
      box-shadow: 0 0 8px var(--accent-glow);
    }}

    .k-pill.special {{
      border-color: rgba(244, 63, 94, 0.4);
    }}

    .k-pill.special.active {{
      background: var(--danger);
      color: white;
      box-shadow: 0 0 10px var(--danger-glow);
    }}

    /* Right Panel: Thought & Diagnostic Deck */
    .detail-deck {{
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    .thought-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
      position: relative;
      overflow: hidden;
    }}

    .thought-card::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--active-stroke-color, var(--accent));
      transition: background 0.3s ease;
    }}

    .thought-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 16px;
    }}

    .stroke-main-info h2 {{
      font-size: 22px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--text-main);
    }}

    .stroke-main-info .sub {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .tool-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
    }}

    /* Content Blocks */
    .block-section {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .block-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .block-title.danger {{
      color: var(--danger);
    }}

    .block-content {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      font-size: 14px;
      color: #e2e8f0;
      line-height: 1.65;
    }}

    .block-content.critique {{
      border-left: 3px solid var(--danger);
      background: rgba(244, 63, 94, 0.05);
    }}

    .param-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
    }}

    .param-box {{
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .param-box .key {{
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
    }}

    .param-box .val {{
      font-size: 13px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #38bdf8;
      word-break: break-all;
    }}

    /* Progress Stats */
    .coverage-meter {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      background: rgba(0,0,0,0.3);
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
    }}

    .meter-label-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
    }}

    .meter-bar-track {{
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }}

    .meter-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
      width: 0%;
      transition: width 0.3s ease;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }}

    /* Ghost Comparison Callout */
    .ghost-callout {{
      display: none;
      background: rgba(244, 63, 94, 0.08);
      border: 1px dashed var(--danger);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      font-size: 13px;
      color: #fda4af;
      line-height: 1.6;
    }}

    .ghost-callout.show {{
      display: block;
      animation: fadeIn 0.3s ease;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Master Stroke Table */
    .table-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius);
      padding: 20px 24px;
      grid-column: 1 / -1;
      margin-top: 10px;
    }}

    .table-card h3 {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    table.stroke-tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}

    table.stroke-tbl th {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-dim);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
    }}

    table.stroke-tbl td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-main);
    }}

    table.stroke-tbl tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
    }}

    table.stroke-tbl tr.row-active td {{
      background: rgba(56, 189, 248, 0.12);
      font-weight: 600;
    }}

    .dot-badge {{
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 6px;
    }}

    /* Global Summary Section */
    .summary-card {{
      grid-column: 1 / -1;
      background: linear-gradient(135deg, rgba(30, 27, 75, 0.4), rgba(15, 23, 42, 0.8));
      border: 1px solid rgba(129, 140, 248, 0.2);
      border-radius: var(--radius);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}

    .summary-card h3 {{
      font-size: 17px;
      font-weight: 700;
      color: #c7d2fe;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}

    .summary-item {{
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      font-size: 13px;
      line-height: 1.6;
    }}

    .summary-item strong {{
      color: #38bdf8;
      display: block;
      margin-bottom: 4px;
    }}

    /* Animations */
    @keyframes pulseGlow {{
      0%, 100% {{ filter: drop-shadow(0 0 2px var(--accent)) drop-shadow(0 0 6px var(--accent-glow)); }}
      50% {{ filter: drop-shadow(0 0 6px var(--accent)) drop-shadow(0 0 14px var(--accent-glow)); }}
    }}

    .active-stroke-path {{
      animation: pulseGlow 2s infinite ease-in-out;
    }}

    /* Interactive point hover tooltip */
    #tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 11px;
      color: white;
      z-index: 100;
      display: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="header-left">
      <div class="glyph-badge">騰</div>
      <div class="title-group">
        <h1>U+9A30 騰 · K (KSX-133032) <span class="tag tag-success">视觉模型标注</span> <span class="tag tag-danger">算法病灶对照</span></h1>
        <p>20 笔全要素时空切片逐笔标注 · 骨架解释覆盖率 {cov_pct} ({claimed_total} / {total_pts} 点) · 攻克 633 巨额残余翻车重灾区</p>
      </div>
    </div>
    <div class="header-right">
      <div class="stat-pill">
        <span class="val">20 笔</span>
        <span class="lbl">正字笔画数</span>
      </div>
      <div class="stat-pill">
        <span class="val">{cov_pct}</span>
        <span class="lbl">骨架解释覆盖率</span>
      </div>
      <div class="stat-pill">
        <span class="val">633 → {unclaimed_count}</span>
        <span class="lbl">残差点剧减 {red_pct}</span>
      </div>
    </div>
  </header>

  <!-- Main View -->
  <main>
    <!-- Left: Canvas & Interactive Scrubber -->
    <div class="viewer-card">
      <div class="canvas-wrapper" id="canvasContainer">
        <div class="canvas-overlay-ui">
          <button class="canvas-btn" id="btnZoomIn" title="放大">➕</button>
          <button class="canvas-btn" id="btnZoomOut" title="缩小">➖</button>
          <button class="canvas-btn" id="btnResetZoom" title="重置">⟲ 100%</button>
          <button class="canvas-btn danger active" id="btnToggleGhost" title="原算法错误路径对照">👻 原算法翻车对照</button>
        </div>

        <svg class="main-canvas" id="mainSvg" viewBox="-2 -2 104 104">
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="0.5"/>
            </pattern>
          </defs>

          <!-- Background grid -->
          <rect x="-2" y="-2" width="104" height="104" fill="url(#grid)" />

          <!-- Font Outline Layer -->
          <path id="glyphOutline" d="{data['svg_outline']}" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.18)" stroke-width="0.35" />

          <!-- Skeleton Points Layer -->
          <g id="skelLayer"></g>

          <!-- Old Algorithm Ghost Path Layer -->
          <g id="ghostLayer"></g>

          <!-- Fitted Strokes Layer -->
          <g id="strokesLayer"></g>

          <!-- Active Stroke Controls & Handles Layer -->
          <g id="activeHandlesLayer"></g>
        </svg>
      </div>

      <!-- Layer Switches -->
      <div class="layer-switches">
        <label class="switch-label">
          <input type="checkbox" id="chkOutline" checked> 字形外轮廓 (PDF Geometry)
        </label>
        <label class="switch-label">
          <input type="checkbox" id="chkSkel" checked> 骨架像素点 (1714 点)
        </label>
        <label class="switch-label">
          <input type="checkbox" id="chkControls" checked> 控制柄 (P0, P1, P2)
        </label>
      </div>

      <!-- Interactive Scrubber -->
      <div class="scrubber-card">
        <div class="scrubber-controls">
          <div class="playback-btns">
            <button class="p-btn" id="btnStart" title="回到起点">⏮</button>
            <button class="p-btn" id="btnPrev" title="上一笔">◀</button>
            <button class="p-btn play" id="btnPlay" title="自动播放/暂停">▶</button>
            <button class="p-btn" id="btnNext" title="下一笔">▶</button>
            <button class="p-btn" id="btnEnd" title="跳到最终状态">⏭</button>
          </div>
          <div class="scrubber-info">
            <span id="stepDisplay">步骤 0 / 20</span>
            <span id="stepLabel" style="color: var(--text-dim);">· 初始骨架</span>
          </div>
        </div>

        <div class="slider-track-wrap">
          <input type="range" class="time-slider" id="timeSlider" min="0" max="20" step="1" value="0">
          <div class="keyframe-pills" id="keyframePills">
            <!-- 21 Pills generated dynamically -->
          </div>
        </div>
      </div>
    </div>

    <!-- Right: Annotation Rationale & Diagnostic Deck -->
    <div class="detail-deck">
      <!-- Active Step Thought Card -->
      <div class="thought-card" id="thoughtCard">
        <div class="thought-header">
          <div class="stroke-main-info">
            <h2 id="cardTitle">初始字源骨架形态</h2>
            <div class="sub">
              <span id="cardComponent">所属部件：全字概览</span>
              <span>·</span>
              <span id="cardOrder">时空序号：0 / 20</span>
            </div>
          </div>
          <div class="tool-badge" id="cardTool">
            🔍 全字态势认知
          </div>
        </div>

        <!-- Coverage Meter -->
        <div class="coverage-meter">
          <div class="meter-label-row">
            <span style="color: var(--text-muted);">累计骨架点解释率</span>
            <span id="meterText" style="font-weight: 700; color: #10b981;">0 / 1714 (0.0%)</span>
          </div>
          <div class="meter-bar-track">
            <div class="meter-bar-fill" id="meterFill" style="width: 0%;"></div>
          </div>
        </div>

        <!-- Parametric Specifications -->
        <div class="block-section" id="secParams" style="display: none;">
          <div class="block-title">📐 原语工具数学参数 (Geometric Primitive Spec)</div>
          <div class="param-grid">
            <div class="param-box">
              <span class="key">原语指令 (SVG Path)</span>
              <span class="val" id="valSvgPath">-</span>
            </div>
            <div class="param-box">
              <span class="key">控制点坐标 (Control Points)</span>
              <span class="val" id="valControls">-</span>
            </div>
            <div class="param-box">
              <span class="key">本笔画归属骨架 (Points Claimed)</span>
              <span class="val" id="valPointsClaimed">-</span>
            </div>
            <div class="param-box">
              <span class="key">起点至终点弦向 (Chord Vector)</span>
              <span class="val" id="valChord">-</span>
            </div>
          </div>
        </div>

        <!-- Visual Model Thought Section -->
        <div class="block-section">
          <div class="block-title">👁️ 视觉模型标注思路 (Visual Model Rationale)</div>
          <div class="block-content" id="valVisionLogic">
            『騰』字结构极为繁复（IDS: ⿰月⿱龹馬），包含月字旁、八字头、天部主干、馬字躯干与四点底，共五大微部件、20 笔。视觉模型首先确立左月、右朕的左右拓扑隔离，继而自上而下分别标注八、天、馬、四点，通过贯通的主轴脊柱与折钩边界原语消除碎片化。
          </div>
        </div>

        <!-- Algorithmic Critique Section -->
        <div class="block-section">
          <div class="block-title danger">⚠️ 现有算法问题在哪里 (Why Previous Algorithm Failed Here)</div>
          <div class="block-content critique" id="valAlgoCritique">
            现有算法在『騰』字上遭受最沉重打击：遗留了多达 633 个未解释残差点（占全字 37%）！核心翻车点在于：第 16 笔『馬·横折钩』直接溃缩至 7 像素噪点，丢弃整条右包围框；第 9 笔『天·大撇』被死板边界截断上方半截主脊柱；马部中横、竖画在密集交点处严重缩水。
          </div>
        </div>

        <!-- Ghost Comparison Alert -->
        <div class="ghost-callout" id="ghostAlert">
          <strong>👻 算法翻车对照：</strong>
          <span id="ghostAlertText"></span>
        </div>
      </div>
    </div>

    <!-- Master Stroke Table -->
    <div class="table-card">
      <h3>
        <span>📋 20 笔画全要素标注矩阵 (Master Annotation Matrix)</span>
        <span style="font-size: 12px; font-weight: normal; color: var(--text-dim);">点击表格行可直接跳转对应时空切片</span>
      </h3>
      <table class="stroke-tbl" id="masterTable">
        <thead>
          <tr>
            <th>笔序</th>
            <th>笔画名</th>
            <th>字形特征</th>
            <th>所属部件</th>
            <th>允许工具</th>
            <th>SVG 原语指令</th>
            <th>覆盖骨架点</th>
            <th>单笔占比</th>
            <th>时空角色</th>
          </tr>
        </thead>
        <tbody>
        </tbody>
      </table>
    </div>

    <!-- Global Diagnostic Summary -->
    <div class="summary-card">
      <h3>💡 深入剖析：为什么当前算法在 U+9A30 騰 会暴跌出 633 巨额残差？</h3>
      <div class="summary-grid">
        <div class="summary-item">
          <strong>1. 复杂字源中的“折钩原语溃缩”灾难</strong>
          在繁复部件『馬』中，横折钩本应承担右立柱（x=84.8, y=69~78）与底部左提钩（y=77.7, x=73~85）的整体包围闭合。原算法由于距离图梯度在直角与钩尖之间震荡，陷入局部最小，将整条大折钩原语萎缩压缩成了仅 7 像素的微型噪点，直接放弃了 130+ 像素的整条走廊！
        </div>
        <div class="summary-item">
          <strong>2. 死板模板 Nominal Box 斩断贯通大主笔</strong>
          『龹』字中心的纵向大撇（第 9 笔）从八字头顶部 (y=11.7) 贯穿两横一路滑向西南腹地。原模板将撇的范围限制在 y=26..48，算法不敢向上寻路，导致顶部整段主脊柱（y=11.7~24.6）被当做不可解释的残余孤立点抛弃。
        </div>
        <div class="summary-item">
          <strong>3. 高密部件交织中的“短横缩水”现象</strong>
          馬部三横两竖极度密集（间距仅 7~8px）。旧算法在正交网格搜索中缺乏“贯通横梁”的强先验，导致首横、中横在遇到竖画交点时提前终止（仅搜出 16~19 像素），无法贯通全幅马身。
        </div>
        <div class="summary-item">
          <strong>4. 视觉第一性原理：自顶向下层级布白与硬折钩</strong>
          视觉认知天然懂得：『月』在左侧为独立列柱；『龹』以高位大撇为轴，两横为翼；『馬』为方正外框套三横一竖；『四点』为底座基石。必须以骨架排他性与折线段（Polyline）原语硬锁折钩，才能彻底消灭残差。
        </div>
      </div>
    </div>
  </main>

  <div id="tooltip"></div>

  <script>
    // Embedded Data
    const DATA = {json.dumps(data, ensure_ascii=False)};

    // State
    let currentStep = 0;
    let isPlaying = false;
    let playTimer = null;
    let showGhost = true;
    let zoomLevel = 1.0;

    // DOM Elements
    const mainSvg = document.getElementById("mainSvg");
    const skelLayer = document.getElementById("skelLayer");
    const strokesLayer = document.getElementById("strokesLayer");
    const activeHandlesLayer = document.getElementById("activeHandlesLayer");
    const ghostLayer = document.getElementById("ghostLayer");
    const glyphOutline = document.getElementById("glyphOutline");
    const timeSlider = document.getElementById("timeSlider");
    const keyframePills = document.getElementById("keyframePills");
    const stepDisplay = document.getElementById("stepDisplay");
    const stepLabel = document.getElementById("stepLabel");
    const btnPlay = document.getElementById("btnPlay");
    const btnPrev = document.getElementById("btnPrev");
    const btnNext = document.getElementById("btnNext");
    const btnStart = document.getElementById("btnStart");
    const btnEnd = document.getElementById("btnEnd");
    const btnToggleGhost = document.getElementById("btnToggleGhost");
    const btnZoomIn = document.getElementById("btnZoomIn");
    const btnZoomOut = document.getElementById("btnZoomOut");
    const btnResetZoom = document.getElementById("btnResetZoom");
    const chkOutline = document.getElementById("chkOutline");
    const chkSkel = document.getElementById("chkSkel");
    const chkControls = document.getElementById("chkControls");
    const tooltip = document.getElementById("tooltip");

    // Detail Elements
    const thoughtCard = document.getElementById("thoughtCard");
    const cardTitle = document.getElementById("cardTitle");
    const cardComponent = document.getElementById("cardComponent");
    const cardOrder = document.getElementById("cardOrder");
    const cardTool = document.getElementById("cardTool");
    const meterText = document.getElementById("meterText");
    const meterFill = document.getElementById("meterFill");
    const secParams = document.getElementById("secParams");
    const valSvgPath = document.getElementById("valSvgPath");
    const valControls = document.getElementById("valControls");
    const valPointsClaimed = document.getElementById("valPointsClaimed");
    const valChord = document.getElementById("valChord");
    const valVisionLogic = document.getElementById("valVisionLogic");
    const valAlgoCritique = document.getElementById("valAlgoCritique");
    const ghostAlert = document.getElementById("ghostAlert");
    const ghostAlertText = document.getElementById("ghostAlertText");
    const masterTable = document.getElementById("masterTable").querySelector("tbody");

    // Initialize Skeleton Points in SVG
    const skelCircleElements = [];
    DATA.skel_pixels.forEach((p, idx) => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", p[0]);
      circle.setAttribute("cy", p[1]);
      circle.setAttribute("r", 0.42);
      circle.setAttribute("fill", "rgba(255, 255, 255, 0.16)");
      circle.dataset.idx = idx;
      circle.dataset.x = p[0];
      circle.dataset.y = p[1];
      skelLayer.appendChild(circle);
      skelCircleElements.push(circle);
    }});

    // Build Master Table & Keyframe Pills
    function initUI() {{
      // Pill 0
      const p0 = document.createElement("div");
      p0.className = "k-pill active";
      p0.innerText = "0·全";
      p0.onclick = () => setStep(0);
      keyframePills.appendChild(p0);

      DATA.strokes.forEach((s, idx) => {{
        // Pill
        const pill = document.createElement("div");
        const isHighlight = (idx === 8 || idx === 15);
        pill.className = "k-pill" + (isHighlight ? " special" : "");
        pill.innerText = `${{idx + 1}}·${{s.feature}}`;
        pill.onclick = () => setStep(idx + 1);
        keyframePills.appendChild(pill);

        // Table Row
        const tr = document.createElement("tr");
        tr.dataset.step = idx + 1;
        tr.onclick = () => setStep(idx + 1);
        tr.style.cursor = "pointer";

        const pct = ((s.claimed_indices.length / DATA.total_pts) * 100).toFixed(1);
        tr.innerHTML = `
          <td><span class="dot-badge" style="background:${{s.color}};"></span>#${{idx + 1}}</td>
          <td><strong>${{s.name}}</strong></td>
          <td><span class="tag">${{s.feature}}</span></td>
          <td>${{s.component}}</td>
          <td><code>${{s.tool}}</code></td>
          <td style="font-family:ui-monospace; font-size:11px; color:#38bdf8;">${{s.path_d}}</td>
          <td><strong>${{s.claimed_indices.length}}</strong> 点</td>
          <td>${{pct}}%</td>
          <td style="color:var(--text-muted); font-size:12px;">${{idx === 15 ? "🔥 灾难溃缩笔画" : idx === 8 ? "🌟 高位长贯主笔" : s.component.includes("月") ? "左月立柱" : s.component.includes("馬") ? "馬身框架" : "四点基座"}}</td>
        `;
        masterTable.appendChild(tr);
      }});
    }}

    // Update Display on Step Change
    function setStep(step) {{
      currentStep = step;
      timeSlider.value = step;

      // Update Pill states
      const pills = keyframePills.children;
      for (let i = 0; i < pills.length; i++) {{
        pills[i].classList.toggle("active", i === step);
      }}

      // Update Table row highlight
      const rows = masterTable.querySelectorAll("tr");
      rows.forEach(r => {{
        r.classList.toggle("row-active", parseInt(r.dataset.step) === step);
      }});

      // 1. Update Skeleton Points Coloring
      skelCircleElements.forEach(c => {{
        c.setAttribute("fill", "rgba(255, 255, 255, 0.16)");
        c.setAttribute("r", 0.42);
      }});

      let cumulativeClaimed = 0;
      for (let sIdx = 0; sIdx < step; sIdx++) {{
        const stroke = DATA.strokes[sIdx];
        const isCurrent = (sIdx === step - 1);
        cumulativeClaimed += stroke.claimed_indices.length;

        stroke.claimed_indices.forEach(pIdx => {{
          const circle = skelCircleElements[pIdx];
          if (circle) {{
            circle.setAttribute("fill", stroke.color);
            circle.setAttribute("r", isCurrent ? 0.65 : 0.42);
          }}
        }});
      }}

      // 2. Render Fitted Strokes up to current step
      strokesLayer.innerHTML = "";
      for (let sIdx = 0; sIdx < step; sIdx++) {{
        const stroke = DATA.strokes[sIdx];
        const isCurrent = (sIdx === step - 1);

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", stroke.path_d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", stroke.color);
        path.setAttribute("stroke-width", isCurrent ? 1.8 : 1.2);
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("stroke-linejoin", "round");
        if (isCurrent) {{
          path.classList.add("active-stroke-path");
        }}
        strokesLayer.appendChild(path);
      }}

      // 3. Render Handles for Active Stroke
      activeHandlesLayer.innerHTML = "";
      if (step > 0) {{
        const activeStroke = DATA.strokes[step - 1];
        const ctrls = activeStroke.controls;

        if (activeStroke.primitive_type === "bezier") {{
          const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
          polyline.setAttribute("points", ctrls.map(p => p.join(",")).join(" "));
          polyline.setAttribute("fill", "none");
          polyline.setAttribute("stroke", "rgba(255, 255, 255, 0.3)");
          polyline.setAttribute("stroke-width", 0.5);
          polyline.setAttribute("stroke-dasharray", "1,1");
          activeHandlesLayer.appendChild(polyline);
        }}

        ctrls.forEach((pt, i) => {{
          const handle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          handle.setAttribute("cx", pt[0]);
          handle.setAttribute("cy", pt[1]);
          handle.setAttribute("r", i === 1 && activeStroke.primitive_type === "bezier" ? 1.1 : 0.9);
          handle.setAttribute("fill", i === 0 ? "#10b981" : i === ctrls.length - 1 ? "#ef4444" : "#38bdf8");
          handle.setAttribute("stroke", "white");
          handle.setAttribute("stroke-width", 0.35);
          activeHandlesLayer.appendChild(handle);
        }});
      }}

      // 4. Render Ghost Layer (Old Algorithm comparison)
      ghostLayer.innerHTML = "";
      if (showGhost) {{
        if (step === 16 || step === 0) {{
          // Stroke 16 ghost (Stroke 15 collapse)
          const ghostPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
          ghostPath.setAttribute("d", "M 84.8,70.0 L 84.8,71.5");
          ghostPath.setAttribute("fill", "none");
          ghostPath.setAttribute("stroke", "rgba(244, 63, 94, 0.95)");
          ghostPath.setAttribute("stroke-width", 2.2);
          ghostPath.setAttribute("stroke-dasharray", "1,1");
          ghostLayer.appendChild(ghostPath);

          const ghostText = document.createElementNS("http://www.w3.org/2000/svg", "text");
          ghostText.setAttribute("x", 86);
          ghostText.setAttribute("y", 73);
          ghostText.setAttribute("fill", "#fda4af");
          ghostText.setAttribute("font-size", "2.6");
          ghostText.setAttribute("font-weight", "bold");
          ghostText.textContent = "⚠ 旧算法仅认领 7 像素 (溃缩)";
          ghostLayer.appendChild(ghostText);
        }}
        if (step === 9 || step === 0) {{
          // Stroke 9 ghost (Stroke 8 truncation)
          const ghostPath2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
          ghostPath2.setAttribute("d", "M 55.0,26.5 Q 52.0,38.0 45.7,48.0");
          ghostPath2.setAttribute("fill", "none");
          ghostPath2.setAttribute("stroke", "rgba(244, 63, 94, 0.7)");
          ghostPath2.setAttribute("stroke-width", 1.5);
          ghostPath2.setAttribute("stroke-dasharray", "2,1");
          ghostLayer.appendChild(ghostPath2);
        }}
      }}

      // 5. Update Information Deck
      if (step === 0) {{
        stepDisplay.innerText = "步骤 0 / 20";
        stepLabel.innerText = "· 初始骨架形态";
        thoughtCard.style.setProperty("--active-stroke-color", "var(--accent)");

        cardTitle.innerText = "初始字源骨架形态与全景拓扑";
        cardComponent.innerText = "所属部件：全字概览 (IDS: ⿰月⿱龹馬)";
        cardOrder.innerText = "时空序号：0 / 20";
        cardTool.innerText = "🔍 全字态势认知";
        cardTool.style.borderColor = "var(--border-subtle)";

        secParams.style.display = "none";
        meterText.innerText = "0 / 1714 点 (0.0%)";
        meterFill.style.width = "0%";

        valVisionLogic.innerText = "『騰』字作为繁复高密字源，骨架点多达 1,714 像素。视觉模型将其划分为左月（4 笔）、右上八（2 笔）、右中天（4 笔）、右下马（6 笔）和四点底（4 笔）。视觉认知依托全局时空序列，逐级消除歧义，防止黑盒优化器在密集交织区域发生溃缩。";
        valAlgoCritique.innerText = "旧算法在『騰』字上产生 633 巨额残余！最致命病灶在第 16 笔『馬·横折钩』直接溃缩至 7 像素，整条立柱与底钩裸露；天部长撇上方主脊柱断头；馬部三横两竖短路缩水。";
        ghostAlert.classList.remove("show");
      }} else {{
        const curStroke = DATA.strokes[step - 1];
        stepDisplay.innerText = `步骤 ${{step}} / 20`;
        stepLabel.innerText = `· ${{curStroke.name}} (${{curStroke.feature}})`;
        thoughtCard.style.setProperty("--active-stroke-color", curStroke.color);

        cardTitle.innerHTML = `${{curStroke.name}} <span class="tag" style="background:${{curStroke.color}}22; color:${{curStroke.color}}; border-color:${{curStroke.color}}55;">${{curStroke.feature}}</span>`;
        cardComponent.innerText = `所属部件：${{curStroke.component}}`;
        cardOrder.innerText = `时空序号：${{step}} / 20`;
        cardTool.innerHTML = `🛠️ 允许工具：<strong>${{curStroke.tool}}</strong>`;
        cardTool.style.borderColor = curStroke.color + "66";

        secParams.style.display = "flex";
        valSvgPath.innerText = curStroke.path_d;
        valControls.innerText = JSON.stringify(curStroke.controls);
        valPointsClaimed.innerText = `${{curStroke.claimed_indices.length}} 点 (${{((curStroke.claimed_indices.length/DATA.total_pts)*100).toFixed(1)}}%)`;

        const p0 = curStroke.controls[0];
        const pEnd = curStroke.controls[curStroke.controls.length - 1];
        const dx = (pEnd[0] - p0[0]).toFixed(1);
        const dy = (pEnd[1] - p0[1]).toFixed(1);
        valChord.innerText = `Δx = ${{dx}}, Δy = ${{dy}} (弦向角: ${{(Math.atan2(dy, dx) * 180 / Math.PI).toFixed(0)}}°)`;

        const cumPct = ((cumulativeClaimed / DATA.total_pts) * 100).toFixed(1);
        meterText.innerText = `${{cumulativeClaimed}} / ${{DATA.total_pts}} 点 (${{cumPct}}%)`;
        meterFill.style.width = `${{cumPct}}%`;

        valVisionLogic.innerText = curStroke.vision_logic;
        valAlgoCritique.innerText = curStroke.algorithm_critique;

        if (step === 16) {{
          ghostAlert.classList.add("show");
          ghostAlertText.innerText = DATA.old_algo_failures[0].failure_summary;
        }} else if (step === 9) {{
          ghostAlert.classList.add("show");
          ghostAlertText.innerText = DATA.old_algo_failures[1].failure_summary;
        }} else {{
          ghostAlert.classList.remove("show");
        }}
      }}
    }}

    // Playback Logic
    function togglePlay() {{
      if (isPlaying) {{
        pause();
      }} else {{
        play();
      }}
    }}

    function play() {{
      isPlaying = true;
      btnPlay.innerText = "⏸";
      btnPlay.title = "暂停";
      if (currentStep >= 20) setStep(0);

      playTimer = setInterval(() => {{
        if (currentStep < 20) {{
          setStep(currentStep + 1);
        }} else {{
          pause();
        }}
      }}, 1300);
    }}

    function pause() {{
      isPlaying = false;
      btnPlay.innerText = "▶";
      btnPlay.title = "自动播放";
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
      }}
    }}

    // Event Listeners
    timeSlider.oninput = (e) => {{
      pause();
      setStep(parseInt(e.target.value));
    }};

    btnPlay.onclick = togglePlay;
    btnStart.onclick = () => {{ pause(); setStep(0); }};
    btnEnd.onclick = () => {{ pause(); setStep(20); }};
    btnPrev.onclick = () => {{ pause(); if (currentStep > 0) setStep(currentStep - 1); }};
    btnNext.onclick = () => {{ pause(); if (currentStep < 20) setStep(currentStep + 1); }};

    btnToggleGhost.onclick = () => {{
      showGhost = !showGhost;
      btnToggleGhost.classList.toggle("active", showGhost);
      setStep(currentStep);
    }};

    chkOutline.onchange = (e) => {{
      glyphOutline.style.display = e.target.checked ? "block" : "none";
    }};

    chkSkel.onchange = (e) => {{
      skelLayer.style.display = e.target.checked ? "block" : "none";
    }};

    chkControls.onchange = (e) => {{
      activeHandlesLayer.style.display = e.target.checked ? "block" : "none";
    }};

    btnZoomIn.onclick = () => {{
      zoomLevel = Math.min(zoomLevel + 0.25, 2.5);
      mainSvg.style.transform = `scale(${{zoomLevel}})`;
    }};

    btnZoomOut.onclick = () => {{
      zoomLevel = Math.max(zoomLevel - 0.25, 0.75);
      mainSvg.style.transform = `scale(${{zoomLevel}})`;
    }};

    btnResetZoom.onclick = () => {{
      zoomLevel = 1.0;
      mainSvg.style.transform = "scale(1)";
    }};

    // Tooltip
    skelLayer.onmousemove = (e) => {{
      const target = e.target;
      if (target.tagName === "circle") {{
        const idx = target.dataset.idx;
        const x = target.dataset.x;
        const y = target.dataset.y;
        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX + 14) + "px";
        tooltip.style.top = (e.clientY + 14) + "px";
        tooltip.innerHTML = `<strong>骨架像素点 #${{idx}}</strong><br>坐标: (${{x}}, ${{y}})`;
      }}
    }};

    skelLayer.onmouseleave = () => {{
      tooltip.style.display = "none";
    }};

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {{
      if (e.key === "ArrowRight") {{
        pause();
        if (currentStep < 20) setStep(currentStep + 1);
      }} else if (e.key === "ArrowLeft") {{
        pause();
        if (currentStep > 0) setStep(currentStep - 1);
      }} else if (e.key === " ") {{
        e.preventDefault();
        togglePlay();
      }}
    }});

    // Init
    initUI();
    setStep(0);
  </script>
</body>
</html>
"""

out_html = Path("artifacts/deterministic-decomposer/stroke-annotation-u9a30-k.html")
out_html.write_text(html_content, encoding="utf-8")
print(f"Interactive HTML for U+9A30 K generated successfully at: {out_html}")
