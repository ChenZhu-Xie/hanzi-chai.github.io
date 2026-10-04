"""Build a single-tab gallery from directed ink trajectory audit JSON files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def severity(stroke: dict) -> float:
    return (
        float(stroke["directedSequenceDtwMeanPercent"])
        + float(stroke["startErrorPercent"])
        + float(stroke["endErrorPercent"])
        + (1.0 - float(stroke["inkPrecision"])) * 10.0
        + (1.0 - float(stroke["inkRecall"])) * 10.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default", default="U+64CE-T")
    args = parser.parse_args()

    cases = []
    for path in sorted(args.audits.glob("*.json")):
        document = json.loads(path.read_text("utf-8"))
        evaluation = document.get("evaluation") or {}
        strokes = evaluation.get("perStrokeTrajectoryAudit") or []
        worst = sorted(strokes, key=severity, reverse=True)[:4]
        cases.append(
            {
                "key": f"{document['unicode']}-{document['source']}",
                "unicode": document["unicode"],
                "character": document["character"],
                "source": document["source"],
                "glyphId": document["glyphId"],
                "status": document["decision"]["status"],
                "dtw": evaluation.get("meanDirectedSequenceDtwPercent"),
                "start": evaluation.get("meanStartErrorPercent"),
                "end": evaluation.get("meanEndErrorPercent"),
                "strokeIou": evaluation.get("strokeMacroIoU"),
                "componentIou": evaluation.get("componentMacroIoU"),
                "issues": evaluation.get("issueCounts", {}),
                "worst": worst,
                "page": Path(document["output"]).name,
            }
        )
    default_index = next(
        (index for index, case in enumerate(cases) if case["key"] == args.default),
        0,
    )
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐笔墨迹轨迹审计</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#e8eef5;color:#172033;font:13px/1.35 "Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden}}header{{height:72px;padding:10px 16px;background:#0f172a;color:white;display:flex;align-items:center;justify-content:space-between;gap:18px}}h1{{font-size:19px;margin:0 0 3px}}header p{{margin:0;color:#cbd5e1}}.legend{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}.pill{{padding:4px 8px;border-radius:999px;background:#334155}}.layout{{height:calc(100vh - 72px);display:grid;grid-template-columns:410px 1fr;gap:10px;padding:10px}}aside,.viewer{{background:white;border:1px solid #cbd5e1;border-radius:10px;overflow:hidden}}aside{{overflow:auto;padding:8px}}button.case{{display:block;width:100%;text-align:left;border:1px solid #cbd5e1;border-left:7px solid var(--accent);border-radius:8px;background:white;padding:7px 8px;margin-bottom:7px;cursor:pointer}}button.case:hover{{background:#f8fafc}}button.case.active{{outline:3px solid #38bdf8;background:#f0f9ff}}.case-title{{display:flex;justify-content:space-between;font-weight:700}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:2px 8px;margin-top:4px;color:#475569}}.worst{{margin-top:5px;color:#7c2d12;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.viewer{{display:grid;grid-template-rows:auto 1fr}}.toolbar{{padding:7px 10px;border-bottom:1px solid #cbd5e1;background:#f8fafc}}.toolbar-top{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.toolbar a{{color:#0369a1}}table{{width:100%;border-collapse:collapse;margin-top:6px;font-size:11px}}th,td{{text-align:left;border-top:1px solid #e2e8f0;padding:3px 5px}}iframe{{border:0;width:100%;height:100%}}@media(max-width:1000px){{body{{overflow:auto}}.layout{{height:auto;grid-template-columns:1fr}}aside{{max-height:45vh}}.viewer{{height:95vh}}}}
</style>
<header><div><h1>当前墨迹模型 × 人工有向轨迹</h1><p>红点＝模型笔尖；青点＝人工笔尖。两者按单调书写逻辑 DTW 对齐，不假设绘制速度或同百分比时间相等。</p></div><div class="legend"><span class="pill">起/收笔</span><span class="pill">转折</span><span class="pill">丁/十字节点</span><span class="pill">覆盖 P/R</span><span class="pill">部件边界</span></div></header>
<div class="layout"><aside id="cases"></aside><section class="viewer"><div class="toolbar"><div class="toolbar-top"><strong id="selected"></strong><a id="standalone" target="_blank">独立标签页打开</a></div><table><thead><tr><th>高风险笔</th><th>部件</th><th>起点</th><th>终点</th><th>DTW</th><th>P/R</th><th>诊断</th></tr></thead><tbody id="worst"></tbody></table></div><iframe id="frame" title="逐笔墨迹轨迹"></iframe></section></div>
<script>const cases={payload},defaultIndex={default_index},list=document.querySelector('#cases'),frame=document.querySelector('#frame'),selected=document.querySelector('#selected'),standalone=document.querySelector('#standalone'),worst=document.querySelector('#worst');
const pct=value=>value==null?'—':(Number(value)*100).toFixed(1)+'%';const num=value=>value==null?'—':Number(value).toFixed(2)+'%';
function selectCase(index){{document.querySelectorAll('.case').forEach((element,other)=>element.classList.toggle('active',other===index));const item=cases[index],url=item.page+'?time=0';frame.src=url;standalone.href=url;selected.textContent=`${{item.unicode}}「${{item.character}}」${{item.source}}源 · glyph ${{item.glyphId}} · ${{item.status}}`;worst.innerHTML=item.worst.map(stroke=>`<tr><td>第 ${{stroke.stroke}} 笔 ${{stroke.feature}}</td><td>${{stroke.componentId}}</td><td>${{stroke.startErrorPercent.toFixed(1)}}%</td><td>${{stroke.endErrorPercent.toFixed(1)}}%</td><td>${{stroke.directedSequenceDtwMeanPercent.toFixed(1)}}%</td><td>${{(stroke.inkPrecision*100).toFixed(0)}}/${{(stroke.inkRecall*100).toFixed(0)}}</td><td>${{stroke.issues.join('、')||'—'}}</td></tr>`).join('')}}
for(const [index,item] of cases.entries()){{const safe=item.status==='safe-candidate',button=document.createElement('button');button.className='case';button.style.setProperty('--accent',safe?'#16a34a':'#dc2626');const issueSummary=Object.entries(item.issues).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([key,count])=>`${{key}}×${{count}}`).join(' · ');button.innerHTML=`<div class="case-title"><span>${{item.unicode}}「${{item.character}}」${{item.source}} · ${{item.glyphId}}</span><span>${{item.status}}</span></div><div class="metrics"><span>DTW ${{num(item.dtw)}}</span><span>逐笔 IoU ${{pct(item.strokeIou)}}</span><span>部件 IoU ${{pct(item.componentIou)}}</span><span>起点 ${{num(item.start)}}</span><span>终点 ${{num(item.end)}}</span></div><div class="worst" title="${{issueSummary}}">${{issueSummary}}</div>`;button.onclick=()=>selectCase(index);list.append(button)}}selectCase(defaultIndex);</script></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, "utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
