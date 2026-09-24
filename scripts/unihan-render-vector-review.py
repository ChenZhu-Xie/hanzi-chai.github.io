"""Build a self-contained, all-vector PDF/candidate review page.

The PDF glyph stays an unmodified outline.  Candidate-colored PDF copies are
explicit hypotheses: repository leaf paths act as SVG masks, never as labels
that are claimed to have been extracted from the PDF.
"""

from __future__ import annotations

import argparse
import copy
import html
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_module("unihan-render-match.py", "unihan_render_match_vector")
PDF = MATCHER.PDF


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def href(element: ET.Element) -> str | None:
    return element.get("href") or element.get(f"{{{XLINK_NS}}}href")


def _number(element: ET.Element, name: str) -> float | None:
    value = element.get(name)
    if value is None:
        return None
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    return float(match.group(0)) if match else None


def select_chart_use(page_svg: str, bbox: list[float]) -> ET.Element:
    """Select the placed font glyph whose origin lies in the chart glyph box."""
    root = ET.fromstring(page_svg)
    # A placed font glyph is positioned by its baseline, which can sit below
    # the visible-ink crop used by the raster matcher.  The full cell still
    # excludes the source-reference text on the following baseline.
    left, top, right, bottom = bbox
    matches = []
    for element in root.iter():
        if local_name(element.tag) != "use" or not href(element):
            continue
        x = _number(element, "x")
        y = _number(element, "y")
        if x is None or y is None:
            continue
        if left - 0.05 <= x <= right + 0.05 and top - 0.05 <= y <= bottom + 0.05:
            matches.append(element)
    if len(matches) != 1:
        coordinates = [
            (_number(item, "x"), _number(item, "y"), href(item)) for item in matches
        ]
        raise ValueError(
            f"expected one chart glyph use, found {len(matches)}: {coordinates}"
        )
    return matches[0]


def _referenced_definitions(root: ET.Element, first_id: str) -> list[ET.Element]:
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
    pending = [first_id]
    found = []
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        element = by_id.get(current)
        if element is None:
            raise ValueError(f"missing PDF SVG definition #{current}")
        found.append(copy.deepcopy(element))
        for child in element.iter():
            reference = href(child)
            if reference and reference.startswith("#"):
                pending.append(reference[1:])
            for value in child.attrib.values():
                pending.extend(re.findall(r"url\(#([^)]+)\)", value))
    return found


def extract_pdf_glyph(page_svg: str, bbox: list[float], prefix: str) -> dict:
    """Return only the selected placed glyph and its recursively used defs."""
    root = ET.fromstring(page_svg)
    placed = select_chart_use(page_svg, bbox)
    reference = href(placed)
    if not reference or not reference.startswith("#"):
        raise ValueError("selected PDF glyph has no local symbol reference")
    definitions = _referenced_definitions(root, reference[1:])
    id_map = {
        element.get("id"): f"{prefix}-{element.get('id')}"
        for definition in definitions
        for element in definition.iter()
        if element.get("id")
    }
    for definition in definitions:
        for element in definition.iter():
            old_id = element.get("id")
            if old_id in id_map:
                element.set("id", id_map[old_id])
            for attribute in ("href", f"{{{XLINK_NS}}}href"):
                value = element.get(attribute)
                if value and value.startswith("#") and value[1:] in id_map:
                    element.set(attribute, f"#{id_map[value[1:]]}")
            for key, value in list(element.attrib.items()):
                element.set(
                    key,
                    re.sub(
                        r"url\(#([^)]+)\)",
                        lambda match: (
                            f"url(#{id_map.get(match.group(1), match.group(1))})"
                        ),
                        value,
                    ),
                )
    use_attributes = {
        key: value
        for key, value in placed.attrib.items()
        if local_name(key) not in {"href"}
    }
    use_attributes["href"] = f"#{id_map[reference[1:]]}"
    return {
        "definitions": "".join(
            ET.tostring(item, encoding="unicode") for item in definitions
        ),
        "useAttributes": use_attributes,
        "pathCount": sum(
            1
            for item in definitions
            for element in item.iter()
            if local_name(element.tag) == "path"
        ),
    }


def attributes_markup(attributes: dict[str, str]) -> str:
    return " ".join(
        f'{html.escape(local_name(key))}="{html.escape(str(value), quote=True)}"'
        for key, value in attributes.items()
    )


def stroke_boundaries(stroke: dict) -> list[list[float]]:
    current = [float(value) for value in stroke["start"]]
    points = [current.copy()]
    for curve in stroke["curveList"]:
        command = curve["command"]
        values = curve["parameterList"]
        if command == "h":
            current[0] += values[0]
        elif command == "v":
            current[1] += values[0]
        elif command != "a" and len(values) >= 2:
            current[0] += values[-2]
            current[1] += values[-1]
        points.append(current.copy())
    return points


def leaf_paths(
    entry: dict,
    *,
    stroke: str | None = None,
    width: float | None = None,
    strokes: list[dict] | None = None,
) -> str:
    root = ET.fromstring(entry["svg"])
    output = []
    path_index = 0
    for element in root.iter():
        if local_name(element.tag) != "path":
            continue
        path = copy.deepcopy(element)
        path.tag = "path"
        if stroke is not None:
            path.set("stroke", stroke)
        if width is not None:
            path.set("stroke-width", str(width))
        if strokes is not None:
            stroke_index = entry["strokeIndices"][path_index]
            path.set("data-stroke-index", str(stroke_index))
            path.set(
                "data-boundaries",
                json.dumps(
                    stroke_boundaries(strokes[stroke_index]), separators=(",", ":")
                ),
            )
        output.append(ET.tostring(path, encoding="unicode", short_empty_elements=True))
        path_index += 1
    return "".join(output)


def candidate_group(entries: list[dict], candidate_id: int, strokes: list[dict]) -> str:
    groups = []
    for entry in entries:
        label = f"部件 {entry['leafId']} · occurrence {entry['occurrence']} · strokes {','.join(map(str, entry['strokeIndices']))}"
        groups.append(
            f'''<g class="leaf" data-component-id="{entry["leafId"]}" data-family-key="{html.escape(str(entry["familyKey"]))}" data-occurrence="{entry["occurrence"]}" data-label="{html.escape(label, quote=True)}" style="--leaf:{entry["color"]}">
              <title>{html.escape(label)}</title>{leaf_paths(entry, strokes=strokes)}</g>'''
        )
    return f'<g class="candidate-raw" data-candidate-id="{candidate_id}">{"".join(groups)}</g>'


def source_use(glyph: dict, css_class: str) -> str:
    return f'<use class="{css_class}" {attributes_markup(glyph["useAttributes"])} />'


def vector_panel(title: str, body: str, css_class: str = "") -> str:
    return f'''<article class="panel {css_class}"><h4>{html.escape(title)}</h4>
      <svg viewBox="0 0 100 100" role="img" aria-label="{html.escape(title, quote=True)}">{body}</svg></article>'''


def candidate_panels(
    glyph: dict,
    candidate_id: int,
    entries: list[dict],
    strokes: list[dict],
    letter: str,
    case_key: str,
) -> str:
    raw = candidate_group(entries, candidate_id, strokes)
    masks = []
    colored_source = []
    for index, entry in enumerate(entries):
        mask_id = f"{case_key}-mask-{candidate_id}-{index}"
        masks.append(
            f'''<mask id="{mask_id}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">
              <rect width="100" height="100" fill="black"/><g class="mask-fit">{leaf_paths(entry, stroke="white", width=13)}</g></mask>'''
        )
        colored_source.append(
            f'''<g class="hyp-leaf" data-component-id="{entry["leafId"]}" data-family-key="{html.escape(str(entry["familyKey"]))}" data-occurrence="{entry["occurrence"]}" data-label="PDF → Candidate {letter} 假设 · 部件 {entry["leafId"]}" style="--leaf:{entry["color"]}" mask="url(#{mask_id})">
              <g class="source-fit" fill="{entry["color"]}">{source_use(glyph, "source-use")}</g></g>'''
        )
    hypothesis = f"""<defs>{"".join(masks)}</defs>
      <g class="source-fit source-underlay" fill="#cbd5e1">{source_use(glyph, "source-use")}</g>{"".join(colored_source)}
      <g class="node-layer projected-nodes"></g>"""
    candidate = f"""<g class="candidate-stage">{raw}</g><g class="node-layer candidate-nodes"></g>"""
    return vector_panel(
        f"PDF → {letter} 假设", hypothesis, "hypothesis"
    ) + vector_panel(
        f"Candidate {letter} · glyph {candidate_id}", candidate, "candidate"
    )


def render_case(record: dict, row: dict, result: dict, glyph: dict, index: int) -> str:
    ids = [int(value) for value in row["candidates"]]
    case_key = f"u{record['unicode']:04x}-{record['source'].lower()}"
    original = f"""<g class="source-fit original-source" fill="#111827">{source_use(glyph, "source-use")}</g>
      <g class="outline-note">PDF outline · {glyph["pathCount"]} path(s)</g>"""
    panels = [
        vector_panel(f"PDF {record['source']} 原始轮廓", original, "source-original")
    ]
    legends = []
    for candidate_index, candidate_id in enumerate(ids):
        letter = chr(65 + candidate_index)
        entries = row["candidateLeafSvgs"][str(candidate_id)]
        panels.append(
            candidate_panels(
                glyph,
                candidate_id,
                entries,
                row["candidates"][str(candidate_id)],
                letter,
                case_key,
            )
        )
        leaves = " ".join(
            f'<span style="--leaf:{entry["color"]}"><i></i>{entry["leafId"]}</span>'
            for entry in entries
        )
        legends.append(f"<div><b>{letter} · {candidate_id}</b> {leaves}</div>")
    expected = result.get("expectedGlyphId")
    return f'''<section class="case" id="{case_key}" data-case-index="{index}">
      <svg class="case-definitions" aria-hidden="true"><defs>{glyph["definitions"]}</defs></svg>
      <header><div><h2>U+{record["unicode"]:04X} {chr(record["unicode"])} · {record["source"]} 源</h2>
      <p>人工答案：{expected if expected is not None else "盲测"}；彩色 PDF 是候选驱动的解释假设，不是 PDF 自带部件标签。</p></div>
      <button class="explode-button" type="button">分离叶部件</button></header>
      <div class="panels">{"".join(panels)}</div><div class="leaf-legend">{"".join(legends)}</div></section>'''


CSS = r"""
:root{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;color:#172033;background:#eef2f7}*{box-sizing:border-box}
body{margin:0}body>header{position:sticky;top:0;z-index:10;padding:14px 22px;background:#0f172aeF;color:white;backdrop-filter:blur(8px)}
h1,h2,h4,p{margin:.2rem 0}body>header p,.case p{font-size:13px;color:#cbd5e1}.toolbar{display:flex;gap:18px;flex-wrap:wrap;margin-top:9px;font-size:13px}
main{padding:18px;display:grid;gap:18px}.case{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:14px;box-shadow:0 7px 22px #0f172a12}
.case-definitions{position:absolute;width:0;height:0;overflow:hidden}
.case>header{display:flex;justify-content:space-between;align-items:start;gap:12px}.case p{color:#64748b}.explode-button{padding:7px 11px;border:1px solid #94a3b8;background:#f8fafc;border-radius:8px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-top:12px}.panel{min-width:0;border:1px solid #dbe3ed;border-radius:10px;background:#fbfdff;overflow:hidden}.panel h4{padding:7px 9px;font-size:13px;background:#f1f5f9;border-bottom:1px solid #dbe3ed}.panel svg{display:block;width:100%;aspect-ratio:1;background:white;overflow:visible}
.leaf path{stroke:var(--leaf)!important;vector-effect:non-scaling-stroke;transition:stroke-width .12s,filter .12s}.leaf:hover path,.leaf.is-hover path{stroke-width:6!important;filter:drop-shadow(0 0 1.5px white)}
.source-underlay{opacity:.48}.hyp-leaf{fill:var(--leaf);transition:filter .12s}.hyp-leaf:hover,.hyp-leaf.is-hover{filter:drop-shadow(0 0 1.3px #111827)}
.node.endpoint{fill:#dc2626;stroke:white;stroke-width:.55}.node.junction{fill:#0284c7;stroke:white;stroke-width:.55}.node.corner{fill:none;stroke:#22c55e;stroke-width:1.2;stroke-linecap:round}.case.exploded .junction-global{display:none}
.outline-note{font-size:3px;fill:#64748b}.leaf-legend{display:grid;gap:5px;margin-top:10px;font-size:12px}.leaf-legend b{display:inline-block;min-width:96px}.leaf-legend span{display:inline-flex;align-items:center;margin-right:9px}.leaf-legend i{width:10px;height:10px;border-radius:50%;background:var(--leaf);margin-right:3px}
#tooltip{position:fixed;display:none;z-index:99;pointer-events:none;max-width:330px;padding:7px 9px;border-radius:7px;background:#0f172a;color:white;font-size:12px;box-shadow:0 5px 18px #0005}
@media(max-width:700px){.panels{grid-template-columns:1fr 1fr}body>header{position:static}}
"""


JS = r"""
const ns='http://www.w3.org/2000/svg';
function fitMatrix(el,pad=8){const b=el.getBBox(),s=Math.min((100-pad*2)/b.width,(100-pad*2)/b.height);return {s,tx:50-s*(b.x+b.width/2),ty:50-s*(b.y+b.height/2)};}
function matrixText(m){return `matrix(${m.s} 0 0 ${m.s} ${m.tx} ${m.ty})`;}
function localPoint(svg,path,p){const pathMatrix=path.getScreenCTM(),svgMatrix=svg.getScreenCTM();if(!pathMatrix||!svgMatrix)return {x:NaN,y:NaN};return new DOMPoint(p.x,p.y).matrixTransform(pathMatrix).matrixTransform(svgMatrix.inverse());}
function svgPoint(svg,path,length){const p=path.getPointAtLength(length);return localPoint(svg,path,p);}
function addCircle(layer,p,kind){if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return;const c=document.createElementNS(ns,'circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',kind==='junction'?1.25:1.05);c.setAttribute('class',`node ${kind}`);layer.append(c);}
function addCorner(layer,p){if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return;const g=document.createElementNS(ns,'g');g.setAttribute('class','node corner');g.innerHTML=`<path d="M ${p.x-1.1} ${p.y-1.1} L ${p.x+1.1} ${p.y+1.1} M ${p.x+1.1} ${p.y-1.1} L ${p.x-1.1} ${p.y+1.1}"/>`;layer.append(g);}
function cluster(points,r=1.55){const out=[];for(const p of points){if(!Number.isFinite(p.x)||!Number.isFinite(p.y))continue;let q=out.find(x=>Math.hypot(x.x-p.x,x.y-p.y)<r);if(q){q.x=(q.x*q.n+p.x)/(q.n+1);q.y=(q.y*q.n+p.y)/(q.n+1);q.n++;}else out.push({x:p.x,y:p.y,n:1});}return out;}
function segmentIntersection(a,b,c,d){const rx=b.x-a.x,ry=b.y-a.y,sx=d.x-c.x,sy=d.y-c.y,den=rx*sy-ry*sx;if(Math.abs(den)<1e-6)return null;const qx=c.x-a.x,qy=c.y-a.y,t=(qx*sy-qy*sx)/den,u=(qx*ry-qy*rx)/den;return t>=0&&t<=1&&u>=0&&u<=1?{x:a.x+t*rx,y:a.y+t*ry}:null;}
function pointSegmentDistance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y,l2=dx*dx+dy*dy;if(!l2)return Math.hypot(p.x-a.x,p.y-a.y);const t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/l2));return Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);}
function markCandidate(svg){const layer=svg.querySelector('.candidate-nodes');if(!layer)return;const leaves=[...svg.querySelectorAll('.leaf')],ownerLayers=leaves.map((leaf,index)=>{const group=document.createElementNS(ns,'g');group.setAttribute('class','leaf-node-group');group.dataset.leafIndex=index;group.dataset.componentId=leaf.dataset.componentId;group.dataset.occurrence=leaf.dataset.occurrence;layer.append(group);return group;}),junctionLayer=document.createElementNS(ns,'g');junctionLayer.setAttribute('class','junction-global');layer.append(junctionLayer);const paths=[...svg.querySelectorAll('.leaf path')],polylines=[],allEndpoints=[];let cornerCount=0;
  paths.forEach((path,pi)=>{const owner=leaves.indexOf(path.closest('.leaf')),L=path.getTotalLength(),points=[];for(let i=0;i<=72;i++)points.push(svgPoint(svg,path,L*i/72));polylines.push(points);allEndpoints.push({x:points[0].x,y:points[0].y,pi},{x:points.at(-1).x,y:points.at(-1).y,pi});addCircle(ownerLayers[owner],points[0],'endpoint');addCircle(ownerLayers[owner],points.at(-1),'endpoint');
    const boundaries=JSON.parse(path.dataset.boundaries||'[]').map(([x,y])=>localPoint(svg,path,{x,y})),corners=[];for(let i=1;i<boundaries.length-1;i++){const a=boundaries[i-1],p=boundaries[i],b=boundaries[i+1],v1=Math.atan2(p.y-a.y,p.x-a.x),v2=Math.atan2(b.y-p.y,b.x-p.x),angle=Math.abs(v2-v1),turn=Math.min(angle,Math.PI*2-angle);if(turn>.55)corners.push(p);}const clustered=cluster(corners,3.2);clustered.forEach(p=>addCorner(ownerLayers[owner],p));cornerCount+=clustered.length;
  });const contacts=[];for(let i=0;i<polylines.length;i++)for(let j=i+1;j<polylines.length;j++){const a=polylines[i],b=polylines[j];for(let ai=0;ai<a.length-1;ai++)for(let bi=0;bi<b.length-1;bi++){const hit=segmentIntersection(a[ai],a[ai+1],b[bi],b[bi+1]);if(hit)contacts.push(hit);}}
  for(const endpoint of allEndpoints)for(let pi=0;pi<polylines.length;pi++)if(pi!==endpoint.pi){const points=polylines[pi];for(let i=0;i<points.length-1;i++)if(pointSegmentDistance(endpoint,points[i],points[i+1])<1.15){contacts.push(endpoint);break;}}
  const junctions=cluster(contacts,2.2);junctions.forEach(p=>addCircle(junctionLayer,p,'junction'));layer.dataset.endpointCount=allEndpoints.length;layer.dataset.junctionCount=junctions.length;layer.dataset.cornerCount=cornerCount;
}
function setupCase(section){if(section.dataset.ready)return;section.dataset.ready='true';const original=section.querySelector('.source-original .source-fit'),sourceM=fitMatrix(original);section.querySelectorAll('.source-fit').forEach(x=>x.setAttribute('transform',matrixText(sourceM)));
  section.querySelectorAll('.candidate').forEach(panel=>{const raw=panel.querySelector('.candidate-raw'),m=fitMatrix(raw);panel.dataset.matrix=JSON.stringify(m);raw.setAttribute('transform',matrixText(m));markCandidate(panel.querySelector('svg'));
    const hypothesis=panel.previousElementSibling;if(hypothesis?.classList.contains('hypothesis')){hypothesis.querySelectorAll('.mask-fit').forEach(x=>x.setAttribute('transform',matrixText(m)));const projected=hypothesis.querySelector('.projected-nodes');panel.querySelectorAll('.candidate-nodes>*').forEach(node=>projected.append(node.cloneNode(true)));}
  });
  section.querySelector('.explode-button').addEventListener('click',event=>{const on=section.classList.toggle('exploded');event.currentTarget.textContent=on?'合拢叶部件':'分离叶部件';section.querySelectorAll('.candidate').forEach(panel=>{const hypothesis=panel.previousElementSibling,leaves=[...panel.querySelectorAll('.leaf')],hypothesisLeaves=[...hypothesis.querySelectorAll('.hyp-leaf')],candidateNodes=[...panel.querySelectorAll('.leaf-node-group')],projectedNodes=[...hypothesis.querySelectorAll('.leaf-node-group')];leaves.forEach((leaf,index)=>{if(!leaf.dataset.explode){const b=leaf.getBBox(),dx=b.x+b.width/2-50,dy=b.y+b.height/2-50,n=Math.hypot(dx,dy)||1;leaf.dataset.explode=`translate(${dx/n*7} ${dy/n*7})`;const markerTransform=`translate(${dx/n*6} ${dy/n*6})`;hypothesisLeaves[index].dataset.explode=markerTransform;candidateNodes[index].dataset.explode=markerTransform;projectedNodes[index].dataset.explode=markerTransform;}leaf.setAttribute('transform',on?leaf.dataset.explode:'');for(const group of [hypothesisLeaves[index],candidateNodes[index],projectedNodes[index]])group.setAttribute('transform',on?group.dataset.explode:'');});});});
}
const tip=document.querySelector('#tooltip');document.querySelectorAll('.leaf,.hyp-leaf').forEach(el=>{el.addEventListener('pointerenter',e=>{const family=el.dataset.familyKey,occ=el.dataset.occurrence,scope=el.closest('.case');[...scope.querySelectorAll('.leaf,.hyp-leaf')].filter(x=>x.dataset.familyKey===family&&x.dataset.occurrence===occ).forEach(x=>x.classList.add('is-hover'));tip.textContent=`${el.dataset.label||`部件 ${el.dataset.componentId}`} · 兄弟族 ${family}`;tip.style.display='block';});el.addEventListener('pointermove',e=>{tip.style.left=`${e.clientX+13}px`;tip.style.top=`${e.clientY+13}px`;});el.addEventListener('pointerleave',()=>{el.closest('.case').querySelectorAll('.is-hover').forEach(x=>x.classList.remove('is-hover'));tip.style.display='none';});});
const cases=[...document.querySelectorAll('.case')],eager=new URLSearchParams(location.search).has('eager')||location.hash==='#explode';if(eager||!('IntersectionObserver'in window))cases.forEach(setupCase);else{const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){setupCase(entry.target);observer.unobserve(entry.target);}}),{rootMargin:'500px'});cases.forEach(section=>observer.observe(section));}if(location.hash==='#explode'){cases.forEach(setupCase);document.querySelectorAll('.explode-button').forEach(x=>x.click());}
"""


def build_document(cases: list[str]) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unihan 全矢量拆分验收</title><style>{CSS}</style></head><body>
    <header><h1>Unihan 全矢量递归叶部件验收</h1><p>没有 PNG、WebP 或 Canvas。左：PDF 字体轮廓；其余：仓库中心线与候选驱动的 PDF 着色假设。</p><div class="toolbar"><span style="color:#f87171">● 端点</span><span style="color:#38bdf8">● 相交/接触</span><span style="color:#4ade80">× 急转折</span><span>悬停部件可联动高亮并查看 ID</span></div></header>
    <main>{"".join(cases)}</main><div id="tooltip"></div><script>{JS}</script></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--only-errors", action="store_true")
    args = parser.parse_args()

    candidate_rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    rows_by_unicode = {row["unicode"]: row for row in candidate_rows}
    results = json.loads(args.results.read_text("utf-8"))["results"]
    selected = []
    for result in results:
        if args.only_errors and result.get("correct"):
            continue
        if result["unicode"] in rows_by_unicode:
            selected.append(result)
        if len(selected) >= args.limit:
            break
    selected_keys = {(item["unicode"], item["source"]): item for item in selected}
    records, _page_sizes = PDF.parse_pdf_cells(args.bbox_cache)
    records_by_key = {(item["unicode"], item["source"]): item for item in records}
    pages = defaultdict(list)
    for key, result in selected_keys.items():
        record = records_by_key.get(key)
        if record is None:
            raise ValueError(f"no PDF chart cell for U+{key[0]:04X} {key[1]}")
        pages[record["page"]].append((record, result))

    cases = []
    for page_number, items in sorted(pages.items()):
        page_svg = MATCHER.load_pdf_page_svg(args.pdf, page_number)
        for record, result in items:
            prefix = f"u{record['unicode']:04x}-{record['source'].lower()}"
            glyph = extract_pdf_glyph(page_svg, record["bbox"], prefix)
            cases.append(
                render_case(
                    record,
                    rows_by_unicode[record["unicode"]],
                    result,
                    glyph,
                    len(cases),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_document(cases), "utf-8")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
