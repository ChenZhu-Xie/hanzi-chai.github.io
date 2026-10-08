"""Interactive, exportable editor for human review of skeleton cuts and ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .candidate_graph import (
    CandidateComponent,
    compile_candidate_graph,
    compile_catalog_candidate_graph,
)
from .grammar import GlyphRepository
from .multiscale import certify_multiscale
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .route_cover import assign_topology_edges, find_partition_cuts
from .skeleton import generate_skeleton, load_annotation_metadata
from .topology import compress_skeleton
from .topology_certificate import certify_skeleton

_COLOURS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#dc2626",
    "#0d9488",
    "#7c3aed",
)


def _leaf_key(path: tuple[int, ...], glyph_id: int) -> str:
    return f"{glyph_id}@{'.'.join(map(str, path)) or 'root'}"


def _tree_payload(node: CandidateComponent) -> dict:
    return {
        "glyphId": node.glyph_id,
        "path": list(node.path),
        "kind": node.kind,
        "operator": node.operator,
        "strokeIndices": list(node.stroke_indices),
        "children": [_tree_payload(child) for child in node.children],
    }


def _skeleton_path(skeleton: np.ndarray) -> str:
    scale = 100 / skeleton.shape[0]
    return "".join(
        f"M{x * scale:.3f},{y * scale:.3f}h{scale:.3f}v{scale:.3f}h-{scale:.3f}z"
        for y, x in np.argwhere(skeleton)
    )


def _case_fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_skeleton_editor_cases(
    pdf_path: Path,
    bbox_path: Path,
    glyph_data_path: Path,
    candidate_catalog_path: Path,
    annotation_paths: list[Path],
    *,
    size: int = 256,
) -> list[dict]:
    """Build editor inputs using annotation metadata but never annotation paths."""
    cells = parse_cells(bbox_path)
    repository = GlyphRepository.load(glyph_data_path)
    catalog = json.loads(candidate_catalog_path.read_text(encoding="utf-8"))
    pages: dict[int, str] = {}
    cases = []
    for annotation_path in annotation_paths:
        metadata = load_annotation_metadata(annotation_path)
        if not all(metadata.get(key) is not None for key in ("unicode", "source", "candidateGlyphId")):
            continue
        codepoint = int(str(metadata["unicode"]).removeprefix("U+"), 16)
        source = str(metadata["source"])
        candidate_id = int(metadata["candidateGlyphId"])
        cell = find_cell(cells, codepoint, source)
        if cell.page not in pages:
            pages[cell.page] = export_page_svg(pdf_path, cell.page)
        geometry = extract_cell_geometry(pages[cell.page], cell)
        multiscale = certify_multiscale(geometry)
        ink, skeleton = generate_skeleton(geometry, size)
        certificate = certify_skeleton(ink, skeleton)
        topology = compress_skeleton(skeleton)
        try:
            candidate = compile_candidate_graph(repository, candidate_id)
        except (KeyError, ValueError):
            candidate = compile_catalog_candidate_graph(catalog, codepoint, candidate_id)
        assignments = assign_topology_edges(topology, candidate, size)
        assignment_by_edge = {item.edge_id: item for item in assignments}
        leaf_keys = sorted(
            {
                _leaf_key(stroke.component_path, stroke.leaf_id)
                for stroke in candidate.strokes
            }
        )
        colour_by_leaf = {
            key: _COLOURS[index % len(_COLOURS)]
            for index, key in enumerate(leaf_keys)
        }
        stroke_count_by_leaf = {key: 0 for key in leaf_keys}
        for stroke in candidate.strokes:
            stroke_count_by_leaf[
                _leaf_key(stroke.component_path, stroke.leaf_id)
            ] += 1
        leaves = [
            {
                "key": key,
                "glyphId": int(key.split("@", 1)[0]),
                "path": [
                    int(value)
                    for value in key.split("@", 1)[1].split(".")
                    if value != "root"
                ],
                "colour": colour_by_leaf[key],
                "strokeCount": stroke_count_by_leaf[key],
            }
            for key in leaf_keys
        ]
        edges = []
        for edge in topology.edges:
            assignment = assignment_by_edge[edge.id]
            owner = _leaf_key(assignment.leaf_key[0], assignment.leaf_key[1])
            edges.append(
                {
                    "id": edge.id,
                    "startNode": edge.start_node,
                    "endNode": edge.end_node,
                    "length": edge.length * 100 / size,
                    "points": [
                        [(x + 0.5) * 100 / size, (y + 0.5) * 100 / size]
                        for x, y in edge.path
                    ],
                    "owner": owner,
                    "cost": assignment.cost,
                    "margin": assignment.margin,
                }
            )
        nodes = [
            {
                "id": node.id,
                "x": (node.centre[0] + 0.5) * 100 / size,
                "y": (node.centre[1] + 0.5) * 100 / size,
                "kind": node.kind,
                "degree": node.degree,
                "edgeIds": list(node.incident_edges),
            }
            for node in topology.nodes
        ]
        proposed = find_partition_cuts(topology, assignments, size)
        case = {
            "key": f"U+{codepoint:04X}-{source}-{candidate_id}",
            "unicode": f"U+{codepoint:04X}",
            "character": chr(codepoint),
            "source": source,
            "candidateGlyphId": candidate_id,
            "annotationBasename": annotation_path.name,
            "truthUsage": "metadata-only; directed paths were not read",
            "skeletonPath": _skeleton_path(skeleton),
            "skeletonPixelCount": int(skeleton.sum()),
            "leaves": leaves,
            "tree": _tree_payload(candidate.root),
            "edges": edges,
            "nodes": nodes,
            "proposedCutNodes": [item.node_id for item in proposed],
            "certificates": {
                "topology": {
                    "certified": certificate.certified,
                    "inkComponents": certificate.ink_components,
                    "skeletonComponents": certificate.skeleton_components,
                    "inkHoles": certificate.ink_holes,
                    "skeletonHoles": certificate.skeleton_holes,
                    "inkEuler": certificate.ink_euler,
                    "skeletonEuler": certificate.skeleton_euler,
                },
                "multiscale": {
                    "certified": multiscale.certified,
                    "sameTopologySignature": multiscale.same_topology_signature,
                    "maximumP95Distance": multiscale.maximum_p95_distance,
                    "minimumBidirectionalCoverage": (
                        multiscale.minimum_bidirectional_coverage
                    ),
                },
            },
        }
        case["baseFingerprint"] = _case_fingerprint(
            {key: value for key, value in case.items() if key != "baseFingerprint"}
        )
        cases.append(case)
    return cases


_EDITOR_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>PDF 骨架人工切割与 IDS 叶归属编辑器</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;font:14px system-ui;color:#172033;background:#eef2f7}header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #cbd5e1;padding:10px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0}.case-nav{display:flex;gap:6px;align-items:center}.case-nav select{min-width:250px;padding:6px}.workspace{display:grid;grid-template-columns:minmax(600px,1.25fr) minmax(390px,.75fr);gap:12px;padding:12px;height:calc(100vh - 62px)}.canvas-panel,.side{background:#fff;border:1px solid #cbd5e1;border-radius:10px;overflow:auto}.canvas-panel{display:flex;flex-direction:column}.toolbar{display:flex;gap:7px;flex-wrap:wrap;padding:9px;border-bottom:1px solid #e2e8f0}.toolbar button,.toolbar select,.actions button{padding:6px 9px;border:1px solid #94a3b8;border-radius:6px;background:#fff;cursor:pointer}.toolbar button.active{background:#1d4ed8;color:#fff;border-color:#1d4ed8}kbd{font:11px ui-monospace;background:#e2e8f0;color:#172033;border-radius:3px;padding:1px 4px}.toolbar button.active kbd{background:#dbeafe}.board-wrap{flex:1;min-height:0;display:grid;place-items:center;padding:8px}.board{width:min(72vh,100%);max-height:100%;aspect-ratio:1;background:#fff;border:1px solid #dbe2ea;touch-action:none}.ink{fill:#e8edf3}.edge{fill:none;stroke:var(--owner);stroke-width:.82;stroke-linecap:round;stroke-linejoin:round}.edge.deleted{stroke:#ef4444;stroke-dasharray:1.5 1;opacity:.25}.edge.selected{stroke-width:1.7;filter:drop-shadow(0 0 1px #111827)}.hit{fill:none;stroke:transparent;stroke-width:4;cursor:pointer}.node{fill:#fff;stroke:#64748b;stroke-width:.38;cursor:pointer}.node.junction{stroke:#ef4444}.node.proposed{stroke:#f59e0b;stroke-width:.75}.node.cut{fill:#ef4444;stroke:#fff}.node.selected{stroke:#111827;stroke-width:1}.split{fill:#fff;stroke:#dc2626;stroke-width:.65;cursor:pointer}.trim{fill:#fff;stroke:#7c3aed;stroke-width:.6}.side{padding:12px}.card{border:1px solid #dbe2ea;border-radius:8px;padding:10px;margin-bottom:10px}.card h2{font-size:15px;margin:0 0 8px}.leaf-list{display:grid;gap:5px}.leaf{display:flex;align-items:center;gap:7px;padding:5px;border:1px solid #e2e8f0;border-radius:5px;cursor:pointer}.leaf.active{outline:2px solid #111827}.swatch{width:14px;height:14px;border-radius:3px;background:var(--colour)}.status.good{background:#dcfce7;color:#166534}.status.warn{background:#fef3c7;color:#92400e}.status.bad{background:#fee2e2;color:#991b1b}.status{padding:8px;border-radius:6px;white-space:pre-wrap}.selected-info{font:12px ui-monospace;white-space:pre-wrap;background:#f8fafc;padding:7px;border-radius:5px}textarea{width:100%;min-height:76px;resize:vertical;margin-top:7px}.actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}.tree ul{margin:3px 0;padding-left:18px}.tree code{font-size:11px}.help{font-size:12px;line-height:1.55}.saved{color:#166534}.unsaved{color:#b45309}@media(max-width:1050px){.workspace{grid-template-columns:1fr;height:auto}.board{width:min(92vw,720px)}}
</style>
<header><h1>PDF 骨架人工切割与 IDS 叶归属编辑器</h1><div class="case-nav"><button id="previous">← 上一个</button><select id="case-select"></select><button id="next">下一个 →</button></div><span id="save-state"></span></header>
<div class="workspace"><section class="canvas-panel"><div class="toolbar">
<button data-mode="inspect"><kbd>Q</kbd> 查看/注释</button><button data-mode="trim"><kbd>W</kbd> 裁剪枝丫</button><button data-mode="delete"><kbd>E</kbd> 删除整链</button><button data-mode="node-cut"><kbd>A</kbd> 节点切口</button><button data-mode="edge-cut"><kbd>S</kbd> 边内切口</button><button data-mode="assign"><kbd>D</kbd> 改归属到叶</button><button id="undo">撤销 Ctrl+Z</button><button id="redo">重做 Ctrl+Shift+Z</button><button id="reset">清空本字操作</button></div><div class="board-wrap"><svg class="board" viewBox="0 0 100 100" aria-label="可编辑 PDF 骨架"></svg></div></section>
<aside class="side"><section class="card"><h2 id="case-title"></h2><div id="validation" class="status"></div></section><section class="card"><h2>当前改归属目标叶</h2><div id="leaves" class="leaf-list"></div></section><section class="card"><h2>所选对象及 comment</h2><div id="selected-info" class="selected-info">尚未选择</div><textarea id="comment" placeholder="说明为什么删、为什么切、为什么改归属；也可记录犹豫点与判据。"></textarea></section><section class="card"><h2>整字总结 comment</h2><textarea id="case-comment" placeholder="总结这个字的正确拆法、关键粘连、IDS/部件映射是否合理。"></textarea></section><section class="card tree"><h2>递归 IDS</h2><div id="tree"></div></section><section class="card"><h2>导入 / 导出</h2><div class="actions"><button id="export">导出全部字 JSON</button><button id="import">导入 JSON</button><input id="file" type="file" accept="application/json" hidden></div><p class="help">自动保存在浏览器 localStorage。导出包含原始 fingerprint、所有点击坐标/链内比例、删枝、截短、节点/链内切口、归属改写和逐对象 comment。</p></section><section class="card help"><b>操作原则</b><br>截短：点击链内一点，自动裁掉离该点更近的一端；可分别修两端。<br>删除：整条最大无分叉链删除/恢复。<br>节点切口：已有红/橙节点处断开语义连通。<br>链内切口：在链内部新增切点；再次点附近可删除。<br>改归属：先选叶，再点链。<br>快捷键仅在非文本输入焦点时生效。</section></aside></div>
<script>
const CASES=__DATA__; const NS='http://www.w3.org/2000/svg', STORAGE='hanzi-chai-skeleton-editor-v1';
const caseSelect=document.querySelector('#case-select'), board=document.querySelector('.board'), comment=document.querySelector('#comment'), caseComment=document.querySelector('#case-comment');
let caseIndex=0,mode='inspect',selected=null,selectedLeaf=null,states={},undoStacks={},redoStacks={};
const emptyState=()=>({deleted:{},trims:{},nodeCuts:{},edgeCuts:[],overrides:{},notes:{case:''}});
try{states=JSON.parse(localStorage.getItem(STORAGE)||'{}')}catch{states={}}
for(const item of CASES){states[item.key]??=emptyState();undoStacks[item.key]=[];redoStacks[item.key]=[];const o=document.createElement('option');o.value=item.key;o.textContent=`${item.unicode} ${item.character} · ${item.source} · ${item.candidateGlyphId}`;caseSelect.append(o)}
function currentCase(){return CASES[caseIndex]} function state(){return states[currentCase().key]}
function clone(value){return JSON.parse(JSON.stringify(value))} function persist(){localStorage.setItem(STORAGE,JSON.stringify(states));const el=document.querySelector('#save-state');el.textContent='已自动保存';el.className='saved'}
function checkpoint(){undoStacks[currentCase().key].push(clone(state()));redoStacks[currentCase().key]=[]}
function mutate(fn){checkpoint();fn(state());persist();render()}
function setMode(next){mode=next;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode))}
function owner(edge){return state().overrides[edge.id]?.leafKey||edge.owner} function noteKey(){return selected?`${selected.type}:${selected.id}`:'case'}
function pointString(points){return points.map(p=>p.map(v=>v.toFixed(3)).join(',')).join(' ')}
function lengths(points){let total=0,out=[0];for(let i=1;i<points.length;i++){total+=Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]);out.push(total)}return {total,out}}
function pointAt(points,fraction){const m=lengths(points),target=m.total*fraction;for(let i=1;i<points.length;i++)if(m.out[i]>=target){const span=m.out[i]-m.out[i-1]||1,t=(target-m.out[i-1])/span;return [points[i-1][0]+(points[i][0]-points[i-1][0])*t,points[i-1][1]+(points[i][1]-points[i-1][1])*t]}return points.at(-1)}
function slicePoints(points,start,end){const m=lengths(points),a=m.total*start,b=m.total*end,out=[pointAt(points,start)];for(let i=1;i<points.length-1;i++)if(m.out[i]>a&&m.out[i]<b)out.push(points[i]);out.push(pointAt(points,end));return out}
function svgPoint(event){const p=board.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform(board.getScreenCTM().inverse());return [q.x,q.y]}
function nearestFraction(points,target){const m=lengths(points);let best={distance:Infinity,fraction:0,point:points[0]};for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy||1,t=Math.max(0,Math.min(1,((target[0]-a[0])*dx+(target[1]-a[1])*dy)/den)),p=[a[0]+dx*t,a[1]+dy*t],d=Math.hypot(target[0]-p[0],target[1]-p[1]);if(d<best.distance)best={distance:d,fraction:(m.out[i-1]+Math.hypot(dx,dy)*t)/m.total,point:p}}return best}
function element(name,attrs={}){const el=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,v);return el}
function selectObject(type,id){selected={type,id};updateSide()}
function edgeClick(event,edge){event.stopPropagation();const hit=nearestFraction(edge.points,svgPoint(event));if(mode==='inspect'){selectObject('edge',edge.id);return}if(mode==='delete'){mutate(s=>{if(s.deleted[edge.id])delete s.deleted[edge.id];else s.deleted[edge.id]={x:hit.point[0],y:hit.point[1],fraction:hit.fraction}});selectObject('delete',edge.id);return}if(mode==='trim'){mutate(s=>{const trim=s.trims[edge.id]||{start:0,end:1};if(hit.fraction<=.5)trim.start=hit.fraction;else trim.end=hit.fraction;s.trims[edge.id]=trim});selectObject('trim',edge.id);return}if(mode==='edge-cut'){const near=state().edgeCuts.findIndex(c=>c.edgeId===edge.id&&Math.abs(c.fraction-hit.fraction)<.025);mutate(s=>{if(near>=0)s.edgeCuts.splice(near,1);else s.edgeCuts.push({id:crypto.randomUUID(),edgeId:edge.id,fraction:hit.fraction,x:hit.point[0],y:hit.point[1]})});const created=state().edgeCuts.find(c=>c.edgeId===edge.id&&Math.abs(c.fraction-hit.fraction)<.025);selectObject(created?'split':'edge',created?.id||edge.id);return}if(mode==='assign'){if(!selectedLeaf)return;mutate(s=>{if(selectedLeaf===edge.owner)delete s.overrides[edge.id];else s.overrides[edge.id]={leafKey:selectedLeaf}});selectObject('assign',edge.id)}}
function nodeClick(event,node){event.stopPropagation();if(mode==='node-cut'){mutate(s=>{if(s.nodeCuts[node.id])delete s.nodeCuts[node.id];else s.nodeCuts[node.id]={x:node.x,y:node.y}});selectObject('node-cut',node.id);return}selectObject('node',node.id)}
function edgeIsSelected(edge){return ['edge','delete','assign'].includes(selected?.type)&&selected.id===edge.id}
function renderBoard(){const c=currentCase(),s=state();board.replaceChildren();const ink=element('path',{d:c.skeletonPath,class:'ink'});board.append(ink);for(const edge of c.edges){const trim=s.trims[edge.id]||{start:0,end:1},points=slicePoints(edge.points,trim.start,trim.end),colour=c.leaves.find(l=>l.key===owner(edge))?.colour||'#111827';const visible=element('polyline',{points:pointString(points),class:`edge ${s.deleted[edge.id]?'deleted':''} ${edgeIsSelected(edge)?'selected':''}`,style:`--owner:${colour}`});board.append(visible);const hit=element('polyline',{points:pointString(edge.points),class:'hit'});hit.addEventListener('click',e=>edgeClick(e,edge));hit.addEventListener('mouseenter',()=>visible.classList.add('selected'));hit.addEventListener('mouseleave',()=>{if(!edgeIsSelected(edge))visible.classList.remove('selected')});board.append(hit);if(trim.start>0){const p=pointAt(edge.points,trim.start);board.append(element('circle',{cx:p[0],cy:p[1],r:.85,class:'trim'}))}if(trim.end<1){const p=pointAt(edge.points,trim.end);board.append(element('circle',{cx:p[0],cy:p[1],r:.85,class:'trim'}))}}
for(const node of c.nodes){const classes=['node',node.kind];if(c.proposedCutNodes.includes(node.id))classes.push('proposed');if(s.nodeCuts[node.id])classes.push('cut');if(['node','node-cut'].includes(selected?.type)&&selected.id===node.id)classes.push('selected');const circle=element('circle',{cx:node.x,cy:node.y,r:node.degree>=3?1.05:.72,class:classes.join(' ')});circle.addEventListener('click',e=>nodeClick(e,node));board.append(circle)}
for(const cut of s.edgeCuts){const circle=element('circle',{cx:cut.x,cy:cut.y,r:1,class:'split'});circle.addEventListener('click',e=>{e.stopPropagation();selectObject('split',cut.id)});board.append(circle)}}
function groups(){const c=currentCase(),s=state(),active=c.edges.filter(e=>!s.deleted[e.id]);const parent=new Map(active.map(e=>[e.id,e.id]));const find=x=>{while(parent.get(x)!==x){parent.set(x,parent.get(parent.get(x)));x=parent.get(x)}return x};const union=(a,b)=>{a=find(a);b=find(b);if(a!==b)parent.set(a,b)};for(const node of c.nodes){if(s.nodeCuts[node.id])continue;const edges=node.edgeIds.map(id=>active.find(e=>e.id===id)).filter(Boolean);for(let i=1;i<edges.length;i++)if(owner(edges[0])===owner(edges[i]))union(edges[0].id,edges[i].id)}const by={};for(const edge of active){const leaf=owner(edge);by[leaf]??=new Set();by[leaf].add(find(edge.id))}for(const cut of s.edgeCuts){const edge=active.find(e=>e.id===cut.edgeId);if(edge)by[owner(edge)].add(`split:${cut.id}`)}return {active,by}}
function validation(){const c=currentCase(),s=state(),g=groups(),expected=new Set(c.leaves.map(l=>l.key)),present=new Set(g.active.map(owner)),missing=[...expected].filter(x=>!present.has(x)),unknown=[...present].filter(x=>!expected.has(x)),manualCuts=Object.keys(s.nodeCuts).length+s.edgeCuts.length,commentsNeeded=Object.keys(s.deleted).length+Object.keys(s.trims).length+Object.keys(s.nodeCuts).length+s.edgeCuts.length+Object.keys(s.overrides).length,commented=new Set(Object.entries(s.notes).filter(([,v])=>String(v).trim()).map(([k])=>k));let uncommented=0;for(const id of Object.keys(s.deleted))if(!commented.has(`delete:${id}`))uncommented++;for(const id of Object.keys(s.trims))if(!commented.has(`trim:${id}`))uncommented++;for(const id of Object.keys(s.nodeCuts))if(!commented.has(`node-cut:${id}`))uncommented++;for(const cut of s.edgeCuts)if(!commented.has(`split:${cut.id}`))uncommented++;for(const id of Object.keys(s.overrides))if(!commented.has(`assign:${id}`))uncommented++;const groupText=c.leaves.map(l=>`${l.key}: ${g.by[l.key]?.size||0} 组`).join('\n'),certificateOk=c.certificates.topology.certified&&c.certificates.multiscale.certified,certificateText=certificateOk?'通过':`未通过（单尺度 ${c.certificates.topology.certified?'通过':'失败'}；多尺度 ${c.certificates.multiscale.certified?'通过':'失败'}）`;return {missing,unknown,manualCuts,commentsNeeded,uncommented,certificateOk,text:`骨架证书 ${certificateText}\n有效链 ${g.active.length}/${c.edges.length}\n叶部件 ${present.size}/${expected.size}\n手工切口 ${manualCuts}\n${groupText}\n缺失叶：${missing.join(', ')||'无'}\n未知叶：${unknown.join(', ')||'无'}\n尚无 comment 的操作：${uncommented}`}}
function treeHtml(node){const label=`${node.operator||node.kind} · ${node.glyphId} · path ${node.path.length?node.path.join('.'):'root'}`;return `<li><code>${label}</code>${node.children.length?`<ul>${node.children.map(treeHtml).join('')}</ul>`:''}</li>`}
function selectedDescription(){if(!selected)return '尚未选择';const c=currentCase(),s=state();if(['edge','delete','assign'].includes(selected.type)){const e=c.edges.find(x=>x.id===selected.id),label={edge:'查看链',delete:'删除整链',assign:'改叶归属'}[selected.type];return `${label} ${e.id}\n节点 ${e.startNode} → ${e.endNode}\n长度 ${e.length.toFixed(2)}\n原归属 ${e.owner}\n当前归属 ${owner(e)}\n候选代价 ${e.cost.toFixed(2)}\n次优差距 ${e.margin.toFixed(2)}\n删除 ${!!s.deleted[e.id]}`}if(selected.type==='trim'){const e=c.edges.find(x=>x.id===selected.id),t=s.trims[e.id];return `链 ${e.id} 的端部裁剪\n保留 fraction ${t.start.toFixed(4)} → ${t.end.toFixed(4)}\n归属 ${owner(e)}`}if(['node','node-cut'].includes(selected.type)){const n=c.nodes.find(x=>x.id===selected.id);return `${selected.type==='node-cut'?'节点切口':'查看节点'} ${n.id}\n${n.kind} / degree ${n.degree}\n位置 ${n.x.toFixed(3)}, ${n.y.toFixed(3)}\n候选切口 ${c.proposedCutNodes.includes(n.id)}\n手工切口 ${!!s.nodeCuts[n.id]}`}if(selected.type==='split'){const x=s.edgeCuts.find(x=>x.id===selected.id);return `边内切口 ${x.id}\n链 ${x.edgeId}\nfraction ${x.fraction.toFixed(5)}\n位置 ${x.x.toFixed(3)}, ${x.y.toFixed(3)}`}return JSON.stringify(selected)}
function updateSide(){const c=currentCase(),s=state(),v=validation();document.querySelector('#case-title').textContent=`${c.unicode} ${c.character} · ${c.source} · candidate ${c.candidateGlyphId}`;const box=document.querySelector('#validation');box.textContent=v.text;box.className=`status ${!v.certificateOk||v.missing.length||v.unknown.length?'bad':v.uncommented?'warn':'good'}`;document.querySelector('#selected-info').textContent=selectedDescription();comment.disabled=!selected;comment.value=selected?(s.notes[noteKey()]||''):'';caseComment.value=s.notes.case||'';document.querySelector('#tree').innerHTML=`<ul>${treeHtml(c.tree)}</ul>`;const leaves=document.querySelector('#leaves');leaves.replaceChildren();for(const leaf of c.leaves){const item=document.createElement('div');item.className=`leaf ${selectedLeaf===leaf.key?'active':''}`;item.style.setProperty('--colour',leaf.colour);item.innerHTML=`<span class="swatch"></span><span>${leaf.key} · ${leaf.strokeCount} 笔</span>`;item.addEventListener('click',()=>{selectedLeaf=leaf.key;updateSide()});leaves.append(item)}}
function render(){caseSelect.selectedIndex=caseIndex;renderBoard();updateSide()}
function changeCase(index){caseIndex=(index+CASES.length)%CASES.length;selected=null;selectedLeaf=currentCase().leaves[0]?.key||null;render()}
caseSelect.addEventListener('change',()=>changeCase(caseSelect.selectedIndex));document.querySelector('#previous').onclick=()=>changeCase(caseIndex-1);document.querySelector('#next').onclick=()=>changeCase(caseIndex+1);document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
comment.addEventListener('input',()=>{if(selected){state().notes[noteKey()]=comment.value;persist()}});caseComment.addEventListener('input',()=>{state().notes.case=caseComment.value;persist()});
function undo(){const u=undoStacks[currentCase().key];if(!u.length)return;redoStacks[currentCase().key].push(clone(state()));states[currentCase().key]=u.pop();persist();render()}function redo(){const r=redoStacks[currentCase().key];if(!r.length)return;undoStacks[currentCase().key].push(clone(state()));states[currentCase().key]=r.pop();persist();render()}
document.querySelector('#undo').onclick=undo;document.querySelector('#redo').onclick=redo;document.querySelector('#reset').onclick=()=>{if(confirm('清空本字的所有骨架操作和 comment？'))mutate(s=>{Object.assign(s,emptyState())})};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.ctrlKey&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?redo():undo();return}const map={q:'inspect',w:'trim',e:'delete',a:'node-cut',s:'edge-cut',d:'assign'};if(map[e.key.toLowerCase()]){e.preventDefault();setMode(map[e.key.toLowerCase()])}})
function exportPayload(){return {schemaVersion:1,kind:'hanzi-chai-skeleton-edits',exportedAt:new Date().toISOString(),cases:CASES.map(c=>({key:c.key,unicode:c.unicode,character:c.character,source:c.source,candidateGlyphId:c.candidateGlyphId,baseFingerprint:c.baseFingerprint,expectedLeaves:c.leaves.map(l=>l.key),edits:clone(states[c.key]),validation:(()=>{const old=caseIndex;caseIndex=CASES.indexOf(c);const v=validation();caseIndex=old;return v})()}))}}
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`hanzi-skeleton-edits-${new Date().toISOString().slice(0,10)}.json`;a.click();URL.revokeObjectURL(a.href)};document.querySelector('#import').onclick=()=>document.querySelector('#file').click();document.querySelector('#file').onchange=async e=>{const data=JSON.parse(await e.target.files[0].text());for(const item of data.cases||[]){const base=CASES.find(c=>c.key===item.key);if(base&&base.baseFingerprint===item.baseFingerprint)states[item.key]=item.edits}persist();render()};
board.addEventListener('click',()=>{selected=null;updateSide()});setMode('inspect');changeCase(0);
</script>"""


def render_skeleton_editor(cases: list[dict]) -> str:
    if not cases:
        raise ValueError("no directed annotation metadata cases were found")
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    return _EDITOR_TEMPLATE.replace("__DATA__", payload)
