import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import shapely.geometry
from shapely.geometry import LineString, Point
from glyph_decomposer.primitive_audit import (
    parse_cells, find_cell, export_page_svg,
    extract_cell_geometry, generate_skeleton
)

print("1. Extracting cell geometry and skeleton...")
cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x7740, "J")
page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

bx0, by0, bx1, by1 = cell.bbox
bw = bx1 - bx0
bh = by1 - by0

def ring_to_d(ring):
    coords = list(ring.coords)
    if not coords: return ""
    d = f"M {(coords[0][0]-bx0)/bw*100:.2f},{(coords[0][1]-by0)/bh*100:.2f}"
    for c in coords[1:]:
        d += f" L {(c[0]-bx0)/bw*100:.2f},{(c[1]-by0)/bh*100:.2f}"
    d += " Z"
    return d

parts = [ring_to_d(geom.exterior)]
for inner in geom.interiors:
    parts.append(ring_to_d(inner))
svg_outline = " ".join(parts)

skel_pixels = []
for y in range(256):
    for x in range(256):
        if skel[y, x]:
            skel_pixels.append([round(x * 100.0 / 256.0, 2), round(y * 100.0 / 256.0, 2), int(x), int(y)])

print(f"Extracted {len(skel_pixels)} skeleton points.")

# Define 12 strokes with exact coordinates
strokes_meta = [
    {
        "index": 0,
        "name": "首笔·左点",
        "feature": "点",
        "component": "羊 (0, 0)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 29.7,9.4 Q 35.0,15.5 40.6,23.0",
        "controls": [[29.7, 9.4], [35.0, 15.5], [40.6, 23.0]],
        "color": "#38bdf8", # sky-400
        "vision_logic": "羊字头左上侧之露锋落笔点，自西北 (29.7, 9.4) 向东南 (40.6, 23.0) 倾斜取势，下端饱满汇入首横交汇处。视觉模型感知为一段微弯且顿挫有力的二次贝塞尔曲线，与右撇形成聚散开阖之态。",
        "algorithm_critique": "【原算法盲区】因首横骨架极贴近点下端，距离场搜索往往将点截短或误借道首横；若加偏旁惩罚又易退化为孤立噪点。视觉模型凭借笔势贯通先验，清晰将其锚定在西北至东南的自然落笔轴线上。"
    },
    {
        "index": 1,
        "name": "次笔·右撇",
        "feature": "撇",
        "component": "羊 (0, 0)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 69.9,12.9 Q 65.0,17.5 59.0,23.0",
        "controls": [[69.9, 12.9], [65.0, 17.5], [59.0, 23.0]],
        "color": "#818cf8", # indigo-400
        "vision_logic": "羊字头右上侧之斜撇，起笔在右上高位 (69.9, 12.9)，向西南方向轻快掠出，收笔于 (59.0, 23.0)。与左点八字相映，两侧布白匀称。",
        "algorithm_critique": "【原算法盲区】在 Chamfer 场中，左右两点与横画交界处形成鞍点。算法经常试图在两点间横向偷连以降低局部距离差，破坏了撇画自右上向左下的动力学矢量。"
    },
    {
        "index": 2,
        "name": "第3笔·首横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 14.1,23.4 L 86.7,23.4",
        "controls": [[14.1, 23.4], [86.7, 23.4]],
        "color": "#34d399", # emerald-400
        "vision_logic": "羊部上横梁，在日文字形（JIS/J-source）中展现出舒展平整的横向大跨度，从最左 (14.1, 23.4) 贯穿至最右 (86.7, 23.4)，左托左点，右承右撇，横平竖直。",
        "algorithm_critique": "【原算法严重失误】原算法依据标准楷书先验误以为此笔为短横（仅 36~64 范围），导致两侧 14~36 与 64~87 的大批骨架被丢弃为残余孤立点！视觉模型直接看到真实字形在 y=23.4 的整条水平线段。"
    },
    {
        "index": 3,
        "name": "第4笔·次横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 20.7,34.0 L 79.7,34.0",
        "controls": [[20.7, 34.0], [79.7, 34.0]],
        "color": "#a3e635", # lime-400
        "vision_logic": "羊部中间横，跨度较首横稍收 (20.7 至 79.7)，平稳刚健，中部与中竖形成正交十字交叉，承上启下。",
        "algorithm_critique": "【原算法盲区】中穿十字节点时，无方向惯性惩罚的寻路器极易在中点向上下竖向拐弯，把一条连续横画折断。视觉模型严格维持单一水平线段拓扑。"
    },
    {
        "index": 4,
        "name": "第5笔·羊中竖",
        "feature": "竖",
        "component": "羊 (0, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 49.6,24.6 L 48.4,44.9",
        "controls": [[49.6, 24.6], [48.4, 44.9]],
        "color": "#facc15", # amber-400
        "vision_logic": "羊部中轴竖笔，上贯首横、中穿次横、下抵托底长横 (48.4, 44.9)，在长横处戛然而止，构成标准 T 形停靠点。",
        "algorithm_critique": "【原算法重大越界病灶】算法此前在此处发生‘穿刺’：因目部下方存在纵向竖画，贪心算法越过长横一路扎进下方的目部，造成跨部件连体畸形！视觉模型确认：羊竖必须止于长横，严禁跨部件延伸！"
    },
    {
        "index": 5,
        "name": "第6笔·托底长横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 10.9,44.9 L 89.5,44.9",
        "controls": [[10.9, 44.9], [89.5, 44.9]],
        "color": "#fb923c", # orange-400
        "vision_logic": "全字最宽宏舒展的水平主梁！横贯全字自最左 (10.9, 44.9) 至最右 (89.5, 44.9)，上承羊头，下悬长撇与目框，结构基石。",
        "algorithm_critique": "【原算法盲区】下方悬挂目左立柱与长撇起笔，交汇支点极多。旧图搜索在此横上裂解出 3 段碎裂候选。视觉认知直接确立整根完整主梁。"
    },
    {
        "index": 6,
        "name": "第7笔·核心长撇",
        "feature": "撇",
        "component": "羊长撇 (0, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 35.9,46.9 Q 28.0,60.0 10.2,77.7",
        "controls": [[35.9, 46.9], [28.0, 60.0], [10.2, 77.7]],
        "color": "#f43f5e", # rose-500 (HIGHLIGHT)
        "vision_logic": "【全字核心病灶笔画】羊部下延大舒展长撇！从长横下方 (35.9, 46.9) 挥毫，沿西南方向大弧度飘逸滑翔，掠过目框左侧，收笔于 (10.2, 77.7)。这是一段优美的二次贝塞尔曲线，覆盖了 x:10~36, y:47~78 的整片广阔骨架！",
        "algorithm_critique": "【原算法致命失败全景剖析】：\n1. 密集体引力场陷阱：目部 5 笔密集，骨架密集度极高；长撇起点就在 (36, 47)，紧挨着目左竖 (35.2, 52.2)，向下走的局部 Chamfer 距离误差极小！\n2. 切向盲区：原算法只验前 5 像素局部切向，而撇画起笔本就微偏下，算法误以为合法；完全缺失了宏观全局弦向（dx=-25, dy=+30，西南 230°）的几何硬约束！\n3. 跨部件免责漏洞：同 Leaf 互斥无法防跨 Leaf 侵占，目属于 Leaf(1,)，盗占目竖未被施加两重/三重重罚！\n视觉模型：以全局空间与骨架独占性，坚决夺回左下斜扫走廊！"
    },
    {
        "index": 7,
        "name": "第8笔·目部左竖",
        "feature": "竖",
        "component": "目 (1,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 35.2,52.2 L 35.5,84.4",
        "controls": [[35.2, 52.2], [35.5, 84.4]],
        "color": "#a855f7", # purple-500
        "vision_logic": "目字框的左立柱竖画。居于长横之下、长撇右侧，垂直坚挺，从 (35.2, 52.2) 笔挺下行至 (35.5, 84.4)，形成目框的左边界基柱。",
        "algorithm_critique": "【原算法连锁反应】当第 7 笔长撇被错误引诱向下时，此左竖被严重重叠挤占；若强加互斥又导致第 8 笔全军覆没！只有长撇回归左下斜道，左竖才能干干净净地独占属于目部的垂直骨架。"
    },
    {
        "index": 8,
        "name": "第9笔·目横折",
        "feature": "横折",
        "component": "目 (1,)",
        "tool": "折线段 (Polyline L+L)",
        "primitive_type": "polyline",
        "path_d": "M 35.2,54.7 L 73.4,54.7 L 73.4,84.0",
        "controls": [[35.2, 54.7], [73.4, 54.7], [73.4, 84.0]],
        "color": "#ec4899", # pink-500
        "vision_logic": "目字框的上框与右立柱。左起 (35.2, 54.7)，水平右行至 (73.4, 54.7)，随即刚直下折至 (73.4, 84.0)。折角锐利方正，定格目框外围几何轮廓。",
        "algorithm_critique": "【原算法盲区】折角若用单条光滑贝塞尔曲线拟合，会在右上角内切‘切角’，导致直角外侧残余大批红色未拟合点。必须采用两段式折线（Polyline）或硬直折点原语。"
    },
    {
        "index": 9,
        "name": "第10笔·目首内横",
        "feature": "横",
        "component": "目 (1,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 36.0,64.5 L 72.8,64.5",
        "controls": [[36.0, 64.5], [72.8, 64.5]],
        "color": "#06b6d4", # cyan-500
        "vision_logic": "目字框内第一根横画，水平横跨目框腹腔，两端精准搭接左竖与右折，将目框上半部分开。",
        "algorithm_critique": "【原算法盲区】目框内横间距极密（约 9.5px）。算法在缺乏等距先验时，容易在两根内横间上下串扰或位置畸变。"
    },
    {
        "index": 10,
        "name": "第11笔·目次内横",
        "feature": "横",
        "component": "目 (1,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 36.5,74.2 L 72.8,74.2",
        "controls": [[36.5, 74.2], [72.8, 74.2]],
        "color": "#14b8a6", # teal-500
        "vision_logic": "目字框内第二根横画，与上方首内横、下方封底横形成严格的三等分布白，匀称端庄。",
        "algorithm_critique": "【原算法盲区】容易与封底横产生距离竞争。视觉模型依循由上至下的时空顺序，稳定锁定中位线。"
    },
    {
        "index": 11,
        "name": "第12笔·目封底横",
        "feature": "横",
        "component": "目 (1,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 35.5,84.4 L 73.8,84.4",
        "controls": [[35.5, 84.4], [73.8, 84.4]],
        "color": "#22c55e", # green-500
        "vision_logic": "目部件最终收官之底横，自左竖底端 (35.5, 84.4) 水平锁至右折底端 (73.8, 84.4)，全字闭合稳固，全神贯注。",
        "algorithm_critique": "【原算法盲区】底横与左右立柱交接处的微小缝隙会导致残余红点。视觉模型确保三点严格交叠闭合。"
    }
]

# Calculate points claimed by each stroke
def sample_bezier_pts(p0, p1, p2, n=100):
    t = np.linspace(0, 1, n)
    bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return list(zip(bx, by))

stroke_lines = []
for s in strokes_meta:
    ctrls = s["controls"]
    if s["primitive_type"] == "bezier":
        pts = sample_bezier_pts(ctrls[0], ctrls[1], ctrls[2], 100)
        stroke_lines.append(LineString(pts))
    elif s["primitive_type"] == "polyline":
        stroke_lines.append(LineString(ctrls))
    else:
        stroke_lines.append(LineString([ctrls[0], ctrls[1]]))

skel_pts = [Point(p[0], p[1]) for p in skel_pixels]

for s in strokes_meta:
    s["claimed_indices"] = []

unclaimed_indices = []
for p_idx, pt in enumerate(skel_pts):
    dists = [pt.distance(sl) for sl in stroke_lines]
    min_d = min(dists)
    best_s = int(np.argmin(dists))
    if min_d <= 2.2:
        strokes_meta[best_s]["claimed_indices"].append(p_idx)
    else:
        unclaimed_indices.append(p_idx)

total_pts = len(skel_pts)
claimed_total = total_pts - len(unclaimed_indices)
print(f"Total points: {total_pts}, Claimed: {claimed_total} ({claimed_total/total_pts*100:.1f}%), Unclaimed: {len(unclaimed_indices)}")

# Old algorithm's buggy route for Stroke 6 for ghost comparison
old_algo_stroke6 = {
    "name": "原算法错误长撇 (盗占目左立柱)",
    "path_d": "M 36.1,47.1 C 35.1,58.1 22.3,66.7 14.3,75.2",
    "route_snippet": [[36.1, 47.1], [36.1, 52.2], [35.5, 60.0], [35.5, 84.4]],
    "failure_summary": "旧算法在此处舍弃了左下方真正的斜向撇骨架走廊（x:10~36, y:47~78），直接竖直扎入目部左竖（x:35.2, y:52~84），不仅导致左下斜骨架大面积未覆盖，还造成与目左竖发生 100% 重叠侵占！"
}

# Serialize payload
payload = {
    "unicode": "U+7740",
    "character": "着",
    "source": "J",
    "glyph_id": 900019,
    "svg_outline": svg_outline,
    "skel_pixels": skel_pixels,
    "strokes": strokes_meta,
    "unclaimed_indices": unclaimed_indices,
    "total_pts": total_pts,
    "claimed_total": claimed_total,
    "old_algo_stroke6": old_algo_stroke6
}

out_json = Path("scratch/u7740_annotation_data.json")
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Data saved to {out_json}")
