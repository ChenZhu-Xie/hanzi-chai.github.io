import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import shapely.geometry
from glyph_decomposer.primitive_audit import (
    parse_cells, find_cell, export_page_svg,
    extract_cell_geometry, generate_skeleton
)

cells = parse_cells(Path(".local/unicode/U4E00-bbox.html"))
cell = find_cell(cells, 0x7740, "J")
page_svg = export_page_svg(Path(".local/unicode/U4E00.pdf"), cell.page)
geom = extract_cell_geometry(page_svg, cell)
ink, skel = generate_skeleton(geom, 256)

# Extract SVG outline in 0..100 space
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

# Extract all skeleton points (x, y in 0..100)
skel_pixels = []
for y in range(256):
    for x in range(256):
        if skel[y, x]:
            skel_pixels.append((round(x * 100.0 / 256.0, 2), round(y * 100.0 / 256.0, 2), x, y))

print(f"Total skel points: {len(skel_pixels)}")

# Now define the visual ground truth annotations for all 12 strokes:
# Each stroke has:
# - name, feature, component
# - primitive_type: 'bezier' | 'line' | 'polyline'
# - fitted_svg_d: SVG path string
# - control_points: list of points [[x, y], ...]
# - vision_logic: detailed markdown annotation thought
# - algorithm_critique: deep analysis of why the algorithm failed here
# - claimed_filter: function or condition to select its skeleton points

stroke_defs = [
    {
        "index": 0,
        "name": "首笔·左点",
        "feature": "点",
        "component": "羊 (0, 0)",
        "primitive_type": "bezier",
        "path_d": "M 27.5,10.2 Q 33.0,17.5 40.5,23.5",
        "controls": [[27.5, 10.2], [33.0, 17.5], [40.5, 23.5]],
        "vision_logic": "羊字头左上侧的斜点，自左上向右下微微顿挫取势。起笔位于 (27.5, 10.2)，顺势向东南斜下延伸，在 (40.5, 23.5) 汇入首横交汇处。视觉上是一段饱满微曲的二次贝塞尔曲线，与右侧斜撇形成聚散开阖之势。",
        "algorithm_critique": "算法陷阱：容易将左点与下方的首横左端点误连，或者因缺少首尾切向角惩罚而把点画过度缩短为孤立的 2 个像素噪点。视觉模型则明确将其识别为自西北向东南的独立笔意。",
        "box": [24.0, 9.0, 42.0, 24.5]
    },
    {
        "index": 1,
        "name": "次笔·右撇",
        "feature": "撇",
        "component": "羊 (0, 0)",
        "primitive_type": "bezier",
        "path_d": "M 73.5,10.2 Q 67.0,17.0 58.5,23.5",
        "controls": [[73.5, 10.2], [67.0, 17.0], [58.5, 23.5]],
        "vision_logic": "羊字头右上侧的斜撇，起笔在右上高位 (73.5, 10.2)，向西南方向轻快撇出，收于 (58.5, 23.5)。与左点呈八字对峙之态，两端对称布白。",
        "algorithm_critique": "算法陷阱：在骨架距离场中，左右两点的终点与首横极度贴近，组合图搜索若无方向先验，常将左点终点与右撇终点直接横向相连伪装成一段短横。视觉模型必须严格区分撇向矢量与横向矢量。",
        "box": [56.0, 9.0, 75.0, 24.5]
    },
    {
        "index": 2,
        "name": "第3笔·首短横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "primitive_type": "line",
        "path_d": "M 36.5,23.4 L 63.5,23.4",
        "controls": [[36.5, 23.4], [63.5, 23.4]],
        "vision_logic": "羊部的上短横，居于左右点撇下方中央，左接点尾，右接撇收，横平竖直。起笔 (36.5, 23.4)，收笔 (63.5, 23.4)，长度适中，不向外泛出。",
        "algorithm_critique": "算法陷阱：此横两端紧邻点与撇，若端点搜索半径（endpoint count）开得过大，优化器容易‘借道’拐进点撇的竖向支路，导致横画端点错位。视觉标注必须锚定严格水平轴向。",
        "box": [35.0, 21.0, 65.0, 26.0]
    },
    {
        "index": 3,
        "name": "第4笔·次中横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "primitive_type": "line",
        "path_d": "M 20.7,34.0 L 79.7,34.0",
        "controls": [[20.7, 34.0], [79.7, 34.0]],
        "vision_logic": "羊部的中间横，跨度较首横明显放大，承上启下。自左侧 (20.7, 34.0) 水平向右劲行，中穿第5笔竖画，收笔于 (79.7, 34.0)。",
        "algorithm_critique": "算法陷阱：此横在中部与羊中竖形成正交十字节点。算法若在交叉点计算 Chamfer 或无接触互斥，往往会将中横在中心折向中竖，破坏直线连续性。视觉认知坚决保持单段横画的直线拓扑贯通。",
        "box": [19.0, 31.5, 81.0, 36.5]
    },
    {
        "index": 4,
        "name": "第5笔·羊中竖",
        "feature": "竖",
        "component": "羊 (0, 0)",
        "primitive_type": "line",
        "path_d": "M 49.6,24.6 L 48.4,44.9",
        "controls": [[49.6, 24.6], [48.4, 44.9]],
        "vision_logic": "羊字中轴短竖，上贯首横、中穿次横、下抵长横。垂直下行自 (49.6, 24.6) 至 (48.4, 44.9)，下端与长横形成 T 字交点，明确在此处截断收笔。",
        "algorithm_critique": "算法陷阱（极严重）：纯骨架图在 (48.4, 44.9) 是多路交汇点，优化器曾把此竖继续向下贪心寻路，跨过长横一路扎进下方的目部，造成跨偏旁连体畸形！视觉模型明确指出：羊竖在长横止步，严禁越界！",
        "box": [46.5, 24.0, 52.0, 46.5]
    },
    {
        "index": 5,
        "name": "第6笔·托底长横",
        "feature": "横",
        "component": "羊 (0, 0)",
        "primitive_type": "line",
        "path_d": "M 10.9,44.9 L 89.5,44.9",
        "controls": [[10.9, 44.9], [89.5, 44.9]],
        "vision_logic": "羊部托底长横，全字最舒展的一笔水平主梁。横贯全字东西，从最左侧 (10.9, 44.9) 一路平稳向右至最右侧 (89.5, 44.9)，下托目框，下悬长撇。",
        "algorithm_critique": "算法陷阱：长横下方挂接了长撇与目框两个大部件。由于交点极多，无层级约束的算法经常在此横上拆成 3 段短碎段，造成 coverage 分割破碎。视觉必须作为一条完整的整体主梁予以覆盖。",
        "box": [9.0, 43.0, 91.0, 47.0]
    },
    {
        "index": 6,
        "name": "第7笔·核心长撇",
        "feature": "撇",
        "component": "羊长撇 (0, 1)",
        "primitive_type": "bezier",
        "path_d": "M 35.9,46.9 Q 28.0,60.0 10.2,77.7",
        "controls": [[35.9, 46.9], [28.0, 60.0], [10.2, 77.7]],
        "vision_logic": "【全字核心病灶笔画】羊部左下大舒展长撇！起笔自长横下方 (35.9, 46.9)，沿左下方大弧度舒展滑翔，掠过目字框左侧，飘逸收笔于 (10.2, 77.7)。这是一条优雅的西南向二次贝塞尔曲线，占据了 x:10~36, y:47~78 的整片开阔骨架！",
        "algorithm_critique": "【原算法致命 Bug 根源剖析】：\n1. 算法此前为何舍弃这段骨架去偷占目部竖画？因为目部 5 笔密集，骨架密度大，距离场引力极强；而撇起点就在 (36, 47)，紧邻目部左竖 (35, 50)，直接向下走距离误差几乎为 0！\n2. 算法切线盲区：原算法仅检查前 5 像素切向，向下微偏在局部被判为合法撇；完全忽略了宏观总弦向必须为强西南向 (dx=-25, dy=+30)！\n3. 跨偏旁免责漏洞：原互斥逻辑仅查同 leaf，对偷占目部竖画零惩罚！\n视觉模型明确：必须以全局弦向与骨架排他性作为硬约束，确保此长撇完全占据左下斜扫轨道！",
        "box": [9.0, 46.0, 39.0, 80.0]
    },
    {
        "index": 7,
        "name": "第8笔·目部左竖",
        "feature": "竖",
        "component": "目 (1,)",
        "primitive_type": "line",
        "path_d": "M 35.2,52.2 L 35.5,84.4",
        "controls": [[35.2, 52.2], [35.5, 84.4]],
        "vision_logic": "目字框的左立柱竖画。居于长横之下、长撇右侧，垂直坚挺，从目框左上 (35.2, 52.2) 笔挺下行至目框左下 (35.5, 84.4)。",
        "algorithm_critique": "算法陷阱：当第7笔长撇被错误引诱向下时，这根左竖被两条笔画重叠压占；若粗暴引入互斥，又会导致第8笔无路可走变成 0/N 失败。只有当长撇回到左下轨道，左竖才能清清爽爽地独占属于目部的垂直骨架。",
        "box": [34.0, 50.0, 38.0, 86.0]
    },
    {
        "index": 8,
        "name": "第9笔·目横折",
        "feature": "横折",
        "component": "目 (1,)",
        "primitive_type": "polyline",
        "path_d": "M 35.2,54.7 L 73.4,54.7 L 73.4,84.0",
        "controls": [[35.2, 54.7], [73.4, 54.7], [73.4, 84.0]],
        "vision_logic": "目字框的上框与右立柱。左起 (35.2, 54.7)，向右横走至 (73.4, 54.7)，随即转折直下至 (73.4, 84.0)。折角锐利清晰，构成目框外廓。",
        "algorithm_critique": "算法陷阱：折角处若用单条平滑曲线拟合，会严重内切剥蚀直角外轮廓，使得折角残余大量红点。必须使用折线（Polyline）或硬直拐点原语。",
        "box": [34.0, 52.5, 75.0, 86.0]
    },
    {
        "index": 9,
        "name": "第10笔·目首内横",
        "feature": "横",
        "component": "目 (1,)",
        "primitive_type": "line",
        "path_d": "M 36.0,64.5 L 72.8,64.5",
        "controls": [[36.0, 64.5], [72.8, 64.5]],
        "vision_logic": "目字框内部的第一根短横，水平横贯目框，将目框上半部分开。起笔 (36.0, 64.5)，平整收于右竖 (72.8, 64.5)。",
        "algorithm_critique": "算法陷阱：目内短横间距密集（约 10px）。若偏旁尺度拟合失衡，算法容易将内部横画上下错位，或者同向挤压。视觉布白要求均分目框空间。",
        "box": [35.0, 62.5, 74.0, 66.5]
    },
    {
        "index": 10,
        "name": "第11笔·目次内横",
        "feature": "横",
        "component": "目 (1,)",
        "primitive_type": "line",
        "path_d": "M 36.5,74.2 L 72.8,74.2",
        "controls": [[36.5, 74.2], [72.8, 74.2]],
        "vision_logic": "目字框内部的第二根短横，与上方首内横、下方封底横严格等距布白。起笔 (36.5, 74.2)，水平行至右竖 (72.8, 74.2)。",
        "algorithm_critique": "算法陷阱：容易与第12笔封底横发生距离混淆，需要自上而下严格的时空先后关系和位置先验约束。",
        "box": [35.0, 72.0, 74.0, 76.5]
    },
    {
        "index": 11,
        "name": "第12笔·目封底横",
        "feature": "横",
        "component": "目 (1,)",
        "primitive_type": "line",
        "path_d": "M 35.5,84.4 L 73.8,84.4",
        "controls": [[35.5, 84.4], [73.8, 84.4]],
        "vision_logic": "目部件最终收官的底封口横。自左立柱底端 (35.5, 84.4) 水平锁至右折立柱底端 (73.8, 84.4)，全字严密闭合。",
        "algorithm_critique": "算法陷阱：若底横与左右立柱收笔处稍有微隙，末端几个骨架点就会成为永久孤立残余。视觉标注确保三点交汇闭合整字下底。",
        "box": [34.5, 82.5, 75.0, 86.5]
    }
]

print(f"Verified {len(stroke_defs)} strokes.")
