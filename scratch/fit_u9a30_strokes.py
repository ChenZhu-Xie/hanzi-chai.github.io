import sys
import json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point

data = json.load(open("scratch/u9a30_raw_data.json", encoding="utf-8"))
skel_pixels = data["skel_pixels"]
print(f"Total skel points: {len(skel_pixels)}")

# Define all 20 strokes with precise calligraphic and visual primitives
stroke_defs = [
    # =========================================================================
    # Component 1: 月 (0,) - Left radical (Strokes 0..3)
    # =========================================================================
    {
        "index": 0,
        "name": "首笔·月竖撇",
        "feature": "撇",
        "component": "月 (0,)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 17.6,15.5 Q 17.6,72.0 9.0,87.5",
        "controls": [[17.6, 15.5], [17.6, 72.0], [9.0, 87.5]],
        "color": "#38bdf8",
        "vision_logic": "月字旁的左立柱竖撇。采用控制点下沉策略 P1=(17.6, 72.0)：让前半程 (y=15.5~55) 紧紧依附垂直纵轴 (x=17.6) 直线下行，后半程 (y>55) 顺畅平滑掠出西南撇脚至 (9.0, 87.5)，完美契合‘先竖后撇’的书法动力学。",
        "algorithm_critique": "【普适算法启发】对于‘竖撇’类复合笔画，无需切断为两条折线段，通过将单段二次贝塞尔的控制点 P1 沿竖直切线大幅向后半程下沉（y1 选取在 75% 深度），即可利用 t^2 的迟滞特性实现‘前 60% 严格保真垂直，后 40% 平滑弧出出锋’的优雅效果！"
    },
    {
        "index": 1,
        "name": "次笔·月横折钩",
        "feature": "横折钩",
        "component": "月 (0,)",
        "tool": "折线段 (Polyline L+L+L)",
        "primitive_type": "polyline",
        "path_d": "M 17.6,15.5 L 31.2,15.5 L 31.2,84.5 L 26.2,80.5",
        "controls": [[17.6, 15.5], [31.2, 15.5], [31.2, 84.5], [26.2, 80.5]],
        "color": "#818cf8",
        "vision_logic": "月字旁的上框、右立柱与底钩。起笔自左竖上端 (17.6, 15.5) 横向至 (31.2, 15.5)，随即折直向下至 (31.2, 84.5)，向西北内扣提钩出锋至 (26.2, 80.5)。",
        "algorithm_critique": "【原算法盲区】折角若用单条光滑贝塞尔曲线拟合会内切削角；底钩在 Chamfer 场中常因噪点干扰把提钩反向延伸为拖尾。"
    },
    {
        "index": 2,
        "name": "第3笔·月上内横",
        "feature": "横",
        "component": "月 (0,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 17.6,35.2 L 31.2,35.2",
        "controls": [[17.6, 35.2], [31.2, 35.2]],
        "color": "#34d399",
        "vision_logic": "月字腹腔内部的第一根短横，水平横跨左右两柱，两端紧搭，位置在全月高的 1/3 黄金分割处。",
        "algorithm_critique": "【原算法盲区】跨距短（仅 13.6px），若端点吸附半径过大，容易误吸左右竖柱上的像素造成横画倾斜。"
    },
    {
        "index": 3,
        "name": "第4笔·月下内横",
        "feature": "横",
        "component": "月 (0,)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 17.6,54.7 L 31.2,54.7",
        "controls": [[17.6, 54.7], [31.2, 54.7]],
        "color": "#a3e635",
        "vision_logic": "月字腹腔内部的第二根短横，与上内横严格平行等距，起笔 (17.6, 54.7)，收笔 (31.2, 54.7)。",
        "algorithm_critique": "【原算法盲区】与右侧『馬』字的首横 (y=54.7) 恰好处于同一水平基线！无跨部件边界隔离时，算法常发生‘横跨月部与马部’的恶性越界偷连！"
    },

    # =========================================================================
    # Component 2: 朕顶部八字头 (1, 0, 0) (Strokes 4..5)
    # =========================================================================
    {
        "index": 4,
        "name": "第5笔·八·左撇",
        "feature": "撇",
        "component": "八 (1, 0, 0)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 52.0,14.8 Q 47.0,18.0 42.2,22.7",
        "controls": [[52.0, 14.8], [47.0, 18.0], [42.2, 22.7]],
        "color": "#facc15",
        "vision_logic": "朕字右上『龹』顶部的左侧斜点/撇。自 (52.0, 14.8) 向西南斜顿滑下至 (42.2, 22.7)。",
        "algorithm_critique": "【原算法盲区】与下方的天部横画左端点极近，贪心搜索常将其误并入横画左端，误判为横画的倒折起笔。"
    },
    {
        "index": 5,
        "name": "第6笔·八·右捺",
        "feature": "捺",
        "component": "八 (1, 0, 0)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 71.9,14.4 Q 78.5,19.0 85.9,23.8",
        "controls": [[71.9, 14.4], [78.5, 19.0], [85.9, 23.8]],
        "color": "#fb923c",
        "vision_logic": "朕字右上『龹』顶部的右侧斜捺。自 (71.9, 14.4) 向东南舒展下行，与左撇呈八字对峙，收于 (85.9, 23.8)。",
        "algorithm_critique": "【原算法盲区】右侧紧贴天部首横右端，Chamfer 场在两笔交会处形成低谷，易与首横右端发生粘连。"
    },

    # =========================================================================
    # Component 3: 朕中部天/大 (1, 0, 1) (Strokes 6..9)
    # =========================================================================
    {
        "index": 6,
        "name": "第7笔·天·首横",
        "feature": "横",
        "component": "天 (1, 0, 1)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 44.9,24.6 L 86.7,24.6",
        "controls": [[44.9, 24.6], [86.7, 24.6]],
        "color": "#f43f5e",
        "vision_logic": "天字部的上横，横贯右半部分东西 (44.9 至 86.7)，承托上方八字头，中穿中心立柱竖。",
        "algorithm_critique": "【原算法盲区】中穿十字节点时容易发生 90 度折射，误向上下竖向拐弯。"
    },
    {
        "index": 7,
        "name": "第8笔·天·次横",
        "feature": "横",
        "component": "天 (1, 0, 1)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 42.2,35.2 L 88.7,35.2",
        "controls": [[42.2, 35.2], [88.7, 35.2]],
        "color": "#e11d48",
        "vision_logic": "天字部的下长横，全字右半部分最宽宏的水平主梁！自 (42.2, 35.2) 一路笔直向右至 (88.7, 35.2)，承上启下。",
        "algorithm_critique": "【原算法盲区】交点极多（大撇、大捺、中心竖皆交汇于此），局部图搜索经常将长横切解为多截碎块。"
    },
    {
        "index": 8,
        "name": "第9笔·天·大撇",
        "feature": "撇",
        "component": "天 (1, 0, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 63.0,11.7 Q 58.0,28.0 45.7,48.0",
        "controls": [[63.0, 11.7], [58.0, 28.0], [45.7, 48.0]],
        "color": "#a855f7",
        "vision_logic": "【核心骨干主笔】从八字头顶部高位 (63.0, 11.7) 贯通而下，如长虹贯日般连穿两横，向西南大幅度舒展滑翔，收于 (45.7, 48.0)。",
        "algorithm_critique": "【原算法致命盲区】原算法以为此撇仅从次横处起笔（截短至 y=26..48），导致上方 11.7~24.6 的一整段纵向主脊柱骨架被当做‘不可解释残差’彻底遗弃！"
    },
    {
        "index": 9,
        "name": "第10笔·天·大捺",
        "feature": "捺",
        "component": "天 (1, 0, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 64.5,35.2 Q 77.0,41.0 89.5,48.5",
        "controls": [[64.5, 35.2], [77.0, 41.0], [89.5, 48.5]],
        "color": "#d946ef",
        "vision_logic": "天字部的向右斜伸大捺笔。自十字交叉中心 (64.5, 35.2) 向东南舒展下行，捺脚饱满展开收于 (89.5, 48.5)。",
        "algorithm_critique": "【原算法盲区】因与下方『馬』字首横端点接近，算法常误将捺脚下弯扎进马框。"
    },

    # =========================================================================
    # Component 4: 馬 (1, 1, 0) - Right-Bottom Radical (Strokes 10..15)
    # =========================================================================
    {
        "index": 10,
        "name": "第11笔·馬·首横",
        "feature": "横",
        "component": "馬 (1, 1, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 47.3,54.7 L 79.3,54.7",
        "controls": [[47.3, 54.7], [79.3, 54.7]],
        "color": "#06b6d4",
        "vision_logic": "馬字部上框首横，从左竖顶端 (47.3, 54.7) 平整向右延伸至 (79.3, 54.7)。",
        "algorithm_critique": "【原算法盲区】与左侧月部下横在同一 y 轴上，无 Leaf 隔离时极易发生跨偏旁互通。"
    },
    {
        "index": 11,
        "name": "第12笔·馬·左竖",
        "feature": "竖",
        "component": "馬 (1, 1, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 47.3,50.0 L 47.3,70.3",
        "controls": [[47.3, 50.0], [47.3, 70.3]],
        "color": "#0891b2",
        "vision_logic": "馬字部的左立柱竖画，顶冲至 (47.3, 50.0)，垂直下行贯穿三横，止于下横交点 (47.3, 70.3)。",
        "algorithm_critique": "【原算法盲区】旧算法只认出 19 像素碎片，评分暴涨至 10.29 异常高损。"
    },
    {
        "index": 12,
        "name": "第13笔·馬·中横",
        "feature": "横",
        "component": "馬 (1, 1, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 47.3,62.1 L 79.7,62.1",
        "controls": [[47.3, 62.1], [79.7, 62.1]],
        "color": "#14b8a6",
        "vision_logic": "馬字身躯中间横，平稳平行居于首横与下横之中，自 (47.3, 62.1) 至 (79.7, 62.1)。",
        "algorithm_critique": "【原算法盲区】旧算法仅取 16 像素，严重缩水导致残差大爆发。"
    },
    {
        "index": 13,
        "name": "第14笔·馬·下横",
        "feature": "横",
        "component": "馬 (1, 1, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 47.3,69.5 L 84.8,69.5",
        "controls": [[47.3, 69.5], [84.8, 69.5]],
        "color": "#059669",
        "vision_logic": "馬字身躯托底长横，连接左竖下端，向右水平贯穿中竖，并向右延伸连接折钩立柱 (84.8, 69.5)。",
        "algorithm_critique": "【原算法盲区】中竖交叉与右立柱接点众多，导致线段无法闭环。"
    },
    {
        "index": 14,
        "name": "第15笔·馬·中竖",
        "feature": "竖",
        "component": "馬 (1, 1, 0)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 64.5,50.0 L 64.5,69.5",
        "controls": [[64.5, 50.0], [64.5, 69.5]],
        "color": "#10b981",
        "vision_logic": "馬字中脊梁骨，自顶部 (64.5, 50.0) 贯穿三横下抵下横 (64.5, 69.5)，居于马身正中。",
        "algorithm_critique": "【原算法盲区】与天部大撇下段相交，旧算法误将此竖延伸借道天部大撇。"
    },
    {
        "index": 15,
        "name": "第16笔·馬·横折钩",
        "feature": "横折钩",
        "component": "馬 (1, 1, 0)",
        "tool": "折线段 (Polyline L+L)",
        "primitive_type": "polyline",
        "path_d": "M 84.8,69.5 L 84.8,77.7 L 73.0,77.7",
        "controls": [[84.8, 69.5], [84.8, 77.7], [73.0, 77.7]],
        "color": "#eab308",
        "vision_logic": "【全字最大崩塌笔画】馬字右下包围折立柱与向左底提钩！自 (84.8, 69.5) 笔挺下行至 (84.8, 77.7)，随后向左平直勾出至 (73.0, 77.7)。",
        "algorithm_critique": "【原算法灾难现场】原算法在此笔直接发生溃缩！pts 仅选了 7 像素，score 爆出 15.01 恶性高分，导致整条右立柱与底钩骨架完全裸露未认领，产生 600+ 巨额残余！"
    },

    # =========================================================================
    # Component 5: 四点底 (1, 1, 1) (Strokes 16..19)
    # =========================================================================
    {
        "index": 16,
        "name": "第17笔·四点·第1点",
        "feature": "点",
        "component": "四点底 (1, 1, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 43.0,78.5 Q 48.0,81.0 53.0,83.5",
        "controls": [[43.0, 78.5], [48.0, 81.0], [53.0, 83.5]],
        "color": "#f97316",
        "vision_logic": "四点底最左侧之斜撇点，左向取势，自 (43.0, 78.5) 斜下顿挫至 (53.0, 83.5)。",
        "algorithm_critique": "【原算法盲区】常因与月字旁底撇距离接近而被误判吸附入月部。"
    },
    {
        "index": 17,
        "name": "第18笔·四点·第2点",
        "feature": "点",
        "component": "四点底 (1, 1, 1)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 65.5,78.5 L 68.5,85.0",
        "controls": [[65.5, 78.5], [68.5, 85.0]],
        "color": "#ef4444",
        "vision_logic": "四点底第二点，小直立点，自 (65.5, 78.5) 直立下顿至 (68.5, 85.0)。",
        "algorithm_critique": "【原算法盲区】四点极密（间距 6~8px），算法易在相邻两点间串扰偷吸。"
    },
    {
        "index": 18,
        "name": "第19笔·四点·第3点",
        "feature": "点",
        "component": "四点底 (1, 1, 1)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 74.5,78.5 L 77.5,86.0",
        "controls": [[74.5, 78.5], [77.5, 86.0]],
        "color": "#ec4899",
        "vision_logic": "四点底第三点，小斜点，自 (74.5, 78.5) 紧凑下顿至 (77.5, 86.0)。",
        "algorithm_critique": "【原算法盲区】上贴馬部底钩，无精准间隙先验时易与底钩横线粘合。"
    },
    {
        "index": 19,
        "name": "第20笔·四点·第4点",
        "feature": "点",
        "component": "四点底 (1, 1, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 81.0,78.5 Q 82.5,82.0 84.0,86.3",
        "controls": [[81.0, 78.5], [82.5, 82.0], [84.0, 86.3]],
        "color": "#d946ef",
        "vision_logic": "四点底最右收官大捺点，顺势向东南饱满顿笔出锋，自 (81.0, 78.5) 顿至 (84.0, 86.3)，托定全字重心。",
        "algorithm_critique": "【原算法盲区】与馬部折钩下角同处右侧边界，容易发生端点争抢。"
    }
]

# Calculate coverage
def sample_bezier_pts(p0, p1, p2, n=100):
    t = np.linspace(0, 1, n)
    bx = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    by = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return list(zip(bx, by))

stroke_lines = []
for s in stroke_defs:
    ctrls = s["controls"]
    if s["primitive_type"] == "bezier":
        pts = sample_bezier_pts(ctrls[0], ctrls[1], ctrls[2], 100)
        stroke_lines.append(LineString(pts))
    elif s["primitive_type"] == "polyline":
        stroke_lines.append(LineString(ctrls))
    else:
        stroke_lines.append(LineString([ctrls[0], ctrls[1]]))

skel_pts = [Point(p[0], p[1]) for p in skel_pixels]

for s in stroke_defs:
    s["claimed_indices"] = []

unclaimed_indices = []
for p_idx, pt in enumerate(skel_pts):
    dists = [pt.distance(sl) for sl in stroke_lines]
    min_d = min(dists)
    best_s = int(np.argmin(dists))
    if min_d <= 2.5:
        stroke_defs[best_s]["claimed_indices"].append(p_idx)
    else:
        unclaimed_indices.append(p_idx)

total_pts = len(skel_pts)
claimed_total = total_pts - len(unclaimed_indices)
pct = claimed_total / total_pts * 100
print(f"Total points: {total_pts}, Claimed: {claimed_total} ({pct:.1f}%), Unclaimed: {len(unclaimed_indices)}")
for i, s in enumerate(stroke_defs):
    print(f"Stroke {i:02d} [{s['name']}]: {len(s['claimed_indices']):3d} pts ({len(s['claimed_indices'])/total_pts*100:.1f}%)")

# Save complete payload for HTML generator
payload = {
    "unicode": "U+9A30",
    "character": "騰",
    "source": "K",
    "glyph_id": 133032,
    "svg_outline": data["svg_outline"],
    "skel_pixels": skel_pixels,
    "strokes": stroke_defs,
    "unclaimed_indices": unclaimed_indices,
    "total_pts": total_pts,
    "claimed_total": claimed_total,
    "old_algo_failures": [
        {
            "strokeIndex": 15,
            "name": "馬·横折钩 (旧算法崩溃缩水为 7 像素)",
            "path_d": "M 84.8,70.0 L 84.8,71.5",
            "score": 15.01,
            "failure_summary": "原算法在此笔发生灾难性溃缩！仅匹配了 7 像素噪点，损失高达 15.01，直接导致馬部右侧整根立柱与底钩骨架（共 130+ 像素）全盘裸露未被认领，造成整字遗留 633 巨额残余！"
        },
        {
            "strokeIndex": 8,
            "name": "天·大撇 (旧算法上方高位断头)",
            "path_d": "M 55.0,26.5 L 45.7,48.0",
            "score": 1.60,
            "failure_summary": "原算法死板依赖 nominal_box，认为天部撇画只在 y=26..48，把从八字头顶部 (y=11.7) 贯通而下的一整段纵向主脊柱骨架当做不可解释噪点截断抛弃！"
        }
    ]
}

Path("scratch/u9a30_annotation_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved to scratch/u9a30_annotation_data.json")
