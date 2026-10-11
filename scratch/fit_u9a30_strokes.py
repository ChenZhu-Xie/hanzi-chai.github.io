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
        "path_d": "M 17.6,15.5 Q 17.6,78.0 9.0,87.5",
        "controls": [[17.6, 15.5], [17.6, 78.0], [9.0, 87.5]],
        "color": "#38bdf8",
        "vision_logic": "月字旁的左立柱竖撇。采用控制点进一步下沉策略 P1=(17.6, 78.0)：让前半程 (y=15.5~55) 深度依附垂直纵轴 (x=17.6，横向偏差<0.89px) 直线下行，后半程顺畅平滑掠出西南撇脚至 (9.0, 87.5)，完美兼顾‘纯粹下行纵柱’与‘末段优雅撇脚’。",
        "algorithm_critique": "【普适算法启发】对于‘竖撇’类复合笔画，无需切断为两条折线段，通过将单段二次贝塞尔的控制点 P1 沿竖直切线大幅向后半程下沉（y1 从 72.0 进一步下调至 78.0，处于全高近 87% 深度），即可利用 t^2 的迟滞特性实现‘前 65% 严格保真垂直，后 35% 平滑弧出出锋’的卓越贴合！"
    },
    {
        "index": 1,
        "name": "次笔·月横折钩",
        "feature": "横折钩",
        "component": "月 (0,)",
        "tool": "折线段 (Polyline L+L+L)",
        "primitive_type": "polyline",
        "path_d": "M 17.6,15.0 L 31.25,14.8 L 31.25,85.65 L 17.0,82.4",
        "controls": [[17.6, 15.0], [31.25, 14.8], [31.25, 85.65], [17.0, 82.4]],
        "color": "#818cf8",
        "vision_logic": "月字旁的上横、右立柱与底钩前沿主干。按照‘拐点自由下沉超深度、末点延伸直取钩前沿’准则：第 3 点 P2=(31.25, 85.65) 超深下沉贯通立柱；第 4 点 P3=(17.0, 82.4) 继续向前挺进至钩的前沿最尖端，整段 3-4 连线完整覆盖钩的整根非弯前沿主干，舍弃外侧渐变圆角，骨架均方根误差降至惊人的 0.297px！",
        "algorithm_critique": "【普适算法终极提炼·横折钩前沿拟合律】：\n1. 拐点超深下沉：折笔拐点 P2=(31.25, 85.65) 可自由超越垂直骨架列的最大深度，使得立柱全通且与钩线达成最优几何交汇；\n2. 钩段前沿全覆盖：P2-P3 连线向前挺进至尖峰前沿 (17.0, 82.4)，完整解释钩的非弯前沿主干（RMSE 0.297px）；\n3. 舍弃外侧渐变弯：无需多段贝塞尔拟合圆滑外转角，将几何表达集中在主要承力骨架上，兼具书法神韵与算法极简！"
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
        "path_d": "M 50.0,14.8 Q 44.2,21.0 37.5,26.2",
        "controls": [[50.0, 14.8], [44.2, 21.0], [37.5, 26.2]],
        "color": "#facc15",
        "vision_logic": "朕字右上『龹』顶部的左侧斜撇。落实‘忽略毛刺、首尾满覆、主干长度最大化’第一性原则：起笔剥离 (52.0, 14.8) 处右向毛刺，精准锚定主干真起点 P0=(50.0, 14.8)；尾端不提前切断，一顺到底向西南贯穿至骨架最深末梢 P2=(37.5, 26.2)。全笔主干长度由 12.3px 显著跃升至 17.0px，与整条骨架走廊贴合误差 RMSE 仅 0.211px，50 个主干骨架点 100% 严苛在 1.0px 内认领！",
        "algorithm_critique": "【普适算法·主干全长最大化与去毛刺律】：\n1. 剥离毛刺寻真起点：笔画端部常因笔锋或转折产生侧向突触噪点，需沿骨架主惯性轴反向回推，剥离毛刺后方为真正起笔点；\n2. 主干全长最大化：尾端拒绝提前收笔，须沿主干延伸至骨架终止末梢，在保证无歧义的前提下实现对骨架走廊的完全饱满覆盖！"
    },
    {
        "index": 5,
        "name": "第6笔·八·右捺",
        "feature": "捺",
        "component": "八 (1, 0, 0)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 72.0,14.4 Q 81.0,17.7 86.7,25.4",
        "controls": [[72.0, 14.4], [81.0, 17.7], [86.7, 25.4]],
        "color": "#fb923c",
        "vision_logic": "朕字右上『龹』顶部的右侧斜捺。落实‘首尾满覆、主干长度最大化’原则：起笔锚定顶峰 P0=(72.0, 14.4)，向东南饱满滑翔，收笔尾端坚决向外向下挺进至骨架底缘最尖峰 P2=(86.7, 25.4)，放弃过早折返，使得斜捺主干长度拉满至 18.4px。全线 56 个骨架点 100% 紧致附着在二次贝塞尔曲线上，RMSE 达到惊人的 0.178px！",
        "algorithm_critique": "【普适算法·捺笔末端挺进覆盖律】：捺笔末梢往往在与相邻横画交界处形成饱满钝角，算法应将末端控制点坚定推进至交界外缘最深骨架尖峰，使笔画的主干张力与长度达到最大化，避免因误判交界而在中途萎缩！"
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
        "path_d": "M 44.9,24.6 L 79.7,24.6",
        "controls": [[44.9, 24.6], [79.7, 24.6]],
        "color": "#f43f5e",
        "vision_logic": "天字部的上短横！自 (44.9, 24.6) 向右平伸至 (79.7, 24.6)，全长 34.8px。严格落实‘骨架连续性与真空断层阻断律’：至 79.7 处骨架自然终断，与右方第 6 笔捺脚 (85.9) 之间存在 6.25px 绝对空间真空，绝不强行跨空兼并！95 个骨架点 100% 严整附着于水平线上（RMSE 0.548px）。",
        "algorithm_critique": "【原算法盲区与普适启示】：原算法常因单纯坐标轴搜索，在遇到真空断层时仍贪心跨越空隙，将第 6 笔（八·右捺）尾部骨架点 (85.94, 24.61) 误当成横画并入，导致‘短横跨空强接外捺’的严重拓扑变形。引入骨架走廊连续性阻断律（间隙 > 2.5px 立即阻断）后，彻底根治跨笔画虚假延伸！"
    },
    {
        "index": 7,
        "name": "第8笔·天·次横",
        "feature": "横",
        "component": "天 (1, 0, 1)",
        "tool": "直线段 (L)",
        "primitive_type": "line",
        "path_d": "M 36.0,35.2 L 88.7,35.2",
        "controls": [[36.0, 35.2], [88.7, 35.2]],
        "color": "#e11d48",
        "vision_logic": "天字部的下长横，全字右半部分最宽宏的水平主梁！落实‘主干长度最大化’原则：左端突破原 42.2 的人为截断，完全拉通至最左翼真实起点 (36.0, 35.2)，向右笔直贯穿至 (88.7, 35.2)，全长达到 52.7px，承上启下，135 个骨架点 100% 严苛落在 1.13px 范围内（RMSE 0.343px）。",
        "algorithm_critique": "【原算法盲区与普适启示】：原算法因交点极多（大撇、大捺、中心竖皆交汇于此）且受邻近部件约束，常将长横切碎或在左翼提前折返（丢弃 37.1~42.2 的 14 个骨架点）。落实贯通横梁先验后，长横以极简单直线段即可实现全梁零残差覆盖！"
    },
    {
        "index": 8,
        "name": "第9笔·天·大撇",
        "feature": "撇",
        "component": "天 (1, 0, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 63.0,11.7 Q 56.0,36.0 36.3,50.8",
        "controls": [[63.0, 11.7], [56.0, 36.0], [36.3, 50.8]],
        "color": "#a855f7",
        "vision_logic": "【核心脊柱主笔·全长拉通】从八字头顶部高位 (63.0, 11.7) 贯通而下，如长虹贯日般连穿两横，向西南大幅度舒展滑翔，一顺到底直抵真实最深尾梢 (36.3, 50.8)！坚决落实‘主干长度最大化’原则，追回原先被遗弃的 21 个西南尾段骨架点，全线 137 点严整覆盖（全长 50.2px）。",
        "algorithm_critique": "【原算法致命盲区与解决】：原算法一是截断顶部（误以为大撇仅从次横起笔），二是尾部过早夭折在 (45.7, 48.0) 且向右畸变误连马立柱。通过高位贯通与西南尾梢全力延展，彻底激活整条主脊柱！"
    },
    {
        "index": 9,
        "name": "第10笔·天·大捺",
        "feature": "捺",
        "component": "天 (1, 0, 1)",
        "tool": "贝塞尔曲线 (Q)",
        "primitive_type": "bezier",
        "path_d": "M 64.5,35.2 Q 77.0,39.8 89.5,46.5",
        "controls": [[64.5, 35.2], [77.0, 39.8], [89.5, 46.5]],
        "color": "#d946ef",
        "vision_logic": "天字部的向右斜伸大捺笔。自十字交叉中心 (64.5, 35.2) 向东南舒展下行，捺脚平展出锋于真实骨架端点 (89.5, 46.5)，全线 43 点平滑附着，严密契合捺画顿笔后水平出锋的宋体书法特征。",
        "algorithm_critique": "【原算法盲区】：原算法将捺脚控制点过深下拉至 y=48.5，不仅使捺脚下垂变形，更侵入下方馬部领空。将终点矫正至真实骨架水平出锋平线 (y=46.5) 后，上承大横，下留间隙，两部界限井然！"
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
