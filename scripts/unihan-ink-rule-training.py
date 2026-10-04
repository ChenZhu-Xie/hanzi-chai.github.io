"""Learn explainable start/junction rules from verified directed strokes.

The output is not a black-box classifier.  Every reusable rule is keyed by
exact leaf component ID, stroke ordinal/count, and stroke feature, and retains
the source cases that support it.  The consumer excludes its current target
case before aggregating, which makes leave-one-out evaluation auditable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INK = load_module("unihan-ink-diffusion.py", "unihan_ink_rule_training_diffusion")
TRANSFER = INK.TRANSFER
MATCHER = INK.MATCHER
PDF = INK.PDF
DIRECTED = INK.RESIDUAL_DECODER.DIRECTED


def normalized(point: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> list[float]:
    value = INK.normalized_point(point, bounds)
    return [round(float(value[0]), 4), round(float(value[1]), 4)]


def source_rule_key(example: dict) -> str:
    """Keep writing conventions isolated even when leaf IDs happen to match."""
    return f"{str(example['source']).upper()}|{example['key']}"


def build_rule_model(examples: list[dict]) -> dict:
    """Build a source-aware rule model from constraint-clean observations.

    A hard-constraint failure means that the observation still needs review;
    it is not evidence that a rejected branch is a useful negative example.
    """
    accepted = [item for item in examples if not item.get("constraintViolations")]
    rejected = [item for item in examples if item.get("constraintViolations")]
    by_signature: dict[str, list[dict]] = defaultdict(list)
    for example in accepted:
        by_signature[source_rule_key(example)].append(example)
    return {
        "schemaVersion": 2,
        "model": "source-aware-exact-leaf-directed-half-edge-rules",
        "minimumSupportForScoring": 2,
        "caseCount": len({example["case"] for example in accepted}),
        "examples": accepted,
        "examplesBySignature": dict(by_signature),
        "rejectedConstraintExampleCount": len(rejected),
        "reviewCases": [
            {
                "case": item["case"],
                "reasons": list(item.get("constraintViolations", ())),
            }
            for item in rejected
        ],
    }


def route_half_edge_evidence(directed, route_edge_ids: tuple[int, ...]) -> tuple[list[int], list[int]]:
    chosen = []
    forbidden = []
    for incoming_id, outgoing_id in zip(route_edge_ids, route_edge_ids[1:]):
        incoming = directed.edges[incoming_id]
        outgoing = directed.edges[outgoing_id]
        if incoming.end_gate != outgoing.start_gate or incoming.road_id == outgoing.road_id:
            continue
        chosen.append(outgoing_id)
        gate = directed.gates[incoming.end_gate]
        forbidden.extend(
            edge_id
            for edge_id in gate.outgoing
            if directed.edges[edge_id].road_id != incoming.road_id
            and edge_id != outgoing_id
        )
    return list(dict.fromkeys(chosen)), list(dict.fromkeys(forbidden))


def learn_example(
    annotation_path: Path,
    rows: list[dict],
    records: list[dict],
    page_cache: dict[int, str],
    pdf: Path,
    canvas: int,
) -> list[dict]:
    document = json.loads(annotation_path.read_text("utf-8-sig"))
    metadata = document["metadata"]
    codepoint = int(str(metadata["unicode"]).removeprefix("U+"), 16)
    source = str(metadata["source"])
    glyph_id = int(metadata["candidateGlyphId"])
    case = f"U+{codepoint:04X}-{source}"
    row = next(item for item in rows if int(item["unicode"]) == codepoint)
    candidate = TRANSFER.candidate_strokes(row, glyph_id)
    truth, _normalized, corrections = TRANSFER.load_human_annotations(
        annotation_path,
        codepoint=codepoint,
        source=source,
        glyph_id=glyph_id,
        candidate=candidate,
        canvas=canvas,
    )
    record = next(
        item
        for item in records
        if int(item["unicode"]) == codepoint and str(item["source"]) == source
    )
    page = int(record["page"])
    page_svg = page_cache.setdefault(page, MATCHER.load_pdf_page_svg(pdf, page))
    source_mask = MATCHER.render_pdf_vector_cell(
        page_svg, MATCHER.chart_glyph_bbox(record["bbox"]), size=canvas
    ) < 224
    target = TRANSFER.normalize_target(source_mask, canvas)
    graph = INK.build_skeleton_graph(TRANSFER.skeletonize(target))
    directed = DIRECTED.build_directed_skeleton(graph)
    signatures = INK.stroke_rule_signatures(candidate)

    glyph_points = np.vstack([stroke["points"] for stroke in truth])
    glyph_bounds = (glyph_points.min(axis=0), glyph_points.max(axis=0))
    component_points: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for stroke in truth:
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        component_points[key].append(stroke["points"])
    component_bounds = {
        key: (points.min(axis=0), points.max(axis=0))
        for key, members in component_points.items()
        for points in [np.vstack(members)]
    }

    examples = []
    previous: list[np.ndarray] = []
    for index, (stroke, signature) in enumerate(zip(truth, signatures)):
        points = np.asarray(stroke["points"], dtype=float)
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        earlier_distance = (
            min(INK.polyline_distance(points, other) for other in previous)
            if previous
            else None
        )
        junctions = INK.junction_decision_details(points, graph)
        route_edge_ids = DIRECTED.trace_route_edges(directed, graph, points)
        chosen_half_edges, forbidden_half_edges = route_half_edge_evidence(
            directed, route_edge_ids
        )
        examples.append(
            {
                "case": case,
                "unicode": f"U+{codepoint:04X}",
                "character": chr(codepoint),
                "source": source,
                "glyphId": glyph_id,
                "annotation": str(annotation_path),
                "annotationOrderCorrections": corrections,
                "stroke": index + 1,
                **signature,
                "start": {
                    "topologyRole": INK.point_topology_role(graph, points[0]),
                    "directionSector": INK.direction_sector(points),
                    "componentNormalized": normalized(points[0], component_bounds[key]),
                    "glyphNormalized": normalized(points[0], glyph_bounds),
                    "touchesEarlierStroke": (
                        earlier_distance is not None and earlier_distance <= canvas * 0.018
                    ),
                    "distanceToEarlierPercent": (
                        None
                        if earlier_distance is None
                        else round(float(earlier_distance) / canvas * 100, 4)
                    ),
                },
                "junctionTurnSequence": [item["chosen"] for item in junctions],
                "junctions": junctions,
                "routeHalfEdges": list(route_edge_ids),
                "chosenHalfEdges": chosen_half_edges,
                "forbiddenHalfEdges": forbidden_half_edges,
                "constraintViolations": [],
            }
        )
        previous.append(points)
    return examples


def markdown_report(model: dict) -> str:
    examples = model["examples"]
    by_signature = model["examplesBySignature"]
    role_counts = Counter(example["start"]["topologyRole"] for example in examples)
    junctions = [junction for example in examples for junction in example["junctions"]]
    chosen_counts = Counter(junction["chosen"] for junction in junctions)
    rejected_counts = Counter(
        rejected for junction in junctions for rejected in junction["rejected"]
    )
    lines = [
        "# 青色人工有向笔画：可解释训练规则",
        "",
        "本报告由人工真值自动提炼。每条规则的主键是 `部件 ID | 部件内第几笔 | 部件笔画数 | 笔画类别`。",
        "评测目标必须排除自身 case；支持数少于 2 的规则只解释、不参与评分。",
        "",
        "## 全局事实",
        "",
        f"- 人工字源：{model['caseCount']}；人工有向笔画：{len(examples)}；可复用签名：{len(by_signature)}。",
        "- 起点拓扑角色：" + "、".join(f"{key}={value}" for key, value in role_counts.most_common()) + "。",
        f"- 真正路口决策：{len(junctions)} 个；选择：" + ("、".join(f"{key}={value}" for key, value in chosen_counts.most_common()) or "无") + "。",
        "- 在真实路口未选择的方向：" + ("、".join(f"{key}={value}" for key, value in rejected_counts.most_common()) or "无") + "。",
        "",
        "## 算法使用原则",
        "",
        "1. 先用部件 ID、部件内笔序、总笔数和笔类定位规则；不同兄弟部件不得混训。",
        "2. 起点必须说明它是端点、普通路径还是路口，但该角色会随相邻笔画交叠而变化，只作解释、不评分。",
        "3. 同一签名在多个独立字源中的全字归一化起点方差很小时，才把起点区域作为弱先验。",
        "4. 到路口时比较有向进入方向和后续语法；只惩罚明确作出的反例转向，不惩罚字体间路口检测数量差异。",
        "5. 一个样本只提供解释，至少两个排除目标后的独立 case 才能改变评分。",
        "6. 无同部件证据时明确回退到方向/转折/覆盖率，不把相似部件偷偷当真值。",
        "",
        "## 可复用部件笔画规则",
        "",
        "| 签名 | case 数 | 起点角色 | 部件内起点均值 | 起向 | 路口序列 | 支持 case |",
        "|---|---:|---|---|---|---|---|",
    ]
    for signature, members in sorted(
        by_signature.items(), key=lambda item: (-len({x['case'] for x in item[1]}), item[0])
    ):
        cases = sorted({member["case"] for member in members})
        roles = Counter(member["start"]["topologyRole"] for member in members)
        sectors = Counter(member["start"]["directionSector"] for member in members)
        turns = Counter(tuple(member["junctionTurnSequence"]) for member in members)
        positions = np.asarray([member["start"]["componentNormalized"] for member in members])
        role = roles.most_common(1)[0]
        sector = sectors.most_common(1)[0]
        turn = turns.most_common(1)[0]
        lines.append(
            f"| `{signature}` | {len(cases)} | {role[0]} {role[1]}/{len(members)} | "
            f"({positions[:,0].mean():.2f}, {positions[:,1].mean():.2f}) | "
            f"{sector[0]} {sector[1]}/{len(members)} | "
            f"{' → '.join(turn[0]) if turn[0] else '不经过路口'} {turn[1]}/{len(members)} | "
            f"{', '.join(cases)} |"
        )
    lines.extend(
        [
            "",
            "## ‘为什么不在路口拐走’的输出约定",
            "",
            "每个路线候选必须同时记录 `routeStartRole`、`routeJunctionTurns`、训练支持 case、",
            "起点区域惩罚和路口惩罚。这样管理员看到的不是‘模型觉得不像’，而是例如：",
            "‘部件 220 的第 2/3 笔在排除当前目标后的两个样本中起向和起点区域稳定；",
            "本路线在真实路口作出了与人工样本相反的 clockwise 转弯，因此降权’。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    records, _sizes = PDF.parse_pdf_cells(args.bbox_cache)
    page_cache: dict[int, str] = {}
    examples = []
    for annotation in args.annotations:
        examples.extend(
            learn_example(
                annotation, rows, records, page_cache, args.pdf, args.canvas
            )
        )
    model = build_rule_model(examples)
    by_signature = model["examplesBySignature"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2), "utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(model), "utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "caseCount": model["caseCount"],
                "strokeCount": len(examples),
                "signatureCount": len(by_signature),
                "scoringRuleCount": sum(
                    len({example["case"] for example in members}) >= 2
                    for members in by_signature.values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
