"""Source-aware stroke expectations and residual pen-down proposals."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SECTORS = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
KNOWN_SOURCE_CONVENTIONS = frozenset({"G", "H", "J", "K", "KP", "N", "T", "U", "UK", "V"})


def _load_transfer():
    name = "unihan_stroke_transfer_for_order"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("unihan-stroke-transfer.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class OrderEvidence:
    level: int
    provenance: str
    rule: str
    hard: bool


@dataclass(frozen=True)
class NormativeEntry:
    source: str
    codepoint: int
    features: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class NormativeCatalog:
    entries: dict[tuple[str, int], NormativeEntry]


@dataclass(frozen=True)
class StrokeExpectation:
    stroke_index: int
    component_id: int
    occurrence: int
    component_ordinal: int
    feature: str
    expected_sector: str
    expected_turns: tuple[str, ...]
    closes_component: bool
    evidence: tuple[OrderEvidence, ...]


@dataclass(frozen=True)
class PenDownCandidate:
    point_index: int
    outgoing_edge_ids: tuple[int, ...]
    score: float
    evidence: tuple[dict, ...]
    hard_rejections: tuple[str, ...]


def load_normative_catalog(path: Path | None) -> NormativeCatalog:
    if path is None:
        return NormativeCatalog(entries={})
    document = json.loads(Path(path).read_text("utf-8-sig"))
    if int(document.get("schemaVersion", 0)) != 1:
        raise ValueError("unsupported normative stroke-order catalog schema")
    entries: dict[tuple[str, int], NormativeEntry] = {}
    for raw in document.get("entries", []):
        source = str(raw["source"]).upper()
        codepoint_text = str(raw["unicode"]).upper().removeprefix("U+")
        codepoint = int(codepoint_text, 16)
        entry = NormativeEntry(
            source=source,
            codepoint=codepoint,
            features=tuple(str(value) for value in raw["features"]),
            provenance=str(raw["provenance"]),
        )
        key = (source, codepoint)
        if key in entries:
            raise ValueError(f"duplicate normative stroke-order entry {source} U+{codepoint:04X}")
        entries[key] = entry
    return NormativeCatalog(entries=entries)


def _direction_sector(points: np.ndarray) -> str:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return "unknown"
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    end = points[1 + int(np.argmax(distances))]
    vector = end - points[0]
    angle = (math.degrees(math.atan2(float(vector[1]), float(vector[0]))) + 360.0) % 360.0
    return SECTORS[int((angle + 22.5) // 45.0) % 8]


def compile_stroke_expectations(
    strokes: list[dict],
    source: str,
    codepoint: int,
    catalog: NormativeCatalog | None = None,
) -> list[StrokeExpectation]:
    source = str(source).upper()
    catalog = catalog or NormativeCatalog(entries={})
    exact = catalog.entries.get((source, int(codepoint)))
    transfer = _load_transfer()
    verified_orders = transfer.VERIFIED_LEAF_STROKE_ORDERS

    groups: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        key = (int(stroke["componentId"]), int(stroke.get("occurrence", 0)))
        groups.setdefault(key, []).append(index)

    output = []
    for index, stroke in enumerate(strokes):
        component_id = int(stroke["componentId"])
        occurrence = int(stroke.get("occurrence", 0))
        key = (component_id, occurrence)
        members = groups[key]
        ordinal = members.index(index) + 1
        feature = str(stroke.get("feature") or "未知")
        evidence = []

        verified = verified_orders.get(component_id)
        group_features = tuple(str(strokes[item].get("feature")) for item in members)
        if verified is not None and tuple(verified) == group_features:
            evidence.append(
                OrderEvidence(
                    level=2,
                    provenance=f"verified leaf component {component_id}",
                    rule="exact-leaf-stroke-order",
                    hard=True,
                )
            )

        if exact is not None:
            if len(exact.features) == len(strokes) and exact.features[index] == feature:
                evidence.append(
                    OrderEvidence(
                        level=3,
                        provenance=exact.provenance,
                        rule="source-character-stroke-order",
                        hard=True,
                    )
                )
            else:
                evidence.append(
                    OrderEvidence(
                        level=3,
                        provenance=exact.provenance,
                        rule="source-character-order-conflict",
                        hard=True,
                    )
                )
        else:
            evidence.append(
                OrderEvidence(
                    level=5,
                    provenance=(
                        f"{source} source has no exact local normative entry"
                        if source in KNOWN_SOURCE_CONVENTIONS
                        else f"unknown source convention {source}"
                    ),
                    rule="source-order-unavailable",
                    hard=False,
                )
            )

        evidence.extend(
            (
                OrderEvidence(
                    level=4,
                    provenance="repository candidate directed stroke grammar",
                    rule="candidate-stroke-order-and-direction",
                    hard=True,
                ),
                OrderEvidence(
                    level=5,
                    provenance="basic scoped stroke-order rules",
                    rule="top-to-bottom-left-to-right-within-compatible-scope",
                    hard=False,
                ),
            )
        )
        output.append(
            StrokeExpectation(
                stroke_index=index,
                component_id=component_id,
                occurrence=occurrence,
                component_ordinal=ordinal,
                feature=feature,
                expected_sector=_direction_sector(np.asarray(stroke["points"], dtype=float)),
                expected_turns=(),
                closes_component=index == members[-1],
                evidence=tuple(sorted(evidence, key=lambda item: item.level)),
            )
        )
    return output


def rank_pen_down_candidates(
    expectation: StrokeExpectation,
    ledger,
    directed,
    graph,
    learned_rule: dict | None = None,
) -> list[PenDownCandidate]:
    height, width = ledger.unexplained.shape
    grouped: dict[int, list[int]] = {}
    for edge in directed.edges:
        grouped.setdefault(int(edge.point_indices[0]), []).append(edge.edge_id)

    output = []
    for point_index, edge_ids in grouped.items():
        y, x = (int(value) for value in graph.points[point_index])
        compatible = tuple(
            edge_id
            for edge_id in edge_ids
            if expectation.expected_sector == "unknown"
            or directed.edges[edge_id].start_sector == expectation.expected_sector
        )
        rejections = []
        if not bool(ledger.available[y, x]):
            rejections.append("start-in-explained-non-contact-ink")
        if not compatible:
            rejections.append("wrong-start-direction")

        y_score = y / max(1, height - 1)
        x_score = x / max(1, width - 1)
        score = y_score + 0.15 * x_score
        evidence = [
            {
                "rule": "scoped-top-to-bottom-left-to-right",
                "value": round(score, 6),
                "point": [x, y],
            }
        ]
        if learned_rule and learned_rule.get("glyphStartMean") is not None:
            expected = np.asarray(learned_rule["glyphStartMean"], dtype=float)
            observed = np.asarray([x / max(1, width - 1), y / max(1, height - 1)])
            learned_distance = float(np.linalg.norm(observed - expected))
            score += min(0.2, learned_distance * 0.2)
            evidence.append(
                {
                    "rule": "bounded-learned-start-prior",
                    "value": round(learned_distance, 6),
                }
            )
        output.append(
            PenDownCandidate(
                point_index=point_index,
                outgoing_edge_ids=compatible,
                score=score,
                evidence=tuple(evidence),
                hard_rejections=tuple(rejections),
            )
        )
    output.sort(key=lambda item: (bool(item.hard_rejections), item.score, item.point_index))
    return output
