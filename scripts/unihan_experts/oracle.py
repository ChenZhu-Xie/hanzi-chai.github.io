"""Benchmark-only oracle ranking. This module must never drive inference."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


INELIGIBLE_STATUSES = frozenset(
    {
        "truth-isolation-failure",
        "timeout",
        "unsupported",
        "process-error",
        "missing-output",
        "malformed-audit",
    }
)


def _metrics(cell: Mapping[str, object]) -> Mapping[str, object]:
    evaluation = cell.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return {}
    metrics = evaluation.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _number(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return fallback


def _review_reasons(cell: Mapping[str, object]) -> list[object]:
    reasons = cell.get("reviewReasons")
    if isinstance(reasons, list):
        return reasons
    evaluation = cell.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(
        evaluation.get("reviewReasons"), list
    ):
        return list(evaluation["reviewReasons"])
    return []


def oracle_sort_key(cell: Mapping[str, object]) -> tuple[object, ...]:
    status = str(cell.get("status"))
    invalid = status in INELIGIBLE_STATUSES
    incomplete = not bool(cell.get("complete"))
    metrics = _metrics(cell)
    return (
        invalid,
        incomplete,
        len(_review_reasons(cell)),
        -_number(metrics.get("strokeMacroIoU"), -math.inf),
        -_number(metrics.get("componentMacroIoU"), -math.inf),
        _number(metrics.get("meanDirectedSequenceDtwPercent"), math.inf),
        _number(metrics.get("meanStartErrorPercent"), math.inf),
        _number(metrics.get("meanEndErrorPercent"), math.inf),
        str(cell.get("expertId", "")),
    )


def rank_case_results(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [dict(cell) for cell in sorted(cells, key=oracle_sort_key)]


def select_oracle_winner(
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ranked = rank_case_results(cells)
    eligible = [
        cell
        for cell in ranked
        if str(cell.get("status")) not in INELIGIBLE_STATUSES
        and bool(cell.get("complete"))
    ]
    if not eligible:
        raise ValueError("case has no complete truth-isolated oracle candidate")
    winner = dict(eligible[0])
    winner["oracle"] = {
        "usesHumanTruth": True,
        "purpose": "benchmark-only",
        "rankContract": [
            "valid",
            "complete",
            "fewer-review-reasons",
            "stroke-macro-iou",
            "component-macro-iou",
            "directed-sequence-error",
            "start-error",
            "end-error",
            "expert-id",
        ],
    }
    return winner
