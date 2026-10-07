"""Deterministic JSON and HTML reports for the historical expert matrix."""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from .oracle import rank_case_results, select_oracle_winner


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metrics(cell: Mapping[str, object]) -> Mapping[str, object]:
    evaluation = cell.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(
        evaluation.get("metrics"), Mapping
    ):
        return evaluation["metrics"]
    return {}


def _process(cell: Mapping[str, object], phase: str) -> Mapping[str, object]:
    value = cell.get(phase)
    if isinstance(value, Mapping) and isinstance(value.get("process"), Mapping):
        return value["process"]
    return {}


def _artifact(cell: Mapping[str, object], key: str) -> str | None:
    evaluation = cell.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    artifacts = evaluation.get("rawArtifacts")
    if not isinstance(artifacts, Mapping):
        return None
    value = artifacts.get(key)
    return str(value) if value else None


def _artifact_href(value: str) -> str:
    path = Path(value)
    return path.resolve().as_uri() if path.is_absolute() else value.replace("\\", "/")


def _oracle_report(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for cell in cells:
        groups.setdefault(str(cell.get("caseId")), []).append(cell)
    cases: dict[str, object] = {}
    for case_id in sorted(groups):
        ranked = rank_case_results(groups[case_id])
        try:
            winner = select_oracle_winner(groups[case_id])
            winner_id = winner["expertId"]
        except ValueError:
            winner_id = None
        cases[case_id] = {
            "winnerExpertId": winner_id,
            "ranked": ranked,
        }
    return {
        "schemaVersion": 1,
        "usesHumanTruth": True,
        "purpose": "benchmark-only",
        "cases": cases,
    }


def _capability_report(
    cells: Sequence[Mapping[str, object]],
    oracle: Mapping[str, object],
    declared: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    winners = {
        str(case_id): value.get("winnerExpertId")
        for case_id, value in oracle["cases"].items()
    }
    experts = sorted({str(cell.get("expertId")) for cell in cells} | set(declared))
    result: dict[str, object] = {}
    for expert in experts:
        own = [cell for cell in cells if cell.get("expertId") == expert]
        win_cases = sorted(case for case, winner in winners.items() if winner == expert)
        complete_cases = sorted(
            str(cell.get("caseId")) for cell in own if cell.get("complete") is True
        )
        ious = [
            float(value)
            for cell in own
            if isinstance((value := _metrics(cell).get("strokeMacroIoU")), (int, float))
        ]
        failures: dict[str, int] = {}
        for cell in own:
            status = str(cell.get("status"))
            if status not in {"complete", "needs-review"}:
                failures[status] = failures.get(status, 0) + 1
        result[expert] = {
            "declared": list(declared.get(expert, ())),
            "measured": {
                "wins": len(win_cases),
                "winCases": win_cases,
                "completeCases": complete_cases,
                "meanStrokeMacroIoU": fmean(ious) if ious else None,
                "failureCounts": dict(sorted(failures.items())),
            },
        }
    return {"schemaVersion": 1, "experts": result}


def _resource_report(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    experts = sorted({str(cell.get("expertId")) for cell in cells})
    result: dict[str, object] = {}
    for expert in experts:
        own = [cell for cell in cells if cell.get("expertId") == expert]
        processes = [
            _process(cell, phase)
            for cell in own
            for phase in ("inference", "evaluation")
        ]
        result[expert] = {
            "cells": len(own),
            "wallSeconds": sum(float(item.get("wallSeconds") or 0) for item in processes),
            "cpuSeconds": sum(float(item.get("cpuSeconds") or 0) for item in processes),
            "peakMemoryBytes": max(
                (int(item.get("peakMemoryBytes") or 0) for item in processes),
                default=0,
            ),
        }
    return {"schemaVersion": 1, "experts": result}


def _html_report(
    cells: Sequence[Mapping[str, object]], oracle: Mapping[str, object], run_id: str
) -> str:
    winner_by_case = {
        case_id: value.get("winnerExpertId")
        for case_id, value in oracle["cases"].items()
    }
    rows = []
    for cell in sorted(cells, key=lambda item: (str(item.get("caseId")), str(item.get("expertId")))):
        case_id = str(cell.get("caseId"))
        expert_id = str(cell.get("expertId"))
        raw_html = _artifact(cell, "htmlPath")
        link = (
            f'<a href="{html.escape(_artifact_href(raw_html))}">raw HTML</a>'
            if raw_html
            else "—"
        )
        metric = _metrics(cell).get("strokeMacroIoU")
        rows.append(
            "<tr>"
            f"<td>{html.escape(case_id)}</td>"
            f"<td>{html.escape(expert_id)}</td>"
            f"<td>{'winner' if winner_by_case.get(case_id) == expert_id else ''}</td>"
            f"<td>{html.escape(str(cell.get('status')))}</td>"
            f"<td>{html.escape(str(metric))}</td>"
            f"<td>{link}</td>"
            "<td>not-run</td><td>not-run</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Ink expert oracle {html.escape(run_id)}</title>
<style>body{{font-family:system-ui;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}.warning{{background:#fff4ce;padding:12px}}</style></head>
<body><h1>Immutable ink expert oracle: {html.escape(run_id)}</h1>
<p class="warning">Benchmark only. This report uses human truth and must not select a production result.</p>
<table><thead><tr><th>Case</th><th>Expert</th><th>Oracle</th><th>Status</th><th>Stroke IoU</th><th>Evidence</th><th>Assisted</th><th>Unassisted shadow</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>
"""


def generate_reports(
    cells: Sequence[Mapping[str, object]],
    output_root: Path,
    *,
    run_id: str,
    declared_capabilities: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Path]:
    ordered = [
        dict(cell)
        for cell in sorted(
            cells, key=lambda item: (str(item.get("expertId")), str(item.get("caseId")))
        )
    ]
    oracle = _oracle_report(ordered)
    capabilities = _capability_report(
        ordered, oracle, declared_capabilities or {}
    )
    resources = _resource_report(ordered)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "matrix": output_root / "matrix.json",
        "oracle": output_root / "oracle.json",
        "capabilities": output_root / "capabilities.json",
        "resources": output_root / "resources.json",
        "html": output_root / "index.html",
    }
    _write_json(
        paths["matrix"],
        {
            "schemaVersion": 1,
            "runId": run_id,
            "assisted": "not-run",
            "unassistedShadow": "not-run",
            "cells": ordered,
        },
    )
    _write_json(paths["oracle"], oracle)
    _write_json(paths["capabilities"], capabilities)
    _write_json(paths["resources"], resources)
    paths["html"].write_text(
        _html_report(ordered, oracle, run_id), encoding="utf-8"
    )
    return paths
