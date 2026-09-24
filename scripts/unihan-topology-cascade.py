"""Combine recursive stroke and complete-subtree topology evidence.

The cascade encodes evidence priority rather than a learned visual weight:

1. A unique exact connected-component match in the complete recursive target
   window wins. This is the U+6461 case where dropping 白 corrupts the answer.
2. If two sibling subtrees differ in exactly one terminal stroke apiece and
   retain the same stroke count, use the conservative 64px direction result.
   Short 横/撇 evidence survives there without being diluted by a common hook.
3. Otherwise use the 128px complete-subtree result for extra/missing strokes,
   component grouping and connection topology.

Optical whole-glyph ranking never outranks these structural stages. Every stage
may still abstain; this script is evaluation-only and enables no database write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def keyed(payload: dict) -> dict[tuple[int, str], dict]:
    return {
        (row["unicode"], row["source"]): row
        for row in payload.get("results", [])
    }


def choose(low: dict, high: dict, subtree: dict) -> tuple[int | None, str, float]:
    high_report = high.get("recursiveTopology", {})
    principal_axis = high_report.get("principalAxis") or {}
    topology_index = principal_axis.get("candidateIndex")
    topology_candidate_ids = high_report.get("candidateGlyphIds", [])
    if (
        topology_index is not None
        and float(principal_axis.get("margin", 0.0)) >= 15.0
        and 0 <= int(topology_index) < len(topology_candidate_ids)
    ):
        return (
            int(topology_candidate_ids[int(topology_index)]),
            "terminal-principal-axis",
            float(principal_axis["margin"]),
        )

    signature = high_report.get("topologySignatureDecision") or {}
    topology_index = signature.get("candidateIndex")
    if (
        topology_index is not None
        and float(signature.get("margin", 0.0)) >= 3.0
        and 0 <= int(topology_index) < len(topology_candidate_ids)
    ):
        return (
            int(topology_candidate_ids[int(topology_index)]),
            "recursive-graph-consensus",
            float(signature["margin"]),
        )

    exact = subtree.get("exactComponentPredictionGlyphId")
    if exact is not None:
        component_distances = sorted(
            float(features[0])
            for features in subtree.get("candidateFeatureDistances", {}).values()
        )
        margin = (
            component_distances[1] - component_distances[0]
            if len(component_distances) > 1
            else 0.0
        )
        return int(exact), "complete-subtree-component-topology", margin

    report = low.get("recursiveTopology", {})
    unique = report.get("uniqueStrokeIndices", [])
    differing = report.get("differingSubtreeStrokeIndices", [])
    single_stroke_substitution = (
        len(unique) == 2
        and all(len(indices) == 1 for indices in unique)
        and len(differing) == 2
        and len(differing[0]) == len(differing[1])
    )
    if single_stroke_substitution:
        return (
            low.get("recursiveTopologyPredictionGlyphId"),
            "terminal-stroke-direction",
            float(report.get("margin", 0.0)),
        )
    return (
        high.get("recursiveTopologyPredictionGlyphId"),
        "recursive-subtree-topology",
        float(high_report.get("margin", 0.0)),
    )


def calibrate_thresholds(rows: list[dict]) -> dict[str, float]:
    """Find per-method margins with zero errors on family-disjoint training rows."""
    thresholds = {}
    methods = {row["method"] for row in rows}
    for method in methods:
        ordered = sorted(
            (
                row
                for row in rows
                if row["method"] == method and not row["abstained"]
            ),
            key=lambda row: row["margin"],
            reverse=True,
        )
        correct = 0
        selected = None
        index = 0
        while index < len(ordered):
            margin = ordered[index]["margin"]
            group_end = index
            while (
                group_end < len(ordered)
                and ordered[group_end]["margin"] == margin
            ):
                correct += ordered[group_end]["correct"]
                group_end += 1
            if correct == group_end:
                selected = margin
            index = group_end
        if selected is not None:
            thresholds[method] = selected
    return thresholds


def accepted_summary(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row.get("safeAccepted")]
    correct = sum(row["correct"] for row in accepted)
    return {
        "evaluated": len(rows),
        "accepted": len(accepted),
        "correct": correct,
        "precision": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(rows) if rows else None,
    }


def evaluate(
    low_payload: dict,
    high_payload: dict,
    subtree_payload: dict,
    baseline_payload: dict | None = None,
) -> dict:
    low = keyed(low_payload)
    high = keyed(high_payload)
    subtree = keyed(subtree_payload)
    baseline = keyed(baseline_payload or {})
    rows = []
    for key in sorted(low):
        if key not in high or key not in subtree:
            continue
        low_row = low[key]
        predicted, method, margin = choose(low_row, high[key], subtree[key])
        # Only the vector principal-axis decision is a sufficiently specific
        # hard override. Graph-count consensus remains useful review evidence,
        # but its family-disjoint errors make it unsafe as an automatic gate.
        if baseline and method != "terminal-principal-axis" and key in baseline:
            baseline_row = baseline[key]
            predicted = baseline_row.get("predictedGlyphId")
            method = f"baseline/{baseline_row.get('method', 'unknown')}"
            margin = float(baseline_row.get("margin", 0.0))
        expected = low_row.get("expectedGlyphId")
        rows.append(
            {
                "unicode": key[0],
                "source": key[1],
                "expectedGlyphId": expected,
                "predictedGlyphId": predicted,
                "correct": predicted is not None and predicted == expected,
                "abstained": predicted is None,
                "method": method,
                "margin": margin,
                "split": low_row.get("split"),
            }
        )
    train = [row for row in rows if row.get("split") == "train"]
    test = [row for row in rows if row.get("split") == "test"]
    thresholds = calibrate_thresholds(train)
    for row in rows:
        threshold = thresholds.get(row["method"])
        row["safeAccepted"] = (
            not row["abstained"]
            and threshold is not None
            and row["margin"] >= threshold
        )
    attempted = [row for row in rows if not row["abstained"]]
    correct = sum(row["correct"] for row in attempted)
    return {
        "metadata": {
            "method": "recursive topology cascade; optical evidence is tie-break only",
            "evaluated": len(rows),
            "attempted": len(attempted),
            "correct": correct,
            "accuracyWhenAttempted": correct / len(attempted) if attempted else None,
            "coverage": len(attempted) / len(rows) if rows else None,
            "safeThresholds": thresholds,
            "safeTrain": accepted_summary(train),
            "safeHeldOut": accepted_summary(test),
            "baselineFallbackEnabled": bool(baseline),
            "automaticWriteEnabled": False,
        },
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-resolution", type=Path, required=True)
    parser.add_argument("--high-resolution", type=Path, required=True)
    parser.add_argument("--subtree-topology", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="fall back to a previously calibrated raster cascade",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.low_resolution.read_text("utf-8")),
        json.loads(args.high_resolution.read_text("utf-8")),
        json.loads(args.subtree_topology.read_text("utf-8")),
        json.loads(args.baseline.read_text("utf-8")) if args.baseline else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
