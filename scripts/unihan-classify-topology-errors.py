"""Classify reviewed cascade errors by the structural evidence still missing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def keyed(payload: dict) -> dict[tuple[int, str], dict]:
    return {
        (row["unicode"], row["source"]): row
        for row in payload.get("results", [])
    }


def classify(cascade_payload: dict, evidence_payload: dict) -> dict:
    evidence = keyed(evidence_payload)
    rows = []
    for result in cascade_payload.get("results", []):
        if result.get("correct") or result.get("abstained"):
            continue
        key = (result["unicode"], result["source"])
        source = evidence.get(key, {})
        report = source.get("recursiveTopology", {})
        unique = report.get("uniqueStrokeIndices", [])
        subtree = report.get("differingSubtreeStrokeIndices", [])
        signatures = report.get("candidateTopologySignatures", [])
        categories = []
        reasons = []

        if report.get("principalAxis"):
            categories.append("terminal-stroke-direction")
            reasons.append("terminal stroke isolated; direction/position gate is decisive")
        if len(subtree) >= 2 and len({len(indices) for indices in subtree}) > 1:
            categories.append("missing-or-extra-stroke")
            reasons.append("candidate sibling subtrees have different stroke counts")
        if signatures and len({item.get("junctions") for item in signatures}) > 1:
            categories.append("intersection-or-extra-connection")
            reasons.append("candidate branch/cross counts differ")
        if signatures and len({item.get("components") for item in signatures}) > 1:
            categories.append("component-boundary")
            reasons.append("candidate connected-component counts differ")
        pdf_components = report.get("pdfComponentCount")
        candidate_components = report.get("candidateComponentCounts", [])
        if (
            pdf_components is not None
            and candidate_components
            and pdf_components > max(candidate_components) + 2
        ):
            categories.append("local-window-neighbour-contamination")
            reasons.append("PDF window contains substantially more components than candidates")
        if source.get("candidateSetIncomplete"):
            categories.append("incomplete-candidate-set")
            reasons.append("all supplied candidates share a structured residual")
        if not categories:
            categories.append("insufficient-local-evidence")
            reasons.append("no currently encoded hard topology invariant separates candidates")

        rows.append(
            {
                **{key: result.get(key) for key in (
                    "unicode",
                    "source",
                    "expectedGlyphId",
                    "predictedGlyphId",
                    "method",
                    "margin",
                    "split",
                )},
                "character": chr(result["unicode"]),
                "categories": categories,
                "reasons": reasons,
                "uniqueStrokeCounts": [len(indices) for indices in unique],
                "subtreeStrokeCounts": [len(indices) for indices in subtree],
            }
        )

    counts = Counter(category for row in rows for category in row["categories"])
    return {
        "metadata": {
            "wrongAttempted": len(rows),
            "codepoints": len({row["unicode"] for row in rows}),
            "categoryCounts": dict(sorted(counts.items())),
            "categoriesMayOverlap": True,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify(
        json.loads(args.cascade.read_text("utf-8")),
        json.loads(args.evidence.read_text("utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
