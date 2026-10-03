"""Audit human-truth evidence rules outside the annotation corpus.

This script deliberately treats the reviewed benchmark answers as scoring data,
not as inputs to either evidence rule.  It records counterexamples so a rule
that improves the small stroke-annotation corpus cannot silently become an
automatic-write heuristic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_transfer_module():
    path = Path(__file__).with_name("unihan-stroke-transfer.py")
    spec = importlib.util.spec_from_file_location("unihan_evidence_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = load_transfer_module()


def annotation_key(path: Path) -> tuple[int, str]:
    metadata = json.loads(path.read_text("utf-8-sig"))["metadata"]
    return (
        int(str(metadata["unicode"]).removeprefix("U+"), 16),
        str(metadata["source"]).upper(),
    )


def summarize(items: list[dict]) -> dict:
    correct = sum(item["correct"] for item in items)
    return {
        "triggered": len(items),
        "correct": correct,
        "precisionWhenTriggered": round(correct / len(items), 6) if items else None,
        "counterexamples": [item for item in items if not item["correct"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--topology-evidence",
        type=Path,
        required=True,
        help="output from unihan-render-match.py on the same candidates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    catalog = TRANSFER.candidate_catalog(rows)
    annotations = list(dict.fromkeys(path.resolve() for path in args.annotations))
    annotated_keys = {annotation_key(path) for path in annotations}

    same_source_items = []
    for row in rows:
        if len(row["candidates"]) < 2:
            continue
        candidates = {
            int(glyph_text): TRANSFER.candidate_strokes(row, int(glyph_text))
            for glyph_text in row["candidates"]
        }
        for source, expected_text in row.get("sourceGlyphs", {}).items():
            key = (int(row["unicode"]), str(source).upper())
            if key in annotated_keys:
                continue
            report = TRANSFER.compare_verified_component_coverage(
                candidates,
                annotations,
                reference_candidates=catalog,
                target_source=source,
            )
            evidence_glyph_id = report["evidenceGlyphId"]
            if evidence_glyph_id is None:
                continue
            expected = int(expected_text)
            same_source_items.append(
                {
                    "case": f"U+{key[0]:04X}-{key[1]}",
                    "expectedGlyphId": expected,
                    "evidenceGlyphId": evidence_glyph_id,
                    "correct": evidence_glyph_id == expected,
                    "discriminatingSameSourceVerifiedIds": report[
                        "discriminatingSameSourceVerifiedIds"
                    ],
                }
            )

    rows_by_unicode = {int(row["unicode"]): row for row in rows}
    topology_document = json.loads(args.topology_evidence.read_text("utf-8"))
    cross_source_items = []
    for result in topology_document["results"]:
        row = rows_by_unicode[int(result["unicode"])]
        optical = int(result["opticalPredictionGlyphId"])
        topology = result.get("recursiveTopologyPredictionGlyphId")
        if topology is None:
            continue
        topology = int(topology)
        other_source_glyphs = sorted(
            {
                int(glyph_id)
                for source, glyph_id in row.get("sourceGlyphs", {}).items()
                if source != result["source"]
            }
        )
        if topology == optical or optical not in other_source_glyphs:
            continue
        expected = int(result["expectedGlyphId"])
        cross_source_items.append(
            {
                "case": f"U+{int(result['unicode']):04X}-{result['source']}",
                "split": result.get("split"),
                "expectedGlyphId": expected,
                "opticalGlyphId": optical,
                "topologyEvidenceGlyphId": topology,
                "otherSourceVerifiedGlyphIds": other_source_glyphs,
                "correct": topology == expected,
            }
        )

    payload = {
        "protocol": {
            "answersUsedOnlyAfterEvidenceIsFrozen": True,
            "automaticWriteEnabled": False,
            "sameSourcePopulation": (
                "reviewed benchmark sources excluding all stroke-annotation keys"
            ),
            "crossSourcePopulation": "all reviewed benchmark sources",
        },
        "sameSourceUniqueLeaf": summarize(same_source_items),
        "crossSourceDivergence": {
            "all": summarize(cross_source_items),
            "train": summarize(
                [item for item in cross_source_items if item["split"] == "train"]
            ),
            "test": summarize(
                [item for item in cross_source_items if item["split"] == "test"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
