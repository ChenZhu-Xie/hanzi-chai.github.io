"""Leave-one-annotation-out glyph selection using verified component templates.

The held-out JSON is not opened until all candidate scores are frozen.  Its
filename supplies only Unicode and source; the candidate glyph ID remains the
answer and is read strictly during the scoring phase.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


def load_transfer_module():
    path = Path(__file__).with_name("unihan-stroke-transfer.py")
    spec = importlib.util.spec_from_file_location("unihan_verified_template_loocv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = load_transfer_module()


def load_render_match_module():
    path = Path(__file__).with_name("unihan-render-match.py")
    spec = importlib.util.spec_from_file_location("unihan_verified_template_topology", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RENDER_MATCH = load_render_match_module()
CASE_NAME = re.compile(r"U\+([0-9A-F]+)-([A-Z]+)-.+-annotations\.json$", re.IGNORECASE)


def case_key(path: Path) -> tuple[int, str]:
    match = CASE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"cannot derive case from {path.name}")
    return int(match.group(1), 16), match.group(2).upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=256)
    args = parser.parse_args()

    paths = list(dict.fromkeys(path.resolve() for path in args.annotations))
    keys = [case_key(path) for path in paths]
    if len(keys) != len(set(keys)):
        raise ValueError("each annotation must have a unique Unicode/source key")
    rows = json.loads(args.candidates.read_text("utf-8"))["rows"]
    reference_candidates = TRANSFER.candidate_catalog(rows)
    rows_by_unicode = {int(row["unicode"]): row for row in rows}
    records, _sizes = TRANSFER.PDF.parse_pdf_cells(args.bbox_cache)
    records_by_key = {(item["unicode"], item["source"]): item for item in records}
    page_cache: dict[int, str] = {}
    evidence_size = 128
    rendered = {}
    stroke_layers = {}
    for row in rows:
        for glyph_text, candidate_definition in row["candidates"].items():
            glyph_id = int(glyph_text)
            if glyph_id in rendered:
                continue
            rendered[glyph_id] = RENDER_MATCH.render_strokes(
                candidate_definition,
                size=max(128, evidence_size * 2),
                output_size=evidence_size,
            )
            stroke_layers[glyph_id] = RENDER_MATCH.render_stroke_layers(
                candidate_definition,
                size=max(128, evidence_size * 2),
                output_size=evidence_size,
            )
    folds = []

    for position, (heldout, (codepoint, source)) in enumerate(zip(paths, keys), start=1):
        row = rows_by_unicode[codepoint]
        record = records_by_key[(codepoint, source)]
        if record["page"] not in page_cache:
            page_cache[record["page"]] = TRANSFER.MATCHER.load_pdf_page_svg(
                args.pdf, record["page"]
            )
        source_mask = TRANSFER.MATCHER.render_pdf_vector_cell(
            page_cache[record["page"]],
            TRANSFER.MATCHER.chart_glyph_bbox(record["bbox"]),
            size=args.canvas,
        ) < 224
        target = TRANSFER.normalize_target(source_mask, args.canvas)
        references = [path for path in paths if path != heldout]
        candidates = {}
        candidate_ids = [int(glyph_text) for glyph_text in row["candidates"]]

        pdf_skeleton = RENDER_MATCH.pdf_vector_skeleton(
            page_cache[record["page"]], record["bbox"], output_size=evidence_size
        )
        candidate_images, alignment = RENDER_MATCH.align_candidates_to_pdf(
            pdf_skeleton, [rendered[glyph_id] for glyph_id in candidate_ids]
        )
        recursive_scores, recursive_report = RENDER_MATCH.recursive_stroke_distances(
            pdf_skeleton,
            candidate_images,
            [row["candidates"][str(glyph_id)] for glyph_id in candidate_ids],
            [stroke_layers[glyph_id] for glyph_id in candidate_ids],
            alignment,
            [
                (
                    row.get("candidateLeafStrokeIds")
                    or row.get("candidateLeafIds")
                    or {}
                ).get(str(glyph_id), [])
                for glyph_id in candidate_ids
            ]
            if row.get("candidateLeafStrokeIds") or row.get("candidateLeafIds")
            else None,
        )
        recursive_report["candidateGlyphIds"] = candidate_ids

        # Prediction phase: the held-out JSON has not been opened.
        for glyph_id in candidate_ids:
            strokes = TRANSFER.candidate_strokes(row, glyph_id)
            selected, search = TRANSFER.search_component_templates_globally(
                strokes,
                references,
                target,
                args.canvas,
                target_unicode=f"U+{codepoint:04X}",
                target_source=source,
                reference_candidates=reference_candidates,
            )
            reconstruction = TRANSFER.candidate_alignment_metrics(
                selected, target, args.canvas
            )
            reconstruction.pop("snapped", None)
            candidates[glyph_id] = {
                "score": float(reconstruction["score"]),
                "templateSearch": search,
                "reconstruction": reconstruction,
            }
        ranking = sorted((item["score"], glyph_id) for glyph_id, item in candidates.items())
        template_predicted = ranking[0][1]
        recursive_ranking = sorted(zip(recursive_scores, candidate_ids))
        exact_component_index = recursive_report.get("exactComponentCandidateIndex")
        recursive_decisive = (
            recursive_report.get("decision") == "candidate"
            and recursive_ranking
            and recursive_ranking[0][0] < float("inf")
        )
        topology_predicted = (
            candidate_ids[exact_component_index]
            if recursive_decisive and exact_component_index is not None
            else recursive_ranking[0][1]
            if recursive_decisive
            else None
        )
        other_source_same_character_glyphs = []
        for reference in references:
            reference_metadata = json.loads(
                reference.read_text("utf-8-sig")
            ).get("metadata", {})
            if (
                str(reference_metadata.get("unicode", "")).upper()
                == f"U+{codepoint:04X}"
                and str(reference_metadata.get("source", "")).upper() != source
                and isinstance(reference_metadata.get("candidateGlyphId"), int)
            ):
                other_source_same_character_glyphs.append(
                    int(reference_metadata["candidateGlyphId"])
                )
        other_source_same_character_glyphs = sorted(
            set(other_source_same_character_glyphs)
        )
        coverage_comparison = TRANSFER.compare_verified_component_coverage(
            {
                glyph_id: TRANSFER.candidate_strokes(row, glyph_id)
                for glyph_id in candidate_ids
            },
            references,
            reference_candidates=reference_candidates,
            target_source=source,
        )
        coverage_suggestion = coverage_comparison["evidenceGlyphId"]
        cross_source_suggestion = (
            topology_predicted
            if (
            topology_predicted is not None
            and topology_predicted != template_predicted
            and template_predicted in other_source_same_character_glyphs
            )
            else None
        )
        predicted = template_predicted
        selection_method = "verified-template-whole-glyph"
        review_suggestion = (
            coverage_suggestion
            if coverage_suggestion is not None
            else cross_source_suggestion
            if cross_source_suggestion is not None
            else template_predicted
        )
        review_suggestion_method = (
            "unique-same-source-verified-leaf-evidence"
            if coverage_suggestion is not None
            else "cross-source-divergence-topology-evidence"
            if cross_source_suggestion is not None
            else "verified-template-whole-glyph"
        )

        # Scoring phase: first and only read of held-out answer JSON.
        heldout_document = json.loads(heldout.read_text("utf-8-sig"))
        expected = int(heldout_document["metadata"]["candidateGlyphId"])
        fold = {
            "case": f"U+{codepoint:04X}-{source}",
            "predictionFrozenBeforeHeldoutRead": True,
            "referenceCount": len(references),
            "expectedGlyphId": expected,
            "predictedGlyphId": predicted,
            "selectionMethod": selection_method,
            "reviewSuggestedGlyphId": review_suggestion,
            "reviewSuggestionMethod": review_suggestion_method,
            "templatePredictedGlyphId": template_predicted,
            "experimentalTopologyPredictedGlyphId": topology_predicted,
            "otherSourceSameCharacterVerifiedGlyphIds": (
                other_source_same_character_glyphs
            ),
            "verifiedCoverageComparison": coverage_comparison,
            "correct": predicted == expected,
            "reviewSuggestionCorrect": review_suggestion == expected,
            "templateCorrect": template_predicted == expected,
            "experimentalTopologyCorrect": topology_predicted == expected,
            "automaticWriteEnabled": False,
            "margin": round(ranking[1][0] - ranking[0][0], 6) if len(ranking) > 1 else None,
            "recursiveTopology": recursive_report,
            "candidates": {str(key): value for key, value in candidates.items()},
        }
        folds.append(fold)
        verdict = "OK" if fold["correct"] else "WRONG"
        print(
            f"[{position}/{len(paths)}] {fold['case']} {verdict} "
            f"expected={expected} predicted={predicted} margin={fold['margin']}",
            flush=True,
        )

    correct = sum(item["correct"] for item in folds)
    review_correct = sum(item["reviewSuggestionCorrect"] for item in folds)
    template_correct = sum(item["templateCorrect"] for item in folds)
    topology_correct = sum(item["experimentalTopologyCorrect"] for item in folds)
    payload = {
        "protocol": {
            "prediction": "held-out annotation unopened",
            "templateEligibility": "exact component ID and exact stroke count",
            "selection": (
                "joint verified-template whole-glyph reconstruction; same-source "
                "leaf and cross-source divergence signals are review-only evidence"
            ),
            "automaticWriteEnabled": False,
            "evaluationStatus": (
                "exploratory; rules were iterated while inspecting this nine-case "
                "corpus and require a future frozen validation set"
            ),
        },
        "summary": {
            "total": len(folds),
            "correct": correct,
            "accuracy": round(correct / len(folds), 6) if folds else None,
            "exploratoryReviewSuggestionCorrect": review_correct,
            "exploratoryReviewSuggestionAccuracy": (
                round(review_correct / len(folds), 6) if folds else None
            ),
            "templateOnlyCorrect": template_correct,
            "experimentalTopologyOnlyCorrect": topology_correct,
        },
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
