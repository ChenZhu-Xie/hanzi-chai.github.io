"""Leave-one-human-truth-out evaluation without reading the held-out answer.

The feature files may contain historical answer fields, but this module only
loads candidate measurements.  Each fold receives every other exported label.
``predict_fold`` has no held-out-label parameter; the held-out annotation is
opened only after its prediction has been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "templateToTargetMean",
    "templateToTargetP95",
    "targetToTemplateMean",
    "targetToTemplateP95",
    "meanLengthLogDistortion",
)
ANNOTATION_NAME = re.compile(
    r"U\+?([0-9A-F]+)-([A-Z]+)(?:-[0-9]+)?-annotations\.json$", re.IGNORECASE
)


@dataclass(frozen=True)
class FeatureCase:
    key: str
    candidate_ids: tuple[int, int]
    vector: np.ndarray
    global_scores: dict[int, float]
    focus_scores: dict[int, float]
    focus_metric_votes: dict[int, int]


def annotation_key(path: Path) -> str:
    match = ANNOTATION_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"cannot derive case key from {path.name}")
    return f"U+{match.group(1).upper()}-{match.group(2).upper()}.png"


def load_candidate_leaf_context(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text("utf-8"))
    return {
        int(row["unicode"]): {
            "candidateLeafIds": {
                int(glyph_id): {int(item) for item in leaf_ids}
                for glyph_id, leaf_ids in row["candidateLeafIds"].items()
            },
            "focusLeafIds": {int(item) for item in row.get("focusLeafIds", [])},
        }
        for row in payload["rows"]
    }


def normalize_component_ids(
    key: str, glyph_id: int, component_ids: set[int], contexts: dict[int, dict]
) -> tuple[list[int], list[dict]]:
    codepoint = int(key.split("+")[1].split("-")[0], 16)
    context = contexts.get(codepoint)
    if context is None or glyph_id not in context["candidateLeafIds"]:
        return sorted(component_ids), []
    allowed = context["candidateLeafIds"][glyph_id]
    focus = context["focusLeafIds"]
    replacement = allowed & focus
    normalized = set()
    corrections = []
    for component_id in component_ids:
        target = component_id
        if component_id not in allowed and component_id in focus and len(replacement) == 1:
            target = next(iter(replacement))
            corrections.append({"from": component_id, "to": target})
        normalized.add(target)
    return sorted(normalized), corrections


def load_truth(path: Path, contexts: dict[int, dict] | None = None) -> dict:
    document = json.loads(path.read_text("utf-8-sig"))
    metadata = document.get("metadata") or {}
    annotations = document.get("annotations") or []
    strokes = [
        item for item in annotations if item.get("type") not in {"lasso", "polygon"}
    ]
    regions = [
        item for item in annotations if item.get("type") in {"lasso", "polygon"}
    ]
    if not strokes or not regions:
        raise ValueError(f"{path} is not complete stroke-and-component truth")
    glyph_id = metadata.get("candidateGlyphId")
    if glyph_id is None:
        raise ValueError(f"{path} has no candidateGlyphId")
    key = annotation_key(path)
    raw_component_ids = {int(item["label"]) for item in regions}
    component_ids, corrections = normalize_component_ids(
        key, int(glyph_id), raw_component_ids, contexts or {}
    )
    return {
        "key": key,
        "glyphId": int(glyph_id),
        "strokeCount": len(strokes),
        "regionCount": len(regions),
        "componentIds": component_ids,
        "componentIdCorrections": corrections,
    }


def _measurement_rows(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Load measurements only; deliberately discard all answer-like fields."""
    payload = json.loads(path.read_text("utf-8"))
    output = {}
    for row in payload["results"]:
        output[row["key"]] = {
            int(glyph_id): {
                name: float(metrics[name]) for name in (*FEATURE_NAMES, "score")
            }
            for glyph_id, metrics in row["candidates"].items()
        }
    return output


def load_feature_cases(global_path: Path, focus_path: Path) -> dict[str, FeatureCase]:
    global_rows = _measurement_rows(global_path)
    focus_rows = _measurement_rows(focus_path)
    output = {}
    for key in sorted(global_rows.keys() & focus_rows.keys()):
        candidate_ids = tuple(sorted(global_rows[key]))
        if len(candidate_ids) != 2 or set(candidate_ids) != set(focus_rows[key]):
            continue
        low, high = candidate_ids
        vector = []
        for rows in (global_rows, focus_rows):
            vector.extend(rows[key][high][name] - rows[key][low][name] for name in FEATURE_NAMES)
        focus_votes = {
            candidate_id: sum(
                focus_rows[key][candidate_id][name]
                < focus_rows[key][other_id][name]
                for name in FEATURE_NAMES
            )
            for candidate_id, other_id in ((low, high), (high, low))
        }
        output[key] = FeatureCase(
            key=key,
            candidate_ids=(low, high),
            vector=np.asarray(vector, dtype=float),
            global_scores={item: global_rows[key][item]["score"] for item in candidate_ids},
            focus_scores={item: focus_rows[key][item]["score"] for item in candidate_ids},
            focus_metric_votes=focus_votes,
        )
    return output


def fit_pairwise_ridge(
    cases: dict[str, FeatureCase], labels: dict[str, int], regularization: float
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    targets = []
    for key, expected in labels.items():
        case = cases[key]
        if expected not in case.candidate_ids:
            raise ValueError(f"truth {expected} is not a candidate for {key}")
        vectors.append(case.vector)
        targets.append(1.0 if expected == case.candidate_ids[1] else -1.0)
    matrix = np.stack(vectors)
    target = np.asarray(targets)
    # RMS scaling is learned from the four training cases only.  No centering
    # and no intercept keeps the result invariant when candidate order swaps.
    scale = np.sqrt(np.mean(matrix * matrix, axis=0))
    scale[scale < 1e-9] = 1.0
    normalized = matrix / scale
    weights = np.linalg.solve(
        normalized.T @ normalized + regularization * np.eye(normalized.shape[1]),
        normalized.T @ target,
    )
    return weights, scale


def predict_fold(
    cases: dict[str, FeatureCase],
    training_labels: dict[str, int],
    heldout_key: str,
    *,
    regularization: float = 1.0,
) -> dict:
    """Predict one held-out case; its expected label is intentionally absent."""
    weights, scale = fit_pairwise_ridge(cases, training_labels, regularization)
    case = cases[heldout_key]
    decision = float((case.vector / scale) @ weights)
    predicted = case.candidate_ids[1] if decision > 0 else case.candidate_ids[0]
    global_prediction = min(case.global_scores, key=case.global_scores.get)
    focus_prediction = min(case.focus_scores, key=case.focus_scores.get)
    maximum_votes = max(case.focus_metric_votes.values())
    vote_winners = [
        glyph_id
        for glyph_id, votes in case.focus_metric_votes.items()
        if votes == maximum_votes
    ]
    focus_vote_prediction = vote_winners[0] if len(vote_winners) == 1 else None
    safe_prediction = (
        predicted if focus_vote_prediction is not None and predicted == focus_vote_prediction else None
    )
    return {
        "predictedGlyphId": predicted,
        "decision": round(decision, 6),
        "absoluteDecision": round(abs(decision), 6),
        "globalBaselineGlyphId": global_prediction,
        "globalBaselineMargin": round(
            abs(case.global_scores[case.candidate_ids[1]] - case.global_scores[case.candidate_ids[0]]),
            6,
        ),
        "focusBaselineGlyphId": focus_prediction,
        "focusMetricVoteGlyphId": focus_vote_prediction,
        "focusMetricVotes": case.focus_metric_votes,
        "safePredictionGlyphId": safe_prediction,
        "safeDecision": (
            "accepted: learned model agrees with the majority of raw focus metrics"
            if safe_prediction is not None
            else "abstained: learned model and raw focus metrics disagree"
        ),
        "weights": {
            f"{mode}.{name}": round(float(weight), 6)
            for mode, offset in (("global", 0), ("focus", len(FEATURE_NAMES)))
            for name, weight in zip(FEATURE_NAMES, weights[offset : offset + len(FEATURE_NAMES)])
        },
    }


def summarize(folds: list[dict], prediction_field: str) -> dict:
    attempted = [row for row in folds if row[prediction_field] is not None]
    correct = sum(row[prediction_field] == row["expectedGlyphId"] for row in attempted)
    return {
        "total": len(folds),
        "attempted": len(attempted),
        "correct": correct,
        "accuracy": round(correct / len(attempted), 6) if attempted else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-results", type=Path, required=True)
    parser.add_argument("--focus-results", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=1.0)
    args = parser.parse_args()

    paths = {annotation_key(path): path for path in args.annotations}
    if len(paths) != len(args.annotations):
        raise ValueError("annotation keys must be unique")
    cases = load_feature_cases(args.global_results, args.focus_results)
    contexts = load_candidate_leaf_context(args.candidates)
    missing = sorted(set(paths) - set(cases))
    if missing:
        raise ValueError(f"feature results are missing {missing}")

    folds = []
    truth_read_order = []
    for heldout_key in sorted(paths):
        training_truth = {}
        training_details = {}
        for key, path in paths.items():
            if key == heldout_key:
                continue
            truth = load_truth(path, contexts)
            truth_read_order.append({"fold": heldout_key, "phase": "train", "key": key})
            training_truth[key] = truth["glyphId"]
            training_details[key] = {
                item: truth[item]
                for item in (
                    "strokeCount",
                    "regionCount",
                    "componentIds",
                    "componentIdCorrections",
                )
            }
        prediction = predict_fold(
            cases,
            training_truth,
            heldout_key,
            regularization=args.regularization,
        )
        truth_read_order.append({"fold": heldout_key, "phase": "prediction-frozen", "key": heldout_key})
        heldout_truth = load_truth(paths[heldout_key], contexts)
        truth_read_order.append({"fold": heldout_key, "phase": "score", "key": heldout_key})
        folds.append(
            {
                "heldout": heldout_key,
                "trainingKeys": sorted(training_truth),
                "trainingTruth": training_details,
                **prediction,
                "expectedGlyphId": heldout_truth["glyphId"],
                "correct": prediction["predictedGlyphId"] == heldout_truth["glyphId"],
                "heldoutTruthReadAfterPrediction": True,
            }
        )
        print(
            f"{heldout_key}: predicted={prediction['predictedGlyphId']} "
            f"expected={heldout_truth['glyphId']} "
            f"{'OK' if folds[-1]['correct'] else 'WRONG'}",
            flush=True,
        )

    payload = {
        "protocol": {
            "name": "leave-one-human-truth-out",
            "trainingAnnotationsPerFold": len(paths) - 1,
            "heldoutAnnotationsPerFold": 1,
            "regularization": args.regularization,
            "model": "pairwise ridge, RMS-scaled, no centering, no intercept",
            "features": [
                f"{mode}.{name}" for mode in ("global", "focus") for name in FEATURE_NAMES
            ],
            "leakageGuards": [
                "feature loader discards expected/predicted/correct fields",
                "predict_fold has no held-out-label parameter",
                "held-out annotation is opened only after prediction is frozen",
                "no intercept and sorted pair differences prevent candidate-order majority shortcuts",
            ],
        },
        "summary": {
            "globalBaseline": summarize(folds, "globalBaselineGlyphId"),
            "focusBaseline": summarize(folds, "focusBaselineGlyphId"),
            "focusMetricVote": summarize(folds, "focusMetricVoteGlyphId"),
            "learned": summarize(folds, "predictedGlyphId"),
            "safeLearned": summarize(folds, "safePredictionGlyphId"),
        },
        "folds": folds,
        "truthReadAudit": truth_read_order,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
