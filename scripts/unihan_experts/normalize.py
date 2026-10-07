"""Conservative normalization for historical ink expert audit schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .hashing import sha256_file, sha256_json


METRIC_KEYS = (
    "strokeMacroIoU",
    "componentMacroIoU",
    "meanDirectedSequenceDtwPercent",
    "meanStartErrorPercent",
    "meanEndErrorPercent",
    "meanCenterlineChamferPercent",
    "meanStartTangentErrorDegrees",
)


def _failure(
    status: str, *, expert_id: str, case_id: str, reason: str
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "expertId": expert_id,
        "caseId": case_id,
        "status": status,
        "complete": False,
        "predictedStrokeCount": None,
        "expectedStrokeCount": None,
        "reviewReasons": [reason],
        "metrics": {key: None for key in METRIC_KEYS},
        "usesHumanTruth": False,
        "truthFirstReadPhase": None,
        "predictionFingerprint": None,
        "nativeEvidence": {},
    }


def prediction_payload(audit: Mapping[str, object]) -> dict[str, object]:
    """Return stable prediction-only evidence, excluding truth and output paths."""

    decision = audit.get("decision")
    stable_decision = dict(decision) if isinstance(decision, Mapping) else {}
    return {
        "unicode": audit.get("unicode"),
        "character": audit.get("character"),
        "source": audit.get("source"),
        "glyphId": audit.get("glyphId"),
        "decision": stable_decision,
        "strokes": audit.get("strokes"),
        "residualReview": audit.get("residualReview"),
    }


def prediction_fingerprint(audit: Mapping[str, object]) -> str:
    return sha256_json(prediction_payload(audit))


def _expected_stroke_count(audit: Mapping[str, object]) -> int | None:
    strokes = audit.get("strokes")
    if isinstance(strokes, list):
        return len(strokes)
    evaluation = audit.get("evaluation")
    if isinstance(evaluation, Mapping):
        per_stroke = evaluation.get("perStrokeIoU")
        if isinstance(per_stroke, list):
            return len(per_stroke)
    return None


def _predicted_stroke_count(audit: Mapping[str, object]) -> int | None:
    decision = audit.get("decision")
    if not isinstance(decision, Mapping):
        return None
    steps = decision.get("steps")
    return len(steps) if isinstance(steps, list) else None


def normalize_audit(
    audit: object, *, expert_id: str, case_id: str
) -> dict[str, object]:
    if not isinstance(audit, Mapping):
        return _failure(
            "malformed-audit",
            expert_id=expert_id,
            case_id=case_id,
            reason="audit root must be an object",
        )
    decision = audit.get("decision")
    if not isinstance(decision, Mapping):
        return _failure(
            "malformed-audit",
            expert_id=expert_id,
            case_id=case_id,
            reason="audit decision must be an object",
        )

    evaluation = audit.get("evaluation")
    evaluated = isinstance(evaluation, Mapping)
    metrics = {
        key: evaluation.get(key) if evaluated else None for key in METRIC_KEYS
    }
    predicted = _predicted_stroke_count(audit)
    expected = _expected_stroke_count(audit)
    partial = bool(decision.get("partialPrediction")) or bool(
        decision.get("failedStroke")
    )
    complete = (
        not partial
        and predicted is not None
        and (expected is None or predicted == expected)
    )

    native_status = str(decision.get("status") or "needs-review")
    if partial or (expected is not None and predicted != expected):
        status = "partial"
        complete = False
    elif native_status == "safe-candidate" and complete:
        status = "complete"
    else:
        status = "needs-review"

    reasons = decision.get("reviewReasons")
    review_reasons = list(reasons) if isinstance(reasons, list) else []
    native_evidence = {
        "decision": dict(decision),
        "residualReview": audit.get("residualReview"),
        "strokes": audit.get("strokes"),
    }
    return {
        "schemaVersion": 1,
        "expertId": expert_id,
        "caseId": case_id,
        "status": status,
        "complete": complete,
        "predictedStrokeCount": predicted,
        "expectedStrokeCount": expected,
        "reviewReasons": review_reasons,
        "metrics": metrics,
        "usesHumanTruth": evaluated,
        "truthFirstReadPhase": "evaluation" if evaluated else None,
        "predictionFingerprint": prediction_fingerprint(audit),
        "routeCount": decision.get("routeHypothesisCount"),
        "unexplainedInkRatio": decision.get("unexplainedSkeletonRatio"),
        "score": decision.get("score"),
        "nativeEvidence": native_evidence,
    }


def normalize_artifacts(
    audit_path: Path,
    html_path: Path,
    *,
    expert_id: str,
    case_id: str,
) -> dict[str, object]:
    missing = [
        label
        for label, path in (("audit", audit_path), ("HTML", html_path))
        if not path.is_file()
    ]
    if missing:
        return _failure(
            "missing-output",
            expert_id=expert_id,
            case_id=case_id,
            reason=f"missing historical {' and '.join(missing)} output",
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result = _failure(
            "malformed-audit",
            expert_id=expert_id,
            case_id=case_id,
            reason=f"cannot parse historical audit: {error}",
        )
    else:
        result = normalize_audit(audit, expert_id=expert_id, case_id=case_id)

    result["rawArtifacts"] = {
        "auditPath": str(audit_path.resolve()),
        "auditSha256": sha256_file(audit_path),
        "htmlPath": str(html_path.resolve()),
        "htmlSha256": sha256_file(html_path),
    }
    return result
