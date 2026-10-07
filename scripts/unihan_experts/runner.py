"""Resumable two-phase execution of immutable historical experts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .adapters import (
    RunContext,
    build_evaluation_command,
    build_inference_command,
    semantic_command_hash,
)
from .hashing import sha256_file, sha256_json
from .models import CaseManifest, ExpertManifest
from .normalize import normalize_artifacts
from .process import CommandResult, run_command


@dataclass(frozen=True)
class CellSpec:
    expert: ExpertManifest
    case: CaseManifest
    expert_content_id: str
    input_hashes: Mapping[str, str]
    python_executable: Path
    interpreter_identity: Mapping[str, object]
    worktree: Path
    shared_artifacts: Mapping[str, Path]
    resolved_rule_artifacts: Mapping[str, Path]
    run_root: Path
    timeout_seconds: float


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _result_payload(result: CommandResult) -> dict[str, object]:
    return {
        "argv": list(result.argv),
        "exitCode": result.exit_code,
        "timedOut": result.timed_out,
        "wallSeconds": result.wall_seconds,
        "cpuSeconds": result.cpu_seconds,
        "peakMemoryBytes": result.peak_memory_bytes,
        "resourceUnavailableReason": result.resource_unavailable_reason,
    }


def _write_process_artifacts(directory: Path, result: CommandResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(result.stderr, encoding="utf-8")


def _phase_failure(
    spec: CellSpec,
    *,
    status: str,
    phase: str,
    reason: str,
    process: CommandResult | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schemaVersion": 1,
        "expertId": spec.expert.expert_id,
        "caseId": spec.case.case_id,
        "status": status,
        "complete": False,
        "failedPhase": phase,
        "reviewReasons": [reason],
        "usesHumanTruth": phase == "evaluation",
    }
    if process is not None:
        result["process"] = _result_payload(process)
    return result


def _cell_directory(spec: CellSpec) -> Path:
    return spec.run_root / spec.expert.expert_id / spec.case.case_id


def _run_phase(
    spec: CellSpec,
    *,
    directory: Path,
    command: Sequence[str],
) -> tuple[dict[str, object], CommandResult]:
    result = run_command(
        command,
        cwd=spec.worktree,
        timeout_seconds=spec.timeout_seconds,
    )
    _write_process_artifacts(directory, result)
    if result.timed_out:
        return (
            _phase_failure(
                spec,
                status="timeout",
                phase=directory.name,
                reason=f"expert exceeded {spec.timeout_seconds:g}s wall budget",
                process=result,
            ),
            result,
        )
    if result.exit_code != 0:
        return (
            _phase_failure(
                spec,
                status="process-error",
                phase=directory.name,
                reason=f"expert exited with status {result.exit_code}",
                process=result,
            ),
            result,
        )

    normalized = normalize_artifacts(
        directory / "raw-audit.json",
        directory / "review.html",
        expert_id=spec.expert.expert_id,
        case_id=spec.case.case_id,
    )
    normalized["process"] = _result_payload(result)
    return normalized, result


def _finalize_failure(cell_dir: Path, failure: dict[str, object]) -> dict[str, object]:
    _atomic_json(cell_dir / "normalized.json", failure)
    return failure


def run_cell(
    spec: CellSpec,
    *,
    annotation_resolver: Callable[[CaseManifest], Path],
    resume: bool = False,
) -> dict[str, object]:
    cell_dir = _cell_directory(spec)
    inference_dir = cell_dir / "inference"
    evaluation_dir = cell_dir / "evaluation"
    inference_context = RunContext(
        python_executable=spec.python_executable,
        worktree=spec.worktree,
        expert=spec.expert,
        case=spec.case,
        shared_artifacts=spec.shared_artifacts,
        resolved_rule_artifacts=spec.resolved_rule_artifacts,
        output_path=inference_dir / "review.html",
        audit_path=inference_dir / "raw-audit.json",
    )
    inference_command = build_inference_command(inference_context)
    semantic_hash = semantic_command_hash(inference_command)
    cache_key = sha256_json(
        {
            "expertContentId": spec.expert_content_id,
            "caseId": spec.case.case_id,
            "inputHashes": spec.input_hashes,
            "semanticCommandHash": semantic_hash,
            "interpreter": spec.interpreter_identity,
            "phase": "inference",
        }
    )
    manifest = {
        "schemaVersion": 1,
        "expertId": spec.expert.expert_id,
        "expertContentId": spec.expert_content_id,
        "caseId": spec.case.case_id,
        "inputHashes": dict(spec.input_hashes),
        "semanticCommandHash": semantic_hash,
        "interpreter": dict(spec.interpreter_identity),
    }
    _atomic_json(cell_dir / "manifest.json", manifest)

    prediction_path = inference_dir / "prediction.json"
    prediction = _load_json(prediction_path) if resume else None
    if prediction is None or prediction.get("cacheKey") != cache_key:
        prediction, _inference_process = _run_phase(
            spec, directory=inference_dir, command=inference_command
        )
        if prediction["status"] in {
            "timeout",
            "process-error",
            "missing-output",
            "malformed-audit",
        }:
            return _finalize_failure(cell_dir, prediction)
        if prediction.get("usesHumanTruth"):
            failure = _phase_failure(
                spec,
                status="truth-isolation-failure",
                phase="inference",
                reason="inference audit reports human truth access",
            )
            failure["inference"] = prediction
            return _finalize_failure(cell_dir, failure)
        prediction["cacheKey"] = cache_key
        prediction["semanticCommandHash"] = semantic_hash
        prediction["expertContentId"] = spec.expert_content_id
        prediction["inputHashes"] = dict(spec.input_hashes)
        _atomic_json(prediction_path, prediction)

    # This call is intentionally after the prediction's atomic persistence.
    annotation = annotation_resolver(spec.case).resolve()
    annotation_hash = sha256_file(annotation)
    evaluation_context = RunContext(
        python_executable=spec.python_executable,
        worktree=spec.worktree,
        expert=spec.expert,
        case=spec.case,
        shared_artifacts=spec.shared_artifacts,
        resolved_rule_artifacts=spec.resolved_rule_artifacts,
        output_path=evaluation_dir / "review.html",
        audit_path=evaluation_dir / "raw-audit.json",
    )
    evaluation_command = build_evaluation_command(evaluation_context, annotation)
    if semantic_command_hash(evaluation_command) != semantic_hash:
        failure = _phase_failure(
            spec,
            status="truth-isolation-failure",
            phase="evaluation",
            reason="evaluation semantic command differs from inference",
        )
        return _finalize_failure(cell_dir, failure)
    evaluation_cache_key = sha256_json(
        {
            "inferenceCacheKey": cache_key,
            "annotationHash": annotation_hash,
            "phase": "evaluation",
        }
    )
    evaluation_path = evaluation_dir / "evaluation.json"
    evaluation = _load_json(evaluation_path) if resume else None
    if evaluation is None or evaluation.get("cacheKey") != evaluation_cache_key:
        evaluation, _evaluation_process = _run_phase(
            spec, directory=evaluation_dir, command=evaluation_command
        )
        if evaluation["status"] in {
            "timeout",
            "process-error",
            "missing-output",
            "malformed-audit",
        }:
            failure = dict(evaluation)
            failure["inference"] = prediction
            return _finalize_failure(cell_dir, failure)
        if not evaluation.get("usesHumanTruth"):
            failure = _phase_failure(
                spec,
                status="truth-isolation-failure",
                phase="evaluation",
                reason="evaluation audit did not report truth usage",
            )
            failure["inference"] = prediction
            failure["evaluation"] = evaluation
            return _finalize_failure(cell_dir, failure)
        evaluation["cacheKey"] = evaluation_cache_key
        evaluation["annotationHash"] = annotation_hash
        evaluation["semanticCommandHash"] = semantic_hash
        _atomic_json(evaluation_path, evaluation)

    matching = (
        prediction.get("predictionFingerprint")
        == evaluation.get("predictionFingerprint")
    )
    status = (
        str(evaluation.get("status"))
        if matching
        else "truth-isolation-failure"
    )
    result = {
        "schemaVersion": 1,
        "expertId": spec.expert.expert_id,
        "caseId": spec.case.case_id,
        "status": status,
        "complete": bool(evaluation.get("complete")) and matching,
        "usesHumanTruth": True,
        "truthFirstReadPhase": "evaluation",
        "predictionFingerprintMatch": matching,
        "inference": prediction,
        "evaluation": evaluation,
    }
    if not matching:
        result["reviewReasons"] = [
            "evaluation prediction differs from truth-free inference"
        ]
    _atomic_json(cell_dir / "normalized.json", result)
    return result


def run_matrix(
    specs: Sequence[CellSpec],
    *,
    annotation_resolver: Callable[[CaseManifest], Path],
    resume: bool = False,
) -> list[dict[str, object]]:
    ordered = sorted(specs, key=lambda item: (item.expert.expert_id, item.case.case_id))
    return [
        run_cell(spec, annotation_resolver=annotation_resolver, resume=resume)
        for spec in ordered
    ]
