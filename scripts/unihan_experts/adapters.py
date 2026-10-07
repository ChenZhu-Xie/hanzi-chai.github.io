"""Explicit CLI adapters for immutable historical ink experts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .hashing import sha256_json
from .models import CaseManifest, ExpertManifest


COMMON_CONFIG = frozenset(
    {"canvas", "ordinalWeight", "coverageWeight", "beamWidth"}
)
ADAPTER_CONFIG = {
    "classic-v1": COMMON_CONFIG,
    "centroid-v2": COMMON_CONFIG | {"leafCentroidWeight"},
    "reservation-v3": COMMON_CONFIG
    | {"leafCentroidWeight", "completedStrokeReservation"},
    "head-v4": COMMON_CONFIG
    | {
        "leafCentroidWeight",
        "completedStrokeReservation",
        "expertFallbackRouteBudget",
        "allowPartial",
    },
}
RULE_FLAGS = {
    "learnedRules": "--learned-rules",
    "strokeOrderCatalog": "--stroke-order-catalog",
}


@dataclass(frozen=True)
class RunContext:
    python_executable: Path
    worktree: Path
    expert: ExpertManifest
    case: CaseManifest
    shared_artifacts: Mapping[str, Path]
    resolved_rule_artifacts: Mapping[str, Path]
    output_path: Path
    audit_path: Path


def _append_option(argv: list[str], flag: str, value: object) -> None:
    argv.extend((flag, str(value)))


def _required_artifact(context: RunContext, alias: str) -> Path:
    try:
        path = context.shared_artifacts[alias]
    except KeyError as error:
        raise ValueError(f"missing shared artifact {alias!r}") from error
    return path.resolve()


def _base_command(context: RunContext) -> list[str]:
    allowed = ADAPTER_CONFIG.get(context.expert.adapter)
    if allowed is None:
        raise ValueError(f"unsupported adapter: {context.expert.adapter}")
    unknown = set(context.expert.config) - allowed
    if unknown:
        raise ValueError(
            f"unsupported config for {context.expert.adapter}: {sorted(unknown)}"
        )

    script = (context.worktree / "scripts" / "unihan-ink-diffusion.py").resolve()
    if not script.is_file():
        raise ValueError(f"historical expert script is missing: {script}")

    argv = [str(context.python_executable.resolve()), str(script)]
    _append_option(argv, "--bbox-cache", _required_artifact(context, "bbox"))
    _append_option(argv, "--pdf", _required_artifact(context, "pdf"))
    _append_option(argv, "--candidates", _required_artifact(context, "candidates"))
    _append_option(argv, "--unicode", context.case.unicode)
    _append_option(argv, "--source", context.case.source)
    _append_option(argv, "--glyph-id", context.case.glyph_id)

    config_flags = (
        ("canvas", "--canvas"),
        ("ordinalWeight", "--ordinal-weight"),
        ("coverageWeight", "--coverage-weight"),
        ("leafCentroidWeight", "--leaf-centroid-weight"),
    )
    for key, flag in config_flags:
        if key in context.expert.config:
            _append_option(argv, flag, context.expert.config[key])
    _append_option(argv, "--decoder", context.expert.decoder)

    for artifact in context.expert.rule_artifacts:
        flag = RULE_FLAGS.get(artifact.alias)
        if flag is None:
            raise ValueError(f"unsupported rule artifact alias: {artifact.alias}")
        try:
            resolved = context.resolved_rule_artifacts[artifact.alias]
        except KeyError as error:
            raise ValueError(
                f"rule artifact {artifact.alias!r} was not resolved"
            ) from error
        _append_option(argv, flag, resolved.resolve())

    if "beamWidth" in context.expert.config:
        _append_option(argv, "--beam-width", context.expert.config["beamWidth"])
    if (
        context.expert.adapter in {"reservation-v3", "head-v4"}
        and context.expert.config.get("completedStrokeReservation") is False
    ):
        argv.append("--disable-completed-stroke-reservation")
    if context.expert.adapter == "head-v4":
        if "expertFallbackRouteBudget" in context.expert.config:
            _append_option(
                argv,
                "--expert-fallback-route-budget",
                context.expert.config["expertFallbackRouteBudget"],
            )
        if context.expert.config.get("allowPartial") is True:
            argv.append("--allow-partial")
    return argv


def build_inference_command(context: RunContext) -> tuple[str, ...]:
    argv = _base_command(context)
    _append_option(argv, "--output", context.output_path.resolve())
    _append_option(argv, "--audit-output", context.audit_path.resolve())
    return tuple(argv)


def build_evaluation_command(
    context: RunContext, annotation: Path
) -> tuple[str, ...]:
    annotation = annotation.resolve()
    if not annotation.is_file():
        raise ValueError(f"annotation is missing: {annotation}")
    argv = _base_command(context)
    _append_option(argv, "--annotations", annotation)
    _append_option(argv, "--output", context.output_path.resolve())
    _append_option(argv, "--audit-output", context.audit_path.resolve())
    return tuple(argv)


def semantic_command_hash(argv: Sequence[str]) -> str:
    excluded_with_value = {"--annotations", "--output", "--audit-output"}
    semantic: list[str] = []
    index = 0
    while index < len(argv):
        value = str(argv[index])
        if value in excluded_with_value:
            index += 2
            continue
        semantic.append(value)
        index += 1
    return sha256_json(semantic)
