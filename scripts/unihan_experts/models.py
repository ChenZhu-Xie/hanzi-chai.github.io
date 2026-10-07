"""Versioned immutable records shared by expert registry and runner layers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath
from typing import Mapping

from .hashing import sha256_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_EXPERT_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CASE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_UNICODE = re.compile(r"^U\+[0-9A-F]{4,6}$")


def _require_sha256(value: str, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_git_object_id(value: str, label: str) -> None:
    if not _GIT_OBJECT_ID.fullmatch(value):
        raise ValueError(f"{label} must be a full lowercase Git object ID")


@dataclass(frozen=True)
class ArtifactRef:
    alias: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.alias:
            raise ValueError("artifact alias is required")
        if not self.path:
            raise ValueError("artifact path is required")
        _require_sha256(self.sha256, f"artifact {self.alias} hash")


@dataclass(frozen=True)
class ExpertManifest:
    expert_id: str
    schema_version: int
    commit: str
    adapter: str
    decoder: str
    config: Mapping[str, object]
    rule_artifacts: tuple[ArtifactRef, ...]
    parents: tuple[str, ...]
    declared_capabilities: tuple[str, ...]
    resource_class: str

    def __post_init__(self) -> None:
        if not _EXPERT_ID.fullmatch(self.expert_id):
            raise ValueError(f"invalid expert ID: {self.expert_id!r}")
        if self.schema_version < 1:
            raise ValueError("expert schema version must be positive")
        if not self.commit:
            raise ValueError("expert commit is required")
        if not self.adapter or not self.decoder:
            raise ValueError("expert adapter and decoder are required")
        if self.expert_id in self.parents:
            raise ValueError("expert cannot be its own parent")
        if not self.resource_class:
            raise ValueError("expert resource class is required")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "expertId": self.expert_id,
            "schemaVersion": self.schema_version,
            "commit": self.commit,
            "adapter": self.adapter,
            "decoder": self.decoder,
            "config": self.config,
            "ruleArtifacts": self.rule_artifacts,
            "parents": self.parents,
            "declaredCapabilities": self.declared_capabilities,
            "resourceClass": self.resource_class,
        }


@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    unicode: str
    source: str
    glyph_id: int
    annotation_basename: str

    def __post_init__(self) -> None:
        if not _CASE_ID.fullmatch(self.case_id):
            raise ValueError(f"invalid case ID: {self.case_id!r}")
        if not _UNICODE.fullmatch(self.unicode):
            raise ValueError(f"invalid Unicode label: {self.unicode!r}")
        if not self.source or not self.source.isascii():
            raise ValueError("source must be a non-empty ASCII label")
        if self.glyph_id < 0:
            raise ValueError("glyph ID must be non-negative")
        windows = PureWindowsPath(self.annotation_basename)
        native = Path(self.annotation_basename)
        if (
            windows.is_absolute()
            or native.is_absolute()
            or PurePath(self.annotation_basename).name != self.annotation_basename
            or windows.name != self.annotation_basename
        ):
            raise ValueError("annotation must be stored as a basename, not a path")


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int
    expert_id: str
    expert_content_id: str
    case_id: str
    input_hashes: Mapping[str, str]
    semantic_command_hash: str
    status: str
    prediction: Mapping[str, object]
    uses_human_truth: bool = False
    truth_first_read_phase: str | None = None
    review_reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("prediction schema version must be positive")
        _require_sha256(self.expert_content_id, "expert content ID")
        _require_sha256(self.semantic_command_hash, "semantic command hash")
        for alias, digest in self.input_hashes.items():
            _require_sha256(digest, f"input {alias} hash")
        if self.uses_human_truth or self.truth_first_read_phase is not None:
            raise ValueError("prediction inference records cannot read human truth")


@dataclass(frozen=True)
class EvaluationRecord:
    schema_version: int
    expert_id: str
    case_id: str
    prediction_fingerprint: str
    metrics: Mapping[str, object]
    uses_human_truth: bool = True
    truth_first_read_phase: str = "evaluation"

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("evaluation schema version must be positive")
        if not self.prediction_fingerprint:
            raise ValueError("evaluation requires a prediction fingerprint")
        _require_sha256(self.prediction_fingerprint, "prediction fingerprint")
        if not self.uses_human_truth or self.truth_first_read_phase != "evaluation":
            raise ValueError("evaluation truth may first be read only during evaluation")


def expert_content_identity(
    manifest: ExpertManifest,
    *,
    resolved_commit: str,
    source_tree_hash: str,
    interpreter_identity: Mapping[str, object],
) -> str:
    """Hash every semantic input that makes an expert independently runnable."""

    _require_git_object_id(resolved_commit, "resolved commit")
    _require_sha256(source_tree_hash, "source tree hash")
    return sha256_json(
        {
            "manifest": manifest.semantic_payload(),
            "resolvedCommit": resolved_commit,
            "sourceTreeHash": source_tree_hash,
            "interpreter": interpreter_identity,
        }
    )
