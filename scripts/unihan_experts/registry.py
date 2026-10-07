"""Load and validate immutable expert and benchmark case registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .models import ArtifactRef, CaseManifest, ExpertManifest


SUPPORTED_SCHEMA_VERSION = 1
ADAPTER_DECODERS = {
    "classic-v1": frozenset({"hybrid", "legacy", "residual"}),
    "centroid-v2": frozenset({"hybrid", "legacy", "residual"}),
    "reservation-v3": frozenset({"hybrid", "legacy", "residual"}),
    "head-v4": frozenset({"ensemble", "hybrid", "legacy", "residual"}),
}


def _read_registry(path: Path, collection_key: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {collection_key} registry schema: {payload.get('schemaVersion')!r}"
        )
    collection = payload.get(collection_key)
    if not isinstance(collection, list):
        raise ValueError(f"registry field {collection_key!r} must be a list")
    return collection


def load_expert_registry(path: Path) -> tuple[ExpertManifest, ...]:
    experts: list[ExpertManifest] = []
    for item in _read_registry(path, "experts"):
        rules = tuple(
            ArtifactRef(
                alias=str(rule["alias"]),
                path=str(rule["path"]),
                sha256=str(rule["sha256"]),
            )
            for rule in item.get("ruleArtifacts", [])
        )
        experts.append(
            ExpertManifest(
                expert_id=str(item["expertId"]),
                schema_version=int(item.get("schemaVersion", 1)),
                commit=str(item["commit"]),
                adapter=str(item["adapter"]),
                decoder=str(item["decoder"]),
                config=dict(item.get("config", {})),
                rule_artifacts=rules,
                parents=tuple(str(value) for value in item.get("parents", [])),
                declared_capabilities=tuple(
                    str(value) for value in item.get("declaredCapabilities", [])
                ),
                resource_class=str(item["resourceClass"]),
            )
        )
    validate_expert_registry(tuple(experts))
    return tuple(experts)


def validate_expert_registry(
    experts: tuple[ExpertManifest, ...],
    *,
    commit_resolver: Callable[[str], str] | None = None,
) -> None:
    by_id: dict[str, ExpertManifest] = {}
    for expert in experts:
        if expert.expert_id in by_id:
            raise ValueError(f"duplicate expert ID: {expert.expert_id}")
        by_id[expert.expert_id] = expert

        allowed = ADAPTER_DECODERS.get(expert.adapter)
        if allowed is None:
            raise ValueError(f"unsupported adapter: {expert.adapter}")
        if expert.decoder not in allowed:
            raise ValueError(
                f"decoder {expert.decoder!r} is unsupported by {expert.adapter}"
            )
        if commit_resolver is not None:
            try:
                resolved = commit_resolver(expert.commit)
            except Exception as error:
                raise ValueError(
                    f"cannot resolve commit {expert.commit!r} for {expert.expert_id}"
                ) from error
            if not resolved:
                raise ValueError(
                    f"cannot resolve commit {expert.commit!r} for {expert.expert_id}"
                )

    for expert in experts:
        for parent in expert.parents:
            if parent not in by_id:
                raise ValueError(
                    f"missing parent {parent!r} for expert {expert.expert_id!r}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(expert_id: str) -> None:
        if expert_id in visiting:
            raise ValueError(f"expert parent cycle includes {expert_id!r}")
        if expert_id in visited:
            return
        visiting.add(expert_id)
        for parent in by_id[expert_id].parents:
            visit(parent)
        visiting.remove(expert_id)
        visited.add(expert_id)

    for expert_id in by_id:
        visit(expert_id)


def load_case_registry(path: Path) -> tuple[CaseManifest, ...]:
    cases: list[CaseManifest] = []
    seen: set[str] = set()
    for item in _read_registry(path, "cases"):
        case = CaseManifest(
            case_id=str(item["caseId"]),
            unicode=str(item["unicode"]),
            source=str(item["source"]),
            glyph_id=int(item["glyphId"]),
            annotation_basename=str(item["annotationBasename"]),
        )
        if case.case_id in seen:
            raise ValueError(f"duplicate case ID: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def resolve_annotation(case: CaseManifest, roots: Iterable[Path]) -> Path:
    matches = [
        candidate.resolve()
        for root in roots
        if (candidate := root / case.annotation_basename).is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"annotation {case.annotation_basename!r} was not found in explicit roots"
        )
    unique = tuple(dict.fromkeys(matches))
    if len(unique) > 1:
        raise ValueError(
            f"annotation {case.annotation_basename!r} is ambiguous: "
            + ", ".join(str(path) for path in unique)
        )
    return unique[0]
