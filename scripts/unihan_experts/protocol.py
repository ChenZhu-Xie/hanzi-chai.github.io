"""Language-neutral consultation records for future expert composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .hashing import sha256_json


SUPPORTED_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ConsultationBudgets:
    max_depth: int
    wall_seconds: float
    cpu_seconds: float
    memory_bytes: int
    candidate_count: int

    def __post_init__(self) -> None:
        values = (
            self.max_depth,
            self.wall_seconds,
            self.cpu_seconds,
            self.memory_bytes,
            self.candidate_count,
        )
        if any(value <= 0 for value in values):
            raise ValueError("every consultation budget must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "maxDepth": self.max_depth,
            "wallSeconds": self.wall_seconds,
            "cpuSeconds": self.cpu_seconds,
            "memoryBytes": self.memory_bytes,
            "candidateCount": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ConsultationBudgets":
        return cls(
            max_depth=int(value["maxDepth"]),
            wall_seconds=float(value["wallSeconds"]),
            cpu_seconds=float(value["cpuSeconds"]),
            memory_bytes=int(value["memoryBytes"]),
            candidate_count=int(value["candidateCount"]),
        )


@dataclass(frozen=True)
class ProposalRequest:
    protocol_version: int
    caller_expert_id: str
    run_id: str
    input_hashes: Mapping[str, str]
    ids_subtree: Mapping[str, object] | None
    stroke_interval: tuple[int, int] | None
    snapshot_hashes: Mapping[str, str]
    required_capabilities: tuple[str, ...]
    budgets: ConsultationBudgets
    visited_experts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.protocol_version != SUPPORTED_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported proposal protocol: {self.protocol_version}"
            )
        if self.ids_subtree is None and self.stroke_interval is None:
            raise ValueError("proposal request requires an IDS or stroke boundary")
        if self.stroke_interval is not None:
            start, end = self.stroke_interval
            if start < 0 or end <= start:
                raise ValueError("stroke interval must be a non-empty forward interval")
        if not self.caller_expert_id or not self.run_id:
            raise ValueError("proposal caller and run ID are required")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocolVersion": self.protocol_version,
            "callerExpertId": self.caller_expert_id,
            "runId": self.run_id,
            "inputHashes": dict(self.input_hashes),
            "idsSubtree": dict(self.ids_subtree) if self.ids_subtree is not None else None,
            "strokeInterval": list(self.stroke_interval)
            if self.stroke_interval is not None
            else None,
            "snapshotHashes": dict(self.snapshot_hashes),
            "requiredCapabilities": list(self.required_capabilities),
            "budgets": self.budgets.to_dict(),
            "visitedExperts": list(self.visited_experts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProposalRequest":
        raw_interval = value.get("strokeInterval")
        interval = (
            (int(raw_interval[0]), int(raw_interval[1]))
            if isinstance(raw_interval, list)
            else None
        )
        raw_subtree = value.get("idsSubtree")
        return cls(
            protocol_version=int(value["protocolVersion"]),
            caller_expert_id=str(value["callerExpertId"]),
            run_id=str(value["runId"]),
            input_hashes=dict(value.get("inputHashes", {})),
            ids_subtree=dict(raw_subtree) if isinstance(raw_subtree, dict) else None,
            stroke_interval=interval,
            snapshot_hashes=dict(value.get("snapshotHashes", {})),
            required_capabilities=tuple(value.get("requiredCapabilities", [])),
            budgets=ConsultationBudgets.from_dict(value["budgets"]),
            visited_experts=tuple(value.get("visitedExperts", [])),
        )

    def content_hash(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ProposalBundle:
    protocol_version: int
    specialist_expert_id: str
    request_hash: str
    proposals: tuple[Mapping[str, object], ...]
    evidence: Mapping[str, object]
    assumptions: tuple[str, ...]
    resource_cost: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.protocol_version != SUPPORTED_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported proposal protocol: {self.protocol_version}"
            )
        if len(self.request_hash) != 64:
            raise ValueError("proposal bundle requires a request SHA-256 hash")
        if not self.specialist_expert_id:
            raise ValueError("specialist expert ID is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocolVersion": self.protocol_version,
            "specialistExpertId": self.specialist_expert_id,
            "requestHash": self.request_hash,
            "proposals": [dict(value) for value in self.proposals],
            "evidence": dict(self.evidence),
            "assumptions": list(self.assumptions),
            "resourceCost": dict(self.resource_cost),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProposalBundle":
        return cls(
            protocol_version=int(value["protocolVersion"]),
            specialist_expert_id=str(value["specialistExpertId"]),
            request_hash=str(value["requestHash"]),
            proposals=tuple(dict(item) for item in value.get("proposals", [])),
            evidence=dict(value.get("evidence", {})),
            assumptions=tuple(value.get("assumptions", [])),
            resource_cost=dict(value.get("resourceCost", {})),
            provenance=dict(value.get("provenance", {})),
        )

    def content_hash(self) -> str:
        return sha256_json(self.to_dict())


def validate_consultation(
    request: ProposalRequest, *, target_expert_id: str
) -> None:
    if target_expert_id in request.visited_experts:
        raise ValueError(f"consultation cycle targets {target_expert_id!r}")
    if len(request.visited_experts) >= request.budgets.max_depth:
        raise ValueError("consultation depth budget is exhausted")
