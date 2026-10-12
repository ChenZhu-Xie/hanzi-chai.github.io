"""Diagnosis of global low candidate fitness and inverse induction of allomorphic sibling components.

This module formalizes Universal Rule 8: The closed-loop residual evaluation
is not merely a passive score; when all existing candidates exhibit poor
coverage or fit, it acts as an active discovery engine for missing
allomorphic sibling components (同类但不同形的兄弟部件), the final soul puzzle
piece for assisted Hanzi decomposition.

Human-in-the-Loop & Bayesian Prior Principles:
1. Zero Silent Failures: Whenever global candidate fitness fails, the system
   unconditionally marks the glyph with `audit_status = "needs_human_audit"`
   in backend databases and frontend audit queues.
2. Bayesian Default Hypothesis: Under official Unicode glyph benchmarks and
   mature skeleton algorithms, image artifacts, orthography divergence, and
   IDS hierarchy errors are rare (<5%). The default primary hypothesis is
   that a single (or rarely, multiple) sibling component must be induced,
   pre-filling `recommended_action = "approve_sibling_component"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


class DiagnosisCause(str, Enum):
    """Root causes when all candidate models fail to achieve high fitness."""
    ALLOMORPHIC_SIBLING_MISSING = "allomorphic_sibling_missing"
    ORTHOGRAPHIC_VARIANT_MISMATCH = "orthographic_variant_mismatch"
    IDS_HIERARCHY_MALFORMATION = "ids_hierarchy_malformation"
    SKELETON_ARTIFACTS = "skeleton_artifacts"


class AuditStatus(str, Enum):
    """Audit queue flag for database schemas and frontend UI."""
    NEEDS_HUMAN_AUDIT = "needs_human_audit"
    AUTOMATED_PASS = "automated_pass"


class HumanInterventionAction(str, Enum):
    """Pre-filled human intervention actions in the audit interface."""
    APPROVE_SIBLING_COMPONENT = "approve_sibling_component"
    FLAG_SKELETON_ARTIFACT = "flag_skeleton_artifact"
    FLAG_ORTHOGRAPHIC_CONFLICT = "flag_orthographic_conflict"
    ADJUST_IDS_HIERARCHY = "adjust_ids_hierarchy"


@dataclass(frozen=True)
class DiagnosticResult:
    """Comprehensive diagnostic assessment with mandatory human audit flagging."""
    primary_cause: DiagnosisCause
    audit_status: AuditStatus
    recommended_action: HumanInterventionAction
    faulty_component_path: tuple[int, ...] | None
    summary: str
    is_sibling_missing_prior: bool = True

    def to_dict(self) -> dict:
        return {
            "primaryCause": self.primary_cause.value,
            "auditStatus": self.audit_status.value,
            "recommendedAction": self.recommended_action.value,
            "faultyComponentPath": (
                list(self.faulty_component_path) if self.faulty_component_path else None
            ),
            "summary": self.summary,
            "isSiblingMissingPrior": self.is_sibling_missing_prior,
        }


@dataclass(frozen=True)
class CandidateFitnessReport:
    """Fitness summary for an existing candidate decomposition."""
    candidate_id: int | str
    coverage_ratio: float
    mean_rmse: float
    component_coverages: dict[tuple[int, ...], float] = field(default_factory=dict)
    unclaimed_point_count: int = 0


@dataclass(frozen=True)
class SiblingComponentProposal:
    """A synthesized allomorphic sibling component recommendation for upstream."""
    base_component_id: int | str
    component_path: tuple[int, ...]
    mutation_type: str
    residual_point_count: int
    bounding_box: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    induced_strokes: tuple[dict, ...]
    expected_coverage_gain: float
    confidence: float
    audit_status: AuditStatus = AuditStatus.NEEDS_HUMAN_AUDIT
    recommended_action: HumanInterventionAction = HumanInterventionAction.APPROVE_SIBLING_COMPONENT

    def to_curation_packet(self) -> dict:
        """Export standardized proposal payload for upstream human-in-the-loop review."""
        return {
            "proposalType": "PROPOSE_NEW_SIBLING_COMPONENT",
            "auditStatus": self.audit_status.value,
            "recommendedAction": self.recommended_action.value,
            "baseComponentId": self.base_component_id,
            "componentPath": list(self.component_path),
            "mutationType": self.mutation_type,
            "residualPointCount": self.residual_point_count,
            "boundingBox": list(self.bounding_box),
            "inducedStrokes": list(self.induced_strokes),
            "expectedCoverageGain": round(self.expected_coverage_gain, 4),
            "confidence": round(self.confidence, 4),
            "availableHumanClassifications": [
                action.value for action in HumanInterventionAction
            ],
        }


def diagnose_global_candidate_failure(
    reports: list[CandidateFitnessReport],
    *,
    coverage_threshold: float = 0.75,
    rmse_threshold: float = 1.5,
    closed_component_threshold: float = 0.85,
) -> DiagnosticResult:
    """Diagnose why all candidate templates fail to explain the glyph skeleton.

    Parameters
    ----------
    reports : list[CandidateFitnessReport]
        Fitness scores of all candidate graphs tested against the skeleton.
    coverage_threshold : float
        Below this coverage, a candidate is considered deficient.
    rmse_threshold : float
        Above this error, a candidate is considered deficient.
    closed_component_threshold : float
        Components with coverage >= this threshold are deemed reliably closed.

    Returns
    -------
    DiagnosticResult
        Diagnostic result with mandatory `audit_status = NEEDS_HUMAN_AUDIT` and
        primary prior defaulting to allomorphic sibling generation.
    """
    if not reports:
        return DiagnosticResult(
            primary_cause=DiagnosisCause.IDS_HIERARCHY_MALFORMATION,
            audit_status=AuditStatus.NEEDS_HUMAN_AUDIT,
            recommended_action=HumanInterventionAction.ADJUST_IDS_HIERARCHY,
            faulty_component_path=None,
            summary="No candidate graphs available for diagnosis; flagged for manual review.",
            is_sibling_missing_prior=False,
        )

    # Check if ALL candidates exhibit low coverage or excessive error
    all_low_coverage = all(r.coverage_ratio < coverage_threshold for r in reports)
    all_high_rmse = all(r.mean_rmse > rmse_threshold for r in reports)

    if not (all_low_coverage or all_high_rmse):
        # Even if passing baseline, keep track of automated pass
        return DiagnosticResult(
            primary_cause=DiagnosisCause.ORTHOGRAPHIC_VARIANT_MISMATCH,
            audit_status=AuditStatus.AUTOMATED_PASS,
            recommended_action=HumanInterventionAction.APPROVE_SIBLING_COMPONENT,
            faulty_component_path=None,
            summary="At least one candidate achieves acceptable baseline fitness.",
            is_sibling_missing_prior=False,
        )

    # Inspect component-level closure consensus across candidates
    best_report = max(reports, key=lambda r: r.coverage_ratio)
    closed_paths = [
        path
        for path, cov in best_report.component_coverages.items()
        if cov >= closed_component_threshold
    ]
    unclosed_paths = [
        path
        for path, cov in best_report.component_coverages.items()
        if cov < closed_component_threshold
    ]

    # Under official Unicode glyphs and mature skeleton algorithms, the dominant
    # prior (>90%) is that a single (or rarely multiple) component's sibling variant
    # is missing from the repository.
    if closed_paths and len(unclosed_paths) in (1, 2) and best_report.unclaimed_point_count > 30:
        target_path = unclosed_paths[0]
        summary = (
            f"Localized failure: {len(closed_paths)} component(s) achieve closed fit "
            f"(e.g. {closed_paths}), but component {target_path} is rejected by template "
            f"mismatch, leaving {best_report.unclaimed_point_count} coherent residual points. "
            "Primary Bayesian prior: missing allomorphic sibling component for this position. "
            "Flagged for human verification with pre-filled sibling generation proposal."
        )
        return DiagnosticResult(
            primary_cause=DiagnosisCause.ALLOMORPHIC_SIBLING_MISSING,
            audit_status=AuditStatus.NEEDS_HUMAN_AUDIT,
            recommended_action=HumanInterventionAction.APPROVE_SIBLING_COMPONENT,
            faulty_component_path=target_path,
            summary=summary,
            is_sibling_missing_prior=True,
        )

    # If no components could be closed at all, macro IDS hierarchy might be flawed,
    # but still flagged for human review.
    if not closed_paths:
        return DiagnosticResult(
            primary_cause=DiagnosisCause.IDS_HIERARCHY_MALFORMATION,
            audit_status=AuditStatus.NEEDS_HUMAN_AUDIT,
            recommended_action=HumanInterventionAction.ADJUST_IDS_HIERARCHY,
            faulty_component_path=None,
            summary=(
                "Global macro-topological failure: no component could be closed. "
                "IDS tree hierarchy or decomposition operator is likely invalid. "
                "Mandatory human review queued."
            ),
            is_sibling_missing_prior=False,
        )

    # Fallback multi-component discrepancy
    return DiagnosticResult(
        primary_cause=DiagnosisCause.ALLOMORPHIC_SIBLING_MISSING,
        audit_status=AuditStatus.NEEDS_HUMAN_AUDIT,
        recommended_action=HumanInterventionAction.APPROVE_SIBLING_COMPONENT,
        faulty_component_path=unclosed_paths[0] if unclosed_paths else None,
        summary=(
            "Widespread discrepancy across components; defaulting to sibling component "
            "deficiency prior under official Unicode benchmark. Human audit required."
        ),
        is_sibling_missing_prior=True,
    )


def induce_sibling_component_from_residual(
    residual_pixels: np.ndarray,
    target_path: tuple[int, ...],
    base_component_id: int | str,
    *,
    total_skeleton_points: int,
    base_strokes_count: int | None = None,
) -> SiblingComponentProposal:
    """Synthesize an allomorphic sibling component proposal from residual skeleton pixels.

    Parameters
    ----------
    residual_pixels : np.ndarray
        Array of (x, y) coordinates of unexplained skeleton pixels.
    target_path : tuple[int, ...]
        Component path in the candidate tree.
    base_component_id : int | str
        Base component ID from the upstream repository.
    total_skeleton_points : int
        Total skeleton pixels of the character.
    base_strokes_count : int | None
        Number of strokes in the standard base component.
    """
    n_pts = len(residual_pixels)
    if n_pts == 0:
        raise ValueError("Cannot induce sibling component from empty residual skeleton.")

    min_x = float(np.min(residual_pixels[:, 0]))
    max_x = float(np.max(residual_pixels[:, 0]))
    min_y = float(np.min(residual_pixels[:, 1]))
    max_y = float(np.max(residual_pixels[:, 1]))
    bbox = (min_x, min_y, max_x, max_y)

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    aspect_ratio = width / height

    # Typological mutation diagnosis based on metric ratios
    if aspect_ratio > 1.3:
        mutation = "FLAT_BASE_LIGATURE_VARIANT"  # 扁平底座/连横变体 (如底座馬、底座魚)
    elif aspect_ratio < 0.7:
        mutation = "COMPRESSED_RADICAL_VARIANT"  # 狭长偏旁挤压变体 (如左旁馬、左旁月)
    else:
        mutation = "STRUCTURAL_STROKE_JUNCTION_VARIANT"  # 构型拓扑搭接变体

    coverage_gain = n_pts / max(total_skeleton_points, 1)

    # Produce induced stroke prototypes summarizing the bounding envelope
    induced_strokes = (
        {
            "role": "induced_corridor_cluster",
            "pointCount": n_pts,
            "bounds": [round(v, 2) for v in bbox],
            "aspectRatio": round(aspect_ratio, 2),
        },
    )

    confidence = min(0.99, 0.70 + 0.25 * min(1.0, n_pts / 100.0))

    return SiblingComponentProposal(
        base_component_id=base_component_id,
        component_path=target_path,
        mutation_type=mutation,
        residual_point_count=n_pts,
        bounding_box=bbox,
        induced_strokes=induced_strokes,
        expected_coverage_gain=coverage_gain,
        confidence=confidence,
        audit_status=AuditStatus.NEEDS_HUMAN_AUDIT,
        recommended_action=HumanInterventionAction.APPROVE_SIBLING_COMPONENT,
    )
