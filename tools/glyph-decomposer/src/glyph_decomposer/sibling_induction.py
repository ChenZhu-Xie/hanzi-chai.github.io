"""Diagnosis of global low candidate fitness and inverse induction of allomorphic sibling components.

This module formalizes Universal Rule 8: The closed-loop residual evaluation
is not merely a passive score; when all existing candidates exhibit poor
coverage or fit, it acts as an active discovery engine for missing
allomorphic sibling components (同类但不同形的兄弟部件), the final soul puzzle
piece for assisted Hanzi decomposition.
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

    def to_curation_packet(self) -> dict:
        """Export standardized proposal payload for upstream human-in-the-loop review."""
        return {
            "proposalType": "PROPOSE_NEW_SIBLING_COMPONENT",
            "baseComponentId": self.base_component_id,
            "componentPath": list(self.component_path),
            "mutationType": self.mutation_type,
            "residualPointCount": self.residual_point_count,
            "boundingBox": list(self.bounding_box),
            "inducedStrokes": list(self.induced_strokes),
            "expectedCoverageGain": round(self.expected_coverage_gain, 4),
            "confidence": round(self.confidence, 4),
        }


def diagnose_global_candidate_failure(
    reports: list[CandidateFitnessReport],
    *,
    coverage_threshold: float = 0.75,
    rmse_threshold: float = 1.5,
    closed_component_threshold: float = 0.85,
) -> tuple[DiagnosisCause, tuple[int, ...] | None, str]:
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
    tuple[DiagnosisCause, tuple[int, ...] | None, str]
        (Primary cause, path of faulty unclosed component if localized, diagnostic summary)
    """
    if not reports:
        return (
            DiagnosisCause.IDS_HIERARCHY_MALFORMATION,
            None,
            "No candidate graphs available for diagnosis.",
        )

    # Check if ALL candidates exhibit low coverage or excessive error
    all_low_coverage = all(r.coverage_ratio < coverage_threshold for r in reports)
    all_high_rmse = all(r.mean_rmse > rmse_threshold for r in reports)

    if not (all_low_coverage or all_high_rmse):
        return (
            DiagnosisCause.ORTHOGRAPHIC_VARIANT_MISMATCH,
            None,
            "At least one candidate achieves acceptable baseline fitness.",
        )

    # Inspect component-level closure consensus across candidates
    # Find components that consistently achieve high closure in the best candidate
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

    # If some components are cleanly closed while one or two remain unclosed,
    # and there is a massive coherent pool of unclaimed skeleton points,
    # the failure is localized to a missing allomorphic sibling component.
    if closed_paths and len(unclosed_paths) in (1, 2) and best_report.unclaimed_point_count > 30:
        target_path = unclosed_paths[0]
        summary = (
            f"Localized failure: {len(closed_paths)} component(s) achieve closed fit "
            f"(e.g. {closed_paths}), but component {target_path} is rejected by template "
            f"mismatch, leaving {best_report.unclaimed_point_count} coherent residual points. "
            "A missing allomorphic sibling component is indicated."
        )
        return DiagnosisCause.ALLOMORPHIC_SIBLING_MISSING, target_path, summary

    if not closed_paths:
        return (
            DiagnosisCause.IDS_HIERARCHY_MALFORMATION,
            None,
            "Global macro-topological failure: no component could be closed. "
            "IDS tree hierarchy or decomposition operator is likely invalid.",
        )

    return (
        DiagnosisCause.ORTHOGRAPHIC_VARIANT_MISMATCH,
        None,
        "Widespread fine-grained discrepancy across multiple components.",
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
    )
