import numpy as np
import pytest

from glyph_decomposer.sibling_induction import (
    CandidateFitnessReport,
    DiagnosisCause,
    diagnose_global_candidate_failure,
    induce_sibling_component_from_residual,
)


def test_diagnose_global_failure_identifies_missing_sibling():
    # Simulate a scenario where all candidates have low global coverage (<70%)
    # but components (0,) [月], (1,0) [龹] are cleanly closed, while (1,1) [馬] fails.
    reports = [
        CandidateFitnessReport(
            candidate_id="cand_1",
            coverage_ratio=0.62,
            mean_rmse=0.68,
            component_coverages={
                (0,): 0.98,
                (1, 0): 0.95,
                (1, 1): 0.35,  # 馬 template fails
            },
            unclaimed_point_count=550,
        ),
        CandidateFitnessReport(
            candidate_id="cand_2",
            coverage_ratio=0.58,
            mean_rmse=0.74,
            component_coverages={
                (0,): 0.96,
                (1, 0): 0.92,
                (1, 1): 0.28,
            },
            unclaimed_point_count=610,
        ),
    ]

    cause, faulty_path, summary = diagnose_global_candidate_failure(reports)

    assert cause == DiagnosisCause.ALLOMORPHIC_SIBLING_MISSING
    assert faulty_path == (1, 1)
    assert "coherent residual points" in summary


def test_diagnose_global_failure_identifies_ids_malformation_when_no_closure():
    reports = [
        CandidateFitnessReport(
            candidate_id="cand_bad",
            coverage_ratio=0.30,
            mean_rmse=2.5,
            component_coverages={
                (0,): 0.20,
                (1,): 0.35,
            },
            unclaimed_point_count=1200,
        )
    ]

    cause, faulty_path, summary = diagnose_global_candidate_failure(reports)

    assert cause == DiagnosisCause.IDS_HIERARCHY_MALFORMATION
    assert faulty_path is None
    assert "IDS tree hierarchy" in summary


def test_induce_sibling_component_from_residual_computes_flat_base_variant():
    # 550 residual pixels simulating the flat base 馬 in U+9A30
    xs = np.linspace(47.0, 80.0, 50)
    ys = np.linspace(47.0, 70.0, 11)
    grid_x, grid_y = np.meshgrid(xs, ys)
    residual_pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    proposal = induce_sibling_component_from_residual(
        residual_pixels,
        target_path=(1, 1),
        base_component_id=110,
        total_skeleton_points=1714,
    )

    assert proposal.base_component_id == 110
    assert proposal.component_path == (1, 1)
    assert proposal.mutation_type == "FLAT_BASE_LIGATURE_VARIANT"
    assert proposal.residual_point_count == len(residual_pixels)
    assert proposal.expected_coverage_gain == pytest.approx(len(residual_pixels) / 1714.0)
    assert proposal.confidence > 0.90

    packet = proposal.to_curation_packet()
    assert packet["proposalType"] == "PROPOSE_NEW_SIBLING_COMPONENT"
    assert packet["mutationType"] == "FLAT_BASE_LIGATURE_VARIANT"
