import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from unihan_experts.protocol import (
    ConsultationBudgets,
    ProposalBundle,
    ProposalRequest,
    validate_consultation,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def make_request(**overrides):
    values = {
        "protocol_version": 1,
        "caller_expert_id": "recursive-ids-8c",
        "run_id": "run-001",
        "input_hashes": {"pdf": SHA_A},
        "ids_subtree": {"operator": "⿰", "path": [1]},
        "stroke_interval": None,
        "snapshot_hashes": {"residual": SHA_B},
        "required_capabilities": ("grass-head",),
        "budgets": ConsultationBudgets(
            max_depth=2,
            wall_seconds=30.0,
            cpu_seconds=20.0,
            memory_bytes=256 * 1024 * 1024,
            candidate_count=32,
        ),
        "visited_experts": ("recursive-ids-8c",),
    }
    values.update(overrides)
    return ProposalRequest(**values)


class ProposalProtocolTests(unittest.TestCase):
    def test_request_round_trips_independently_of_python_objects(self):
        request = make_request()
        restored = ProposalRequest.from_dict(request.to_dict())
        self.assertEqual(restored, request)
        self.assertEqual(restored.content_hash(), request.content_hash())

    def test_bundle_round_trips_with_provenance(self):
        request = make_request()
        bundle = ProposalBundle(
            protocol_version=1,
            specialist_expert_id="grass-005",
            request_hash=request.content_hash(),
            proposals=({"routeIds": [1, 3]},),
            evidence={"coverageDelta": 0.12},
            assumptions=("IDS subtree is stable",),
            resource_cost={"wallSeconds": 1.2},
            provenance={"commit": "0058190"},
        )
        self.assertEqual(ProposalBundle.from_dict(bundle.to_dict()), bundle)

    def test_request_requires_component_or_stroke_boundary(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_request(ids_subtree=None, stroke_interval=None)

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_consultation(make_request(), target_expert_id="recursive-ids-8c")

    def test_depth_exhaustion_is_rejected(self):
        request = make_request(
            visited_experts=("recursive-ids-8c", "scan-front-7fc"),
            budgets=ConsultationBudgets(
                max_depth=2,
                wall_seconds=30,
                cpu_seconds=20,
                memory_bytes=1024,
                candidate_count=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "depth"):
            validate_consultation(request, target_expert_id="grass-005")

    def test_non_positive_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            ConsultationBudgets(
                max_depth=1,
                wall_seconds=0,
                cpu_seconds=1,
                memory_bytes=1,
                candidate_count=1,
            )

    def test_content_hash_changes_with_snapshot(self):
        left = make_request()
        right = make_request(snapshot_hashes={"residual": SHA_A})
        self.assertNotEqual(left.content_hash(), right.content_hash())


if __name__ == "__main__":
    unittest.main()
