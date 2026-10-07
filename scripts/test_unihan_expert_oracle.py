import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from unihan_experts.oracle import rank_case_results, select_oracle_winner


def cell(
    expert,
    *,
    status="complete",
    complete=True,
    reasons=(),
    stroke=0.8,
    component=0.8,
    dtw=2.0,
    start=2.0,
    end=2.0,
):
    return {
        "expertId": expert,
        "caseId": "case",
        "status": status,
        "complete": complete,
        "reviewReasons": list(reasons),
        "evaluation": {
            "metrics": {
                "strokeMacroIoU": stroke,
                "componentMacroIoU": component,
                "meanDirectedSequenceDtwPercent": dtw,
                "meanStartErrorPercent": start,
                "meanEndErrorPercent": end,
            }
        },
    }


class OracleTests(unittest.TestCase):
    def test_truth_isolation_failure_is_ineligible(self):
        invalid = cell("invalid", status="truth-isolation-failure", stroke=1.0)
        valid = cell("valid", stroke=0.5)
        self.assertEqual(select_oracle_winner([invalid, valid])["expertId"], "valid")

    def test_complete_beats_high_scoring_partial(self):
        partial = cell("partial", status="partial", complete=False, stroke=1.0)
        complete = cell("complete", stroke=0.4)
        self.assertEqual(select_oracle_winner([partial, complete])["expertId"], "complete")

    def test_fewer_hard_reasons_beat_metric_before_iou(self):
        brittle = cell(
            "brittle", reasons=("hard-geometry-fallback",), stroke=0.99
        )
        stable = cell("stable", reasons=(), stroke=0.7)
        self.assertEqual(select_oracle_winner([brittle, stable])["expertId"], "stable")

    def test_soft_review_reason_count_does_not_outrank_truth_metric(self):
        weak = cell(
            "weak",
            reasons=("large-unexplained-skeleton",),
            stroke=0.057634,
        )
        strong = cell(
            "strong",
            reasons=("large-unexplained-skeleton", "same-leaf-contact-fallback"),
            stroke=0.648445,
        )
        self.assertEqual(select_oracle_winner([weak, strong])["expertId"], "strong")

    def test_metrics_follow_declared_priority(self):
        lower_stroke = cell("component", stroke=0.8, component=1.0)
        higher_stroke = cell("stroke", stroke=0.9, component=0.1)
        self.assertEqual(
            select_oracle_winner([lower_stroke, higher_stroke])["expertId"],
            "stroke",
        )

    def test_true_tie_is_deterministic_by_expert_id(self):
        ranked = rank_case_results([cell("zeta"), cell("alpha")])
        self.assertEqual([item["expertId"] for item in ranked], ["alpha", "zeta"])

    def test_oracle_marks_human_truth_usage(self):
        winner = select_oracle_winner([cell("one")])
        self.assertTrue(winner["oracle"]["usesHumanTruth"])
        self.assertEqual(winner["oracle"]["purpose"], "benchmark-only")


if __name__ == "__main__":
    unittest.main()
