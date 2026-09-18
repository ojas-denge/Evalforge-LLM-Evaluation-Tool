from app.evaluation.ci_gate import EvaluationCIGate
from app.evaluation.comparison import EvaluationComparison
from app.evaluation.regression import RegressionPolicy


def make_comparison(
    *,
    mrr_delta: float,
    latency_delta: float,
) -> EvaluationComparison:
    return EvaluationComparison(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        metric_deltas={
            "mean_hit_at_1": 0.0,
            "mean_hit_at_3": 0.0,
            "mean_hit_at_5": 0.0,
            "mean_recall_at_1": 0.0,
            "mean_recall_at_3": 0.0,
            "mean_recall_at_5": 0.0,
            "mean_mrr": mrr_delta,
            "mean_retrieval_latency_ms": latency_delta,
        },
        case_changes=[],
    )


def make_gate() -> EvaluationCIGate:
    return EvaluationCIGate(
        RegressionPolicy(
            max_mrr_drop=0.02,
            max_latency_increase_ms=100.0,
        )
    )


def test_ci_gate_passes_when_no_regression():
    comparison = make_comparison(
        mrr_delta=0.01,
        latency_delta=50.0,
    )

    result = make_gate().evaluate(comparison)

    assert result.passed is True
    assert result.reasons == []


def test_ci_gate_fails_on_mrr_regression():
    comparison = make_comparison(
        mrr_delta=-0.05,
        latency_delta=50.0,
    )

    result = make_gate().evaluate(comparison)

    assert result.passed is False
    assert "mean_mrr" in result.reasons


def test_ci_gate_fails_on_latency_regression():
    comparison = make_comparison(
        mrr_delta=0.05,
        latency_delta=150.0,
    )

    result = make_gate().evaluate(comparison)

    assert result.passed is False
    assert "mean_retrieval_latency_ms" in result.reasons


def test_ci_gate_fails_on_correctness_regression():
    comparison = EvaluationComparison(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        metric_deltas={
            "mean_hit_at_1": 0.0,
            "mean_hit_at_3": 0.0,
            "mean_hit_at_5": 0.0,
            "mean_recall_at_1": 0.0,
            "mean_recall_at_3": 0.0,
            "mean_recall_at_5": 0.0,
            "mean_mrr": 0.0,
            "mean_retrieval_latency_ms": 0.0,
        },
        case_changes=[],
        mean_correctness_delta=-0.15,
    )
    gate = EvaluationCIGate(
        RegressionPolicy(
            max_mrr_drop=0.02,
            max_latency_increase_ms=100.0,
            max_correctness_drop=0.1,
        )
    )
    result = gate.evaluate(comparison)
    assert result.passed is False
    assert "mean_correctness" in result.reasons
