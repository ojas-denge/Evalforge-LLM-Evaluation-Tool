from datetime import datetime, timezone

from app.evaluation.comparison import (
    EvaluationComparison,
    compare_runs,
)
from app.evaluation.regression import RegressionPolicy
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievalConfig,
)


def make_result(
    case_id: str,
    *,
    hit_at_1: float,
    mrr: float,
    latency: float,
    failure_type: str = "PASS",
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        question=f"Question {case_id}",
        expected_documents=["doc-1"],
        retrieved_evidence=[],
        failure_type=failure_type,
        relevant_documents_found=failure_type == "PASS",
        first_relevant_rank=1 if failure_type == "PASS" else None,
        missing_documents=[],
        confounding_documents=[],
        hit_at_1=hit_at_1,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_1=hit_at_1,
        recall_at_3=1.0,
        recall_at_5=1.0,
        mrr=mrr,
        retrieval_latency_ms=latency,
    )


def make_run(
    run_id: str,
    results: list[EvaluationResult],
) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        dataset_size=len(results),
        results=results,
        mean_hit_at_1=sum(r.hit_at_1 for r in results) / len(results),
        mean_hit_at_3=sum(r.hit_at_3 for r in results) / len(results),
        mean_hit_at_5=sum(r.hit_at_5 for r in results) / len(results),
        mean_recall_at_1=sum(r.recall_at_1 for r in results) / len(results),
        mean_recall_at_3=sum(r.recall_at_3 for r in results) / len(results),
        mean_recall_at_5=sum(r.recall_at_5 for r in results) / len(results),
        mean_mrr=sum(r.mrr for r in results) / len(results),
        mean_retrieval_latency_ms=(
            sum(r.retrieval_latency_ms for r in results) / len(results)
        ),
        created_at=datetime.now(timezone.utc),
        retrieval_config=RetrievalConfig(
            mode="dense",
            top_k=5,
            candidate_k=None,
            reranking_enabled=False,
            reranker_candidate_k=None,
            hybrid_retrieval_enabled=False,
        ),
    )


def test_regression_policy_passes_when_no_metric_degrades():
    baseline = make_run(
        "baseline",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=10.0,
            )
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=10.0,
            )
        ],
    )

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy()

    result = policy.evaluate(comparison)

    assert result.regression_detected is False
    assert result.reasons == []


def test_regression_policy_detects_mrr_drop():
    baseline = make_run(
        "baseline",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=10.0,
            )
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result(
                "case-1",
                hit_at_1=0.0,
                mrr=0.5,
                latency=10.0,
            )
        ],
    )

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_mrr_drop=0.1,
    )

    result = policy.evaluate(comparison)

    assert result.regression_detected is True
    assert "mean_mrr" in result.reasons


def test_regression_policy_detects_latency_increase():
    baseline = make_run(
        "baseline",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=100.0,
            )
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=150.0,
            )
        ],
    )

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_latency_increase_ms=20.0,
    )

    result = policy.evaluate(comparison)

    assert result.regression_detected is True
    assert "mean_retrieval_latency_ms" in result.reasons


def test_regression_policy_allows_changes_within_thresholds():
    baseline = make_run(
        "baseline",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=100.0,
            )
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=0.95,
                latency=110.0,
            )
        ],
    )

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_mrr_drop=0.1,
        max_latency_increase_ms=20.0,
    )

    result = policy.evaluate(comparison)

    assert result.regression_detected is False
    assert result.reasons == []

def test_regression_policy_can_disable_latency_gating():
    baseline = make_run(
        "baseline",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=100.0,
            )
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result(
                "case-1",
                hit_at_1=1.0,
                mrr=1.0,
                latency=500.0,
            )
        ],
    )

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_latency_increase_ms=None,
    )

    result = policy.evaluate(comparison)

    assert result.regression_detected is False
    assert result.reasons == []


def test_regression_policy_detects_correctness_drop():
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
    policy = RegressionPolicy(max_correctness_drop=0.1)
    result = policy.evaluate(comparison)
    assert result.regression_detected is True
    assert "mean_correctness" in result.reasons


def test_regression_policy_ignores_correctness_when_not_configured():
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
        mean_correctness_delta=-0.5,
    )
    policy = RegressionPolicy(max_correctness_drop=None)
    result = policy.evaluate(comparison)
    assert result.regression_detected is False


def test_regression_policy_detects_groundedness_drop():
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
        mean_groundedness_delta=-0.15,
    )
    policy = RegressionPolicy(max_groundedness_drop=0.1)
    result = policy.evaluate(comparison)
    assert result.regression_detected is True
    assert "mean_groundedness" in result.reasons


def test_regression_policy_ignores_none_correctness_delta():
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
        mean_correctness_delta=None,
    )
    policy = RegressionPolicy(max_correctness_drop=0.1)
    result = policy.evaluate(comparison)
    assert result.regression_detected is False


def test_regression_policy_detects_cost_increase():
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
        cost_delta_usd=0.5,
    )
    policy = RegressionPolicy(max_cost_increase_usd=0.1)
    result = policy.evaluate(comparison)
    assert result.regression_detected is True
    assert "cost_usd" in result.reasons


def test_regression_policy_detects_conflict_rate():
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
        conflict_rate_delta=0.2,
    )
    policy = RegressionPolicy(max_conflict_rate=0.1)
    result = policy.evaluate(comparison)
    assert result.regression_detected is True
    assert "conflict_rate" in result.reasons


def test_regression_policy_validates_new_thresholds():
    import pytest
    with pytest.raises(ValueError, match="max_correctness_drop must be non-negative"):
        RegressionPolicy(max_correctness_drop=-0.1)
    with pytest.raises(ValueError, match="max_groundedness_drop must be non-negative"):
        RegressionPolicy(max_groundedness_drop=-0.1)
    with pytest.raises(ValueError, match="max_cost_increase_usd must be non-negative"):
        RegressionPolicy(max_cost_increase_usd=-0.1)
    with pytest.raises(ValueError, match="max_conflict_rate must be non-negative"):
        RegressionPolicy(max_conflict_rate=-0.1)

