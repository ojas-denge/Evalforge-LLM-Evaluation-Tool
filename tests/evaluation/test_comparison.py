from datetime import datetime, timezone

from app.evaluation.comparison import compare_runs
from app.models.evaluation import (
    AnswerJudgeResult,
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
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        question=f"Question {case_id}",
        expected_documents=["doc-1"],
        retrieved_evidence=[],
        failure_type="PASS",
        relevant_documents_found=True,
        first_relevant_rank=1,
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


def test_compare_runs_calculates_metric_deltas():
    baseline = make_run(
        "baseline",
        [
            make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0),
            make_result("case-2", hit_at_1=0.0, mrr=0.5, latency=20.0),
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=15.0),
            make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=25.0),
        ],
    )

    comparison = compare_runs(baseline, candidate)

    assert comparison.baseline_run_id == "baseline"
    assert comparison.candidate_run_id == "candidate"

    assert comparison.metric_deltas["mean_hit_at_1"] == 0.5
    assert comparison.metric_deltas["mean_mrr"] == 0.25
    assert comparison.metric_deltas["mean_retrieval_latency_ms"] == 5.0


def test_compare_runs_aligns_cases_by_case_id():
    baseline = make_run(
        "baseline",
        [
            make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0),
            make_result("case-2", hit_at_1=0.0, mrr=0.5, latency=20.0),
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=25.0),
            make_result("case-1", hit_at_1=0.0, mrr=0.5, latency=15.0),
        ],
    )

    comparison = compare_runs(baseline, candidate)

    assert len(comparison.case_changes) == 2

    changes = {
        change.case_id: change
        for change in comparison.case_changes
    }

    assert changes["case-1"].hit_at_1_delta == -1.0
    assert changes["case-2"].hit_at_1_delta == 1.0

    assert changes["case-1"].baseline_failure_type == "PASS"
    assert changes["case-1"].candidate_failure_type == "PASS"
    assert changes["case-2"].baseline_failure_type == "PASS"
    assert changes["case-2"].candidate_failure_type == "PASS"


def test_compare_runs_rejects_different_case_sets():
    baseline = make_run(
        "baseline",
        [
            make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0),
        ],
    )

    candidate = make_run(
        "candidate",
        [
            make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=10.0),
        ],
    )

    try:
        compare_runs(baseline, candidate)
    except ValueError as exc:
        assert "case IDs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for different case sets")


def test_compare_runs_captures_failure_type_transition():
    baseline_result = make_result(
        "case-1",
        hit_at_1=0.0,
        mrr=0.0,
        latency=20.0,
    )

    candidate_result = make_result(
        "case-1",
        hit_at_1=1.0,
        mrr=1.0,
        latency=25.0,
    )

    baseline_result = baseline_result.model_copy(
        update={"failure_type": "RANKING_FAILURE"}
    )

    candidate_result = candidate_result.model_copy(
        update={"failure_type": "PASS"}
    )

    baseline = make_run(
        "baseline",
        [baseline_result],
    )

    candidate = make_run(
        "candidate",
        [candidate_result],
    )

    comparison = compare_runs(
        baseline,
        candidate,
    )

    change = comparison.case_changes[0]

    assert change.case_id == "case-1"
    assert change.baseline_failure_type == "RANKING_FAILURE"
    assert change.candidate_failure_type == "PASS"
    assert change.hit_at_1_delta == 1.0
    assert change.mrr_delta == 1.0
    assert change.latency_delta_ms == 5.0


def test_compare_runs_computes_correctness_delta():
    baseline_result1 = make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )
    baseline_result2 = make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=False, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )
    candidate_result1 = baseline_result1.model_copy()
    candidate_result2 = make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )

    baseline = make_run("baseline", [baseline_result1, baseline_result2])
    candidate = make_run("candidate", [candidate_result1, candidate_result2])

    comparison = compare_runs(baseline, candidate)

    assert comparison.mean_correctness_delta == 0.5


def test_compare_runs_computes_groundedness_delta():
    baseline_result1 = make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )
    baseline_result2 = make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=False, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )
    candidate_result1 = baseline_result1.model_copy()
    candidate_result2 = make_result("case-2", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )

    baseline = make_run("baseline", [baseline_result1, baseline_result2])
    candidate = make_run("candidate", [candidate_result1, candidate_result2])

    comparison = compare_runs(baseline, candidate)

    assert comparison.mean_groundedness_delta == 0.5


def test_compare_runs_correctness_is_none_without_judge():
    baseline = make_run("baseline", [make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0)])
    candidate = make_run("candidate", [make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0)])

    comparison = compare_runs(baseline, candidate)

    assert comparison.mean_correctness_delta is None
    assert comparison.mean_groundedness_delta is None


def test_compare_runs_correctness_is_none_when_only_baseline_has_judge():
    baseline_result = make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0).model_copy(
        update={"answer_judge": AnswerJudgeResult(answer_correct=True, answer_grounded=True, topics_covered=[], topics_missing=[], unsupported_claims=[], reasoning="test reasoning")}
    )
    baseline = make_run("baseline", [baseline_result])
    candidate = make_run("candidate", [make_result("case-1", hit_at_1=1.0, mrr=1.0, latency=10.0)])

    comparison = compare_runs(baseline, candidate)

    assert comparison.mean_correctness_delta is None
    assert comparison.mean_groundedness_delta is None

