from app.evaluation.triage import TriageDecision, triage_case, triage_dataset
from app.models.evaluation import EvaluationResult


def make_result(
    *,
    failure_type: str = "PASS",
    topic_coverage: float | None = None,
    case_id: str = "test-001",
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        question="Test question?",
        expected_documents=["doc-1"],
        retrieved_evidence=[],
        failure_type=failure_type,
        relevant_documents_found=failure_type != "RETRIEVAL_FAILURE",
        first_relevant_rank=1 if failure_type == "PASS" else None,
        missing_documents=[],
        confounding_documents=[],
        topic_coverage=topic_coverage,
        hit_at_1=1.0 if failure_type == "PASS" else 0.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_1=1.0 if failure_type == "PASS" else 0.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        mrr=1.0 if failure_type == "PASS" else 0.0,
        retrieval_latency_ms=10.0,
    )


def test_triage_retrieval_failure_is_deterministic_fail():
    result = make_result(failure_type="RETRIEVAL_FAILURE")
    assert triage_case(result) == TriageDecision.DETERMINISTIC_FAIL


def test_triage_pass_with_high_coverage_is_deterministic_pass():
    result = make_result(failure_type="PASS", topic_coverage=0.95)
    assert triage_case(result) == TriageDecision.DETERMINISTIC_PASS


def test_triage_pass_with_low_coverage_needs_judge():
    result = make_result(failure_type="PASS", topic_coverage=0.5)
    assert triage_case(result) == TriageDecision.NEEDS_JUDGE


def test_triage_pass_without_coverage_needs_judge():
    result = make_result(failure_type="PASS", topic_coverage=None)
    assert triage_case(result) == TriageDecision.NEEDS_JUDGE


def test_triage_ranking_failure_needs_judge():
    result = make_result(failure_type="RANKING_FAILURE")
    assert triage_case(result) == TriageDecision.NEEDS_JUDGE


def test_triage_partial_recall_needs_judge():
    result = make_result(failure_type="PARTIAL_RECALL")
    assert triage_case(result) == TriageDecision.NEEDS_JUDGE


def test_triage_custom_threshold():
    result = make_result(failure_type="PASS", topic_coverage=0.85)
    # Default threshold 0.9 → NEEDS_JUDGE
    assert triage_case(result) == TriageDecision.NEEDS_JUDGE
    # Lower threshold 0.8 → DETERMINISTIC_PASS
    assert triage_case(result, topic_coverage_pass_threshold=0.8) == TriageDecision.DETERMINISTIC_PASS


def test_triage_exact_threshold_is_pass():
    result = make_result(failure_type="PASS", topic_coverage=0.9)
    assert triage_case(result) == TriageDecision.DETERMINISTIC_PASS


def test_triage_dataset_partitions_correctly():
    results = [
        make_result(case_id="c1", failure_type="RETRIEVAL_FAILURE"),
        make_result(case_id="c2", failure_type="PASS", topic_coverage=1.0),
        make_result(case_id="c3", failure_type="RANKING_FAILURE"),
        make_result(case_id="c4", failure_type="PASS", topic_coverage=0.5),
    ]
    partitioned = triage_dataset(results)

    assert len(partitioned[TriageDecision.DETERMINISTIC_FAIL]) == 1
    assert partitioned[TriageDecision.DETERMINISTIC_FAIL][0].case_id == "c1"

    assert len(partitioned[TriageDecision.DETERMINISTIC_PASS]) == 1
    assert partitioned[TriageDecision.DETERMINISTIC_PASS][0].case_id == "c2"

    assert len(partitioned[TriageDecision.NEEDS_JUDGE]) == 2
    judge_ids = [r.case_id for r in partitioned[TriageDecision.NEEDS_JUDGE]]
    assert "c3" in judge_ids
    assert "c4" in judge_ids
