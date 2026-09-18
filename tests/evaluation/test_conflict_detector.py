from app.models.evaluation import AnswerJudgeResult, EvaluationResult
from app.evaluation.conflict_detector import ConflictDetector
from app.evaluation.judge_result import JudgeStatus, JudgeVerdict
from app.evaluation.trust import ConflictType
from app.generation.usage import GenerationUsage


def make_result(
    *,
    failure_type: str = "PASS",
    topic_coverage: float | None = None,
    answer_judge: AnswerJudgeResult | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        case_id="test-001",
        question="Test question?",
        expected_documents=["doc-1"],
        generated_answer="Answer",
        retrieved_evidence=[],
        failure_type=failure_type,
        relevant_documents_found=failure_type != "RETRIEVAL_FAILURE",
        first_relevant_rank=1 if failure_type == "PASS" else None,
        missing_documents=[],
        confounding_documents=[],
        topic_coverage=topic_coverage,
        answer_judge=answer_judge,
        hit_at_1=1.0 if failure_type == "PASS" else 0.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_1=1.0 if failure_type == "PASS" else 0.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        mrr=1.0 if failure_type == "PASS" else 0.0,
        retrieval_latency_ms=10.0,
    )


def make_judge(
    *,
    answer_correct: bool = True,
    answer_grounded: bool = True,
    topics_covered: list[str] | None = None,
    topics_missing: list[str] | None = None,
    unsupported_claims: list[str] | None = None,
) -> AnswerJudgeResult:
    return AnswerJudgeResult(
        answer_correct=answer_correct,
        answer_grounded=answer_grounded,
        topics_covered=topics_covered or [],
        topics_missing=topics_missing or [],
        unsupported_claims=unsupported_claims or [],
        reasoning="Test judgment.",
    )


def test_no_conflicts_when_all_signals_agree_pass():
    judge = make_judge(answer_correct=True, answer_grounded=True)
    result = make_result(
        failure_type="PASS",
        topic_coverage=1.0,
        answer_judge=judge,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.final_verdict == "PASS"
    assert assessment.confidence == "HIGH"
    assert assessment.conflicts == []


def test_no_conflicts_when_all_signals_agree_fail():
    judge = make_judge(answer_correct=False, answer_grounded=True)
    result = make_result(failure_type="PASS", answer_judge=judge)

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.final_verdict == "FAIL"
    assert assessment.confidence == "HIGH"
    assert assessment.conflicts == []


def test_detects_judge_contradicts_retrieval():
    judge = make_judge(answer_correct=True)
    result = make_result(
        failure_type="RETRIEVAL_FAILURE",
        answer_judge=judge,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert ConflictType.JUDGE_CONTRADICTS_RETRIEVAL in assessment.conflicts
    assert assessment.final_verdict == "REVIEW"


def test_detects_judge_contradicts_coverage():
    judge = make_judge(answer_correct=True)
    result = make_result(topic_coverage=0.1, answer_judge=judge)

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert ConflictType.JUDGE_CONTRADICTS_COVERAGE in assessment.conflicts
    assert assessment.final_verdict == "REVIEW"


def test_detects_judge_internal_inconsistency():
    judge = make_judge(
        answer_grounded=True,
        unsupported_claims=["some claim"],
    )
    result = make_result(answer_judge=judge)

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert ConflictType.JUDGE_INTERNAL_INCONSISTENCY in assessment.conflicts
    assert assessment.final_verdict == "REVIEW"


def test_detects_multiple_conflicts():
    judge = make_judge(
        answer_correct=True,
        answer_grounded=True,
        unsupported_claims=["claim"],
    )
    result = make_result(
        failure_type="RETRIEVAL_FAILURE",
        answer_judge=judge,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert ConflictType.JUDGE_CONTRADICTS_RETRIEVAL in assessment.conflicts
    assert ConflictType.JUDGE_INTERNAL_INCONSISTENCY in assessment.conflicts
    assert assessment.final_verdict == "REVIEW"


def test_no_judge_with_retrieval_pass():
    result = make_result(
        failure_type="PASS",
        answer_judge=None,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.final_verdict == "PASS"
    assert assessment.confidence == "MEDIUM"


def test_no_judge_with_retrieval_failure():
    result = make_result(
        failure_type="RETRIEVAL_FAILURE",
        answer_judge=None,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.final_verdict == "FAIL"
    assert assessment.confidence == "MEDIUM"


def test_custom_topic_coverage_threshold():
    detector = ConflictDetector(topic_coverage_threshold=0.5)

    judge = make_judge(answer_correct=True)

    result1 = make_result(
        topic_coverage=0.4,
        answer_judge=judge,
    )
    assessment1 = detector.assess(result1)

    assert (
        ConflictType.JUDGE_CONTRADICTS_COVERAGE
        in assessment1.conflicts
    )

    result2 = make_result(
        topic_coverage=0.6,
        answer_judge=judge,
    )
    assessment2 = detector.assess(result2)

    assert (
        ConflictType.JUDGE_CONTRADICTS_COVERAGE
        not in assessment2.conflicts
    )


def test_deterministic_failure_cannot_be_promoted_to_pass():
    judge = make_judge(
        answer_correct=True,
        answer_grounded=True,
    )
    result = make_result(
        failure_type="RANKING_FAILURE",
        answer_judge=judge,
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.final_verdict != "PASS"
    assert assessment.final_verdict == "REVIEW"


def test_judge_system_error_is_not_a_semantic_conflict():
    failed_judge = JudgeVerdict(
        result=None,
        status=JudgeStatus.SYSTEM_ERROR,
        judge_model="fake-judge",
        judge_provider="fake",
        judge_latency_ms=10.0,
        judge_tokens=GenerationUsage(
            input_tokens=10,
            output_tokens=20,
        ),
        judge_cost_usd=0.05,
        structured_output_valid=False,
        judge_failure="API failure",
    )

    result = make_result(
        failure_type="PASS",
        topic_coverage=1.0,
    ).model_copy(
        update={"judge_verdict": failed_judge}
    )

    detector = ConflictDetector()
    assessment = detector.assess(result)

    assert assessment.conflicts == []
    assert assessment.final_verdict == "SYSTEM_ERROR"
    assert assessment.confidence == "LOW"
    assert assessment.system_error == "API failure"
    assert assessment.tier_3_signals == {}
