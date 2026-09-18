from enum import Enum

from app.models.evaluation import EvaluationResult


class TriageDecision(str, Enum):
    DETERMINISTIC_PASS = "deterministic_pass"
    DETERMINISTIC_FAIL = "deterministic_fail"
    NEEDS_JUDGE = "needs_judge"


def triage_case(
    result: EvaluationResult,
    *,
    topic_coverage_pass_threshold: float = 0.9,
) -> TriageDecision:
    """Decide whether a case needs LLM judging based on deterministic signals.

    Cases with clear deterministic outcomes skip the LLM entirely.
    Only ambiguous cases get sent to the judge.
    """
    if result.failure_type == "RETRIEVAL_FAILURE":
        return TriageDecision.DETERMINISTIC_FAIL

    if (
        result.failure_type == "PASS"
        and result.topic_coverage is not None
        and result.topic_coverage >= topic_coverage_pass_threshold
    ):
        return TriageDecision.DETERMINISTIC_PASS

    return TriageDecision.NEEDS_JUDGE


def triage_dataset(
    results: list[EvaluationResult],
    *,
    topic_coverage_pass_threshold: float = 0.9,
) -> dict[TriageDecision, list[EvaluationResult]]:
    """Partition evaluation results by triage decision."""
    partitioned: dict[TriageDecision, list[EvaluationResult]] = {
        TriageDecision.DETERMINISTIC_PASS: [],
        TriageDecision.DETERMINISTIC_FAIL: [],
        TriageDecision.NEEDS_JUDGE: [],
    }

    for result in results:
        decision = triage_case(
            result,
            topic_coverage_pass_threshold=topic_coverage_pass_threshold,
        )
        partitioned[decision].append(result)

    return partitioned
