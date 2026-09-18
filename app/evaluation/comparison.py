from dataclasses import dataclass

from app.models.evaluation import EvaluationRun


@dataclass(frozen=True)
class CaseChange:
    case_id: str
    hit_at_1_delta: float
    mrr_delta: float
    latency_delta_ms: float
    baseline_failure_type: str
    candidate_failure_type: str
    baseline_first_relevant_rank: int | None
    candidate_first_relevant_rank: int | None
    baseline_missing_documents: list[str]
    candidate_missing_documents: list[str]


@dataclass(frozen=True)
class EvaluationComparison:
    baseline_run_id: str
    candidate_run_id: str
    metric_deltas: dict[str, float]
    case_changes: list[CaseChange]
    mean_correctness_delta: float | None = None
    mean_groundedness_delta: float | None = None
    conflict_rate_delta: float | None = None
    cost_delta_usd: float | None = None


def _mean_correctness(run: EvaluationRun) -> float | None:
    judged = [
        result for result in run.results
        if result.answer_judge is not None
    ]
    if not judged:
        return None
    return sum(
        1.0 for r in judged if r.answer_judge.answer_correct
    ) / len(judged)


def _mean_groundedness(run: EvaluationRun) -> float | None:
    judged = [
        result for result in run.results
        if result.answer_judge is not None
    ]
    if not judged:
        return None
    return sum(
        1.0 for r in judged if r.answer_judge.answer_grounded
    ) / len(judged)


def compare_runs(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
) -> EvaluationComparison:
    baseline_cases = {
        result.case_id: result
        for result in baseline.results
    }

    candidate_cases = {
        result.case_id: result
        for result in candidate.results
    }

    if set(baseline_cases) != set(candidate_cases):
        raise ValueError(
            "Evaluation runs must contain the same case IDs"
        )

    metric_deltas = {
        "mean_hit_at_1": (
            candidate.mean_hit_at_1
            - baseline.mean_hit_at_1
        ),
        "mean_hit_at_3": (
            candidate.mean_hit_at_3
            - baseline.mean_hit_at_3
        ),
        "mean_hit_at_5": (
            candidate.mean_hit_at_5
            - baseline.mean_hit_at_5
        ),
        "mean_recall_at_1": (
            candidate.mean_recall_at_1
            - baseline.mean_recall_at_1
        ),
        "mean_recall_at_3": (
            candidate.mean_recall_at_3
            - baseline.mean_recall_at_3
        ),
        "mean_recall_at_5": (
            candidate.mean_recall_at_5
            - baseline.mean_recall_at_5
        ),
        "mean_mrr": (
            candidate.mean_mrr
            - baseline.mean_mrr
        ),
        "mean_retrieval_latency_ms": (
            candidate.mean_retrieval_latency_ms
            - baseline.mean_retrieval_latency_ms
        ),
    }

    case_changes = [
        CaseChange(
            case_id=case_id,
            hit_at_1_delta=(
                candidate_cases[case_id].hit_at_1
                - baseline_cases[case_id].hit_at_1
            ),
            mrr_delta=(
                candidate_cases[case_id].mrr
                - baseline_cases[case_id].mrr
            ),
            latency_delta_ms=(
                candidate_cases[case_id].retrieval_latency_ms
                - baseline_cases[case_id].retrieval_latency_ms
            ),
            baseline_failure_type=(
                baseline_cases[case_id].failure_type
            ),
            candidate_failure_type=(
                candidate_cases[case_id].failure_type
            ),
            baseline_first_relevant_rank=(
                baseline_cases[case_id].first_relevant_rank
            ),
            candidate_first_relevant_rank=(
                candidate_cases[case_id].first_relevant_rank
            ),
            baseline_missing_documents=(
                baseline_cases[case_id].missing_documents
            ),
            candidate_missing_documents=(
                candidate_cases[case_id].missing_documents
            ),
        )
        for case_id in sorted(baseline_cases)
    ]

    baseline_correctness = _mean_correctness(baseline)
    candidate_correctness = _mean_correctness(candidate)
    mean_correctness_delta = None
    if baseline_correctness is not None and candidate_correctness is not None:
        mean_correctness_delta = candidate_correctness - baseline_correctness

    baseline_groundedness = _mean_groundedness(baseline)
    candidate_groundedness = _mean_groundedness(candidate)
    mean_groundedness_delta = None
    if baseline_groundedness is not None and candidate_groundedness is not None:
        mean_groundedness_delta = candidate_groundedness - baseline_groundedness

    return EvaluationComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        metric_deltas=metric_deltas,
        case_changes=case_changes,
        mean_correctness_delta=mean_correctness_delta,
        mean_groundedness_delta=mean_groundedness_delta,
    )
