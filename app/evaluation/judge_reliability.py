from pydantic import BaseModel, Field


class JudgeReliabilityReport(BaseModel):
    """Aggregated reliability metrics for a judge across a dataset."""
    structured_output_validity_rate: float = Field(ge=0.0, le=1.0)
    repeatability_score: float | None = Field(default=None, ge=0.0, le=1.0)
    perturbation_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    inter_judge_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    gold_set_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    total_cases: int = Field(ge=0)
    failure_count: int = Field(ge=0)


def compute_repeatability(
    runs: list[list[dict]],
) -> float:
    """Compute repeatability across N runs of the same dataset.

    Each run is a list of dicts with at minimum:
    - case_id: str
    - answer_correct: bool
    - answer_grounded: bool

    Returns the fraction of cases where all runs agree on
    both answer_correct and answer_grounded.
    """
    if not runs or len(runs) < 2:
        raise ValueError("Need at least 2 runs to compute repeatability")

    # Index runs by case_id
    indexed_runs = []
    for run in runs:
        indexed_runs.append({item["case_id"]: item for item in run})

    # Get common case_ids
    case_ids = set(indexed_runs[0].keys())
    for indexed in indexed_runs[1:]:
        case_ids &= set(indexed.keys())

    if not case_ids:
        raise ValueError("No common case IDs across runs")

    agreed = 0
    for case_id in case_ids:
        verdicts = [
            (indexed[case_id]["answer_correct"], indexed[case_id]["answer_grounded"])
            for indexed in indexed_runs
        ]
        if len(set(verdicts)) == 1:
            agreed += 1

    return agreed / len(case_ids)


def compute_inter_judge_agreement(
    judge_results: list[list[dict]],
) -> float:
    """Compute agreement across different judges on the same dataset.

    Same interface as compute_repeatability — each list is results
    from a different judge (not repeated runs of the same judge).
    """
    # The math is identical — we're just checking if different judges
    # agree on each case.
    return compute_repeatability(judge_results)


def compute_gold_set_agreement(
    judge_results: list[dict],
    gold_labels: list[dict],
) -> float:
    """Compute agreement between judge results and human gold labels.

    Both lists have dicts with:
    - case_id: str
    - answer_correct: bool
    - answer_grounded: bool
    """
    judge_index = {item["case_id"]: item for item in judge_results}
    gold_index = {item["case_id"]: item for item in gold_labels}

    common_ids = set(judge_index.keys()) & set(gold_index.keys())

    if not common_ids:
        raise ValueError("No common case IDs between judge and gold")

    agreed = 0
    for case_id in common_ids:
        j = judge_index[case_id]
        g = gold_index[case_id]
        if (
            j["answer_correct"] == g["answer_correct"]
            and j["answer_grounded"] == g["answer_grounded"]
        ):
            agreed += 1

    return agreed / len(common_ids)
