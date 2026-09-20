import sys

from app.db.repository import EvaluationRepository
from app.evaluation.comparison import compare_runs
from app.evaluation.regression import RegressionPolicy
from app.evaluation.reporting import print_comparison


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/compare_persisted_runs.py "
            "<baseline_run_id> <candidate_run_id>"
        )
        raise SystemExit(1)

    baseline_id = sys.argv[1]
    candidate_id = sys.argv[2]

    repository = EvaluationRepository()

    baseline = repository.get_run(baseline_id)
    candidate = repository.get_run(candidate_id)

    if baseline is None:
        print(f"Baseline run not found: {baseline_id}")
        raise SystemExit(1)

    if candidate is None:
        print(f"Candidate run not found: {candidate_id}")
        raise SystemExit(1)

    comparison = compare_runs(baseline, candidate)

    policy = RegressionPolicy(
        max_mrr_drop=0.02,
        max_latency_increase_ms=100.0,
    )

    regression = policy.evaluate(comparison)

    print("\n================================")
    print("   PERSISTED EVALUATION COMPARE")
    print("================================")

    print(f"\nBaseline:  {baseline.run_id}")
    print(f"Candidate: {candidate.run_id}")

    print_comparison(comparison)

    print("\n=== Regression Policy ===")
    print(f"Regression detected: {regression.regression_detected}")

    if regression.reasons:
        print("Reasons:")
        for reason in regression.reasons:
            print(f"  - {reason}")
    else:
        print("Reasons: none")


if __name__ == "__main__":
    main()