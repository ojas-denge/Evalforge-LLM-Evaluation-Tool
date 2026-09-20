from app.evaluation.ci_gate import EvaluationCIGate
from app.evaluation.comparison import compare_runs
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.regression import RegressionPolicy
from app.evaluation.reporting import print_comparison
from app.evaluation.runner import Evaluator
from app.retrieval.retriever import Retriever


def main():
    dataset = EvaluationDataset.load(
        "data/evaluation_cases.json"
    )

    baseline_retriever = Retriever(
        reranking_enabled=False,
    )

    candidate_retriever = Retriever(
        reranking_enabled=True,
        reranker_candidate_k=10,
    )

    baseline_evaluator = Evaluator(
        retriever=baseline_retriever,
    )

    candidate_evaluator = Evaluator(
        retriever=candidate_retriever,
    )

    print("Running baseline evaluation...")
    baseline = baseline_evaluator.evaluate_dataset(dataset)

    print("Running candidate evaluation...")
    candidate = candidate_evaluator.evaluate_dataset(dataset)

    comparison = compare_runs(
        baseline,
        candidate,
    )

    policy = RegressionPolicy(
        max_mrr_drop=0.02,
        max_latency_increase_ms=100.0,
    )

    gate = EvaluationCIGate(policy)
    gate_result = gate.evaluate(comparison)

    print("\n================================")
    print("        EVALFORGE COMPARISON")
    print("================================")

    print(f"\nBaseline:  {baseline.run_id}")
    print(f"Candidate: {candidate.run_id}")

    print_comparison(comparison)

    print("\n=== CI Gate ===")
    print(f"Passed: {gate_result.passed}")

    if gate_result.reasons:
        print("Reasons:")
        for reason in gate_result.reasons:
            print(f"  - {reason}")
    else:
        print("Reasons: none")

    if not gate_result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()