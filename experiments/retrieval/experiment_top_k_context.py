from collections import Counter

from app.evaluation import EvaluationDataset
from app.evaluation.metrics import topic_coverage
from app.evaluation.runner import Evaluator
from app.retrieval.retriever import Retriever


def build_top_k_answer(retrieved_documents) -> str:
    return "\n\n".join(
        document.text
        for document in retrieved_documents
    )


def main() -> None:
    dataset = EvaluationDataset.load()

    retriever = Retriever(
        hybrid_retrieval_enabled=False,
        reranking_enabled=False,
        lexical_only=False,
    )

    evaluator = Evaluator(retriever=retriever)

    scores = []
    baseline_scores = []

    for case in dataset.cases:
        if not case.expected_topics:
            continue

        retrieval = retriever.retrieve(
            query=case.question,
            top_k=5,
        )

        top_1_answer = (
            retrieval.results[0].text
            if retrieval.results
            else "No relevant context was retrieved."
        )

        top_5_answer = build_top_k_answer(
            retrieval.results
        )

        baseline_score = topic_coverage(
            case.expected_topics,
            top_1_answer,
        )

        top_5_score = topic_coverage(
            case.expected_topics,
            top_5_answer,
        )

        baseline_scores.append(baseline_score)
        scores.append(top_5_score)

        print(
            f"{case.case_id}: "
            f"rank1={baseline_score:.2f} "
            f"top5={top_5_score:.2f} "
            f"delta={top_5_score - baseline_score:+.2f}"
        )

    print()
    print("TOP-K CONTEXT EXPERIMENT")
    print(f"Labeled cases: {len(scores)}")

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    top5_mean = sum(scores) / len(scores)

    print(f"Baseline rank-1 topic coverage: {baseline_mean:.3f}")
    print(f"Top-5 topic coverage:           {top5_mean:.3f}")
    print(f"Absolute improvement:            {top5_mean - baseline_mean:+.3f}")

    improved = sum(
        top5 > baseline
        for top5, baseline in zip(scores, baseline_scores)
    )
    unchanged = sum(
        top5 == baseline
        for top5, baseline in zip(scores, baseline_scores)
    )
    regressed = sum(
        top5 < baseline
        for top5, baseline in zip(scores, baseline_scores)
    )

    print()
    print("CASE OUTCOMES")
    print(f"Improved:   {improved}")
    print(f"Unchanged:  {unchanged}")
    print(f"Regressed:  {regressed}")


if __name__ == "__main__":
    main()
