import json

from app.evaluation import EvaluationDataset
from app.evaluation.runner import Evaluator
from app.retrieval.retriever import Retriever


def evaluate(candidate_k: int, dataset: EvaluationDataset):
    retriever = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=candidate_k,
    )

    evaluator = Evaluator(retriever=retriever)

    return [
        evaluator.evaluate_case(case)
        for case in dataset.cases
    ]


def main() -> None:
    dataset = EvaluationDataset.load()

    k5_results = evaluate(5, dataset)
    k10_results = evaluate(10, dataset)

    differences = []

    for case, k5, k10 in zip(
        dataset.cases,
        k5_results,
        k10_results,
    ):
        k5_docs = [
            evidence.document_id
            for evidence in k5.retrieved_evidence
        ]

        k10_docs = [
            evidence.document_id
            for evidence in k10.retrieved_evidence
        ]

        if k5_docs != k10_docs:
            differences.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "k5": {
                        "documents": k5_docs,
                        "failure": k5.failure_type,
                    },
                    "k10": {
                        "documents": k10_docs,
                        "failure": k10.failure_type,
                    },
                }
            )

    print(
        json.dumps(
            {
                "total_cases": len(dataset.cases),
                "different_cases": len(differences),
                "cases": differences,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
