import json

from app.evaluation import EvaluationDataset
from app.retrieval.retriever import Retriever


def run_case(
    retriever: Retriever,
    question: str,
    expected_documents: list[str],
    top_k: int = 5,
) -> dict:
    if retriever.embedding_service is None:
        raise RuntimeError("Dense retrieval requires embeddings")

    query_embedding = retriever.embedding_service.embed_query(question)

    candidate_k = retriever._resolve_candidate_k(top_k)

    dense_candidates = retriever.vector_store.search(
        query_embedding=query_embedding,
        top_k=candidate_k,
    )

    candidate_documents = [
        result["metadata"]["document_id"]
        for result in dense_candidates
    ]

    reranker_scores = retriever.reranker.score(
        question,
        [result["document"] for result in dense_candidates],
    )

    ranked_candidates = [
        result
        for _, result in sorted(
            enumerate(dense_candidates),
            key=lambda item: (
                -reranker_scores[item[0]],
                item[0],
            ),
        )[:top_k]
    ]

    final_documents = [
        result["metadata"]["document_id"]
        for result in ranked_candidates
    ]

    expected = set(expected_documents)

    candidate_hits = [
        document
        for document in expected
        if document in candidate_documents
    ]

    final_ranks = {
        document: (
            final_documents.index(document) + 1
            if document in final_documents
            else None
        )
        for document in expected
    }

    return {
        "candidate_k": candidate_k,
        "candidate_documents": candidate_documents,
        "expected_documents": expected_documents,
        "candidate_hits": candidate_hits,
        "final_documents": final_documents,
        "final_ranks": final_ranks,
    }


def main() -> None:
    dataset = EvaluationDataset.load()

    retriever_k5 = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=5,
    )

    retriever_k10 = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=10,
    )

    differences = []

    for case in dataset.cases:
        result_k5 = run_case(
            retriever_k5,
            case.question,
            case.expected_documents,
        )

        result_k10 = run_case(
            retriever_k10,
            case.question,
            case.expected_documents,
        )

        if result_k5["final_documents"] != result_k10["final_documents"]:
            differences.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "k5": result_k5,
                    "k10": result_k10,
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
