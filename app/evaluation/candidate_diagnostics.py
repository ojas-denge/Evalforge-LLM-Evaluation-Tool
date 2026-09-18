from app.evaluation import EvaluationDataset
from app.retrieval.retriever import Retriever


def main() -> None:
    dataset = EvaluationDataset.load()

    reranked = Retriever(
        reranking_enabled=True,
        candidate_k=17,
    )

    failures = []

    for case in dataset.cases:
        result = reranked.retrieve(case.question, top_k=5)
        retrieved = [item.document_id for item in result.results]

        if not all(
            document in retrieved
            for document in case.expected_documents
        ):
            failures.append(case)

    dense = Retriever(candidate_k=17)
    bm25 = Retriever(
        lexical_only=True,
        candidate_k=17,
    )
    hybrid = Retriever(
        hybrid_retrieval_enabled=True,
        candidate_k=17,
    )

    for case in failures:
        print(f"\n{case.case_id}")

        expected = set(case.expected_documents)
        print(f"expected: {case.expected_documents}")

        for name, retriever in (
            ("dense", dense),
            ("bm25", bm25),
            ("hybrid", hybrid),
        ):
            result = retriever.retrieve(
                case.question,
                top_k=17,
            )

            documents = [
                item.document_id
                for item in result.results
            ]

            found = expected & set(documents)

            ranks = {
                document: documents.index(document) + 1
                for document in found
            }

            print(
                f"{name}: "
                f"found={sorted(found)} "
                f"ranks={ranks}"
            )


if __name__ == "__main__":
    main()
