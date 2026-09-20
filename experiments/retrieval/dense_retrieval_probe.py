from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


def main() -> None:
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    queries = [
        "How does EvalForge route requests between different language models?",
        "What are the rules for retrieving documents?",
        "What is the latency target for the system?",
        "How does EvalForge control inference costs?",
        "How are structured outputs validated?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        query_embedding = embedding_service.embed_query(query)

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=3,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\n#{rank}")
            print(f"Chunk:    {result['chunk_id']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Document: {result['metadata']['document_id']}")
            print(f"Text:     {result['document'][:300]}...")


if __name__ == "__main__":
    main()
