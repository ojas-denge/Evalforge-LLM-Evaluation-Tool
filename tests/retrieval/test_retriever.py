from app.retrieval.retriever import Retriever
from contextlib import contextmanager


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        self.requested_top_k = top_k
        return [
            {
                "chunk_id": "chunk-a",
                "document": "first candidate",
                "metadata": {"document_id": "a.md"},
                "distance": 0.1,
            },
            {
                "chunk_id": "chunk-b",
                "document": "second candidate",
                "metadata": {"document_id": "b.md"},
                "distance": 0.2,
            },
            {
                "chunk_id": "chunk-c",
                "document": "third candidate",
                "metadata": {"document_id": "c.md"},
                "distance": 0.3,
            },
        ][:top_k]


class FakeReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.2, 0.9, 0.4][: len(documents)]

class FakeTracer:
    def __init__(self) -> None:
        self.calls = []

    @contextmanager
    def retrieval(self, name, *, input=None, metadata=None):
        observation = FakeObservation()
        self.calls.append(
            {
                "name": name,
                "input": input,
                "metadata": metadata,
                "observation": observation,
            }
        )
        yield observation


class FakeObservation:
    def __init__(self) -> None:
        self.output = None
        self.metadata = None

    def update(self, *, output=None, metadata=None):
        self.output = output
        self.metadata = metadata


def test_dense_retrieval_preserves_vector_store_order() -> None:
    vector_store = FakeVectorStore()
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranking_enabled=False,
    )

    result = retriever.retrieve("query", top_k=2)

    assert retriever.mode == "dense"
    assert vector_store.requested_top_k == 2
    assert [item.document_id for item in result.results] == ["a.md", "b.md"]


def test_reranking_rescores_expanded_dense_candidates() -> None:
    vector_store = FakeVectorStore()
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranker=FakeReranker(),
        reranker_candidate_k=3,
    )

    result = retriever.retrieve("query", top_k=2)

    assert retriever.mode == "dense_reranked"
    assert vector_store.requested_top_k == 3
    assert [item.document_id for item in result.results] == ["b.md", "c.md"]
    assert [item.rank for item in result.results] == [1, 2]

def test_explicit_candidate_k_controls_dense_candidate_pool() -> None:
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranker=FakeReranker(),
        candidate_k=2,
        reranker_candidate_k=10,
    )

    retriever.retrieve("query", top_k=2)

    assert vector_store.requested_top_k == 2


def test_candidate_k_cannot_be_smaller_than_top_k() -> None:
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranker=FakeReranker(),
        candidate_k=1,
    )

    retriever.retrieve("query", top_k=2)

    assert vector_store.requested_top_k == 2


def test_bm25_mode_does_not_initialize_embeddings(monkeypatch) -> None:
    import app.retrieval.retriever as retriever_module

    class FailingEmbeddingService:
        def __init__(self) -> None:
            raise AssertionError(
                "Embedding service should not be initialized for BM25"
            )

    monkeypatch.setattr(
        retriever_module,
        "EmbeddingService",
        FailingEmbeddingService,
    )

    retriever = Retriever(
        vector_store=FakeVectorStore(),
        lexical_only=True,
    )

    assert retriever.mode == "bm25"
    assert retriever.embedding_service is None


def test_retrieval_emits_observability_data() -> None:
    vector_store = FakeVectorStore()
    tracer = FakeTracer()

    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        reranking_enabled=False,
        tracer=tracer,
    )

    result = retriever.retrieve(
        "query",
        top_k=2,
        trace_metadata={
            "evaluation_run_id": "run-1",
            "case_id": "case-1",
        },
    )

    assert len(tracer.calls) == 1

    call = tracer.calls[0]

    assert call["name"] == "retrieval"
    assert call["input"] == {"query": "query"}

    assert call["metadata"]["mode"] == "dense"
    assert call["metadata"]["candidate_k"] == 2
    assert call["metadata"]["top_k"] == 2
    assert call["metadata"]["evaluation_run_id"] == "run-1"
    assert call["metadata"]["case_id"] == "case-1"

    assert call["observation"].output == {
        "status": "success",
        "result_count": 2,
        "document_ids": ["a.md", "b.md"],
        "ranked_results": [
            {
                "rank": 1,
                "chunk_id": "chunk-a",
                "document_id": "a.md",
                "distance": 0.1,
            },
            {
                "rank": 2,
                "chunk_id": "chunk-b",
                "document_id": "b.md",
                "distance": 0.2,
            },
        ],
    }

    assert call["observation"].metadata["latency_ms"] >= 0
    assert len(result.results) == 2
