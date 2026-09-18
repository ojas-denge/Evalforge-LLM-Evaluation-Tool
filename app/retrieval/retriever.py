from dataclasses import dataclass
from time import perf_counter
from app.observability.tracing import Tracer

from app.core.config import get_settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.lexical import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedDocument:
    rank: int
    chunk_id: str
    document_id: str
    text: str
    distance: float | None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    results: list[RetrievedDocument]
    latency_ms: float


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        reranking_enabled: bool | None = None,
        reranker_candidate_k: int | None = None,
        hybrid_retrieval_enabled: bool | None = None,
        lexical_retriever: BM25Retriever | None = None,
        lexical_only: bool = False,
        candidate_k: int | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        settings = get_settings()
        self.tracer = tracer or Tracer()

        self.lexical_only = lexical_only
        self.vector_store = vector_store or VectorStore()

        self.reranking_enabled = (
            reranking_enabled
            if reranking_enabled is not None
            else settings.reranking_enabled
        )

        self.reranker_candidate_k = (
            reranker_candidate_k
            if reranker_candidate_k is not None
            else settings.reranker_candidate_k
        )

        self.hybrid_retrieval_enabled = (
            hybrid_retrieval_enabled
            if hybrid_retrieval_enabled is not None
            else settings.hybrid_retrieval_enabled
        )

        self.lexical_candidate_k = settings.lexical_candidate_k
        self.lexical_retriever = lexical_retriever or BM25Retriever()

        self.candidate_k = candidate_k

        if self.reranker_candidate_k <= 0:
            raise ValueError("reranker_candidate_k must be greater than zero")

        if self.candidate_k is not None and self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        # BM25-only retrieval does not need embeddings.
        if self.lexical_only:
            self.embedding_service = None
        else:
            self.embedding_service = (
                embedding_service or EmbeddingService()
            )

        if reranker is not None:
            self.reranker = reranker
            self.reranking_enabled = True
        elif self.reranking_enabled:
            self.reranker = CrossEncoderReranker(
                settings.reranker_model
            )
        else:
            self.reranker = None

    @property
    def mode(self) -> str:
        if self.lexical_only:
            return "bm25"

        if self.hybrid_retrieval_enabled and self.reranking_enabled:
            return "hybrid_reranked"

        if self.hybrid_retrieval_enabled:
            return "hybrid"

        if self.reranking_enabled:
            return "dense_reranked"

        return "dense"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start_time = perf_counter()
        candidate_k = self._resolve_candidate_k(top_k)

        with self.tracer.retrieval(
            name="retrieval",
            input={"query": query},
            metadata={
                "mode": self.mode,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "reranking_enabled": self.reranking_enabled,
                "hybrid_retrieval_enabled": self.hybrid_retrieval_enabled,
            },
        ) as observation:

            if self.lexical_only:
                raw_results = self.lexical_retriever.search(
                    query,
                    self.vector_store.all_chunks(),
                    top_k=candidate_k,
                )
            else:
                if self.embedding_service is None:
                    raise RuntimeError(
                        "Embedding service is required for dense retrieval"
                    )

                query_embedding = self.embedding_service.embed_query(query)

                dense_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=candidate_k,
                )

                if self.hybrid_retrieval_enabled:
                    lexical_results = self.lexical_retriever.search(
                        query,
                        self.vector_store.all_chunks(),
                        top_k=candidate_k,
                    )

                    raw_results = self._fuse_candidates(
                        dense_results,
                        lexical_results,
                    )
                else:
                    raw_results = dense_results

            if self.reranker is not None:
                scores = self.reranker.score(
                    query,
                    [result["document"] for result in raw_results],
                )

                if len(scores) != len(raw_results):
                    raise ValueError(
                        "Reranker must return one score per candidate"
                    )

                raw_results = [
                    result
                    for _, result in sorted(
                        enumerate(raw_results),
                        key=lambda candidate: (
                            -scores[candidate[0]],
                            candidate[0],
                        ),
                    )[:top_k]
                ]
            else:
                raw_results = raw_results[:top_k]

            results = [
                RetrievedDocument(
                    rank=rank,
                    chunk_id=result["chunk_id"],
                    document_id=result["metadata"]["document_id"],
                    text=result["document"],
                    distance=result["distance"],
                )
                for rank, result in enumerate(raw_results, start=1)
            ]

            latency_ms = (perf_counter() - start_time) * 1000

            if observation is not None:
                observation.update(
                    output={
                        "result_count": len(results),
                        "document_ids": [
                            result.document_id
                            for result in results
                        ],
                    },
                    metadata={
                        "latency_ms": latency_ms,
                    },
                )

        return RetrievalResult(
            query=query,
            results=results,
            latency_ms=latency_ms,
        )

    def _resolve_candidate_k(self, top_k: int) -> int:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.candidate_k is not None:
            return max(self.candidate_k, top_k)

        if self.reranking_enabled:
            return max(self.reranker_candidate_k, top_k)

        return top_k

    @staticmethod
    def _fuse_candidates(
        dense_results: list[dict],
        lexical_results: list[dict],
    ) -> list[dict]:
        candidates: dict[str, tuple[float, dict]] = {}

        for results in (dense_results, lexical_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                score, existing = candidates.get(
                    chunk_id,
                    (0.0, result),
                )

                candidates[chunk_id] = (
                    score + 1.0 / (60 + rank),
                    existing,
                )

        return [
            result
            for _, result in sorted(
                candidates.values(),
                key=lambda item: (
                    -item[0],
                    item[1]["chunk_id"],
                ),
            )
        ]
