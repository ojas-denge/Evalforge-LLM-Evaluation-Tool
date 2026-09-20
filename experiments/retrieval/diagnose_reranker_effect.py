import json
from pathlib import Path
from time import perf_counter

from app.evaluation import EvaluationDataset
from app.evaluation.diagnostics import analyze_retrieval
from app.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

TOP_K = 5
CANDIDATE_K = 5
OUTPUT_PATH = Path("logs/dense_vs_crossencoder_forensic.json")


def document_ids(results):
    return [r["metadata"]["document_id"] for r in results]


def metrics(expected, retrieved):
    return {
        "hit_at_1": hit_at_k(expected, retrieved, 1),
        "hit_at_3": hit_at_k(expected, retrieved, 3),
        "hit_at_5": hit_at_k(expected, retrieved, 5),
        "recall_at_1": recall_at_k(expected, retrieved, 1),
        "recall_at_3": recall_at_k(expected, retrieved, 3),
        "recall_at_5": recall_at_k(expected, retrieved, 5),
        "mrr": reciprocal_rank(expected, retrieved),
    }


def classify(dense_docs, reranked_docs, expected):
    dense_hits = set(dense_docs) & set(expected)
    reranked_hits = set(reranked_docs) & set(expected)

    dense_mrr = reciprocal_rank(expected, dense_docs)
    reranked_mrr = reciprocal_rank(expected, reranked_docs)

    evidence_delta = len(reranked_hits) - len(dense_hits)
    mrr_delta = reranked_mrr - dense_mrr

    if evidence_delta > 0 and mrr_delta >= 0:
        return "EVIDENCE_GAIN"
    if evidence_delta < 0 and mrr_delta <= 0:
        return "EVIDENCE_LOSS"
    if evidence_delta == 0 and mrr_delta > 0:
        return "RANKING_IMPROVEMENT"
    if evidence_delta == 0 and mrr_delta < 0:
        return "RANKING_REGRESSION"
    if evidence_delta > 0:
        return "MIXED_IMPROVEMENT"
    if evidence_delta < 0:
        return "MIXED_REGRESSION"
    if dense_docs != reranked_docs:
        return "NEUTRAL_REORDER"
    return "UNCHANGED"


def main():
    dataset = EvaluationDataset.load()

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    reranker = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=CANDIDATE_K,
    )

    if reranker.reranker is None:
        raise RuntimeError("CrossEncoder reranker was not initialized")

    counts = {}
    transitions = {}
    cases = []

    metric_names = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
    )

    aggregate_dense = {k: 0.0 for k in metric_names}
    aggregate_ce = {k: 0.0 for k in metric_names}

    dense_total_ms = 0.0
    ce_total_ms = 0.0

    for number, case in enumerate(dataset.cases, 1):
        # One and only one first-stage retrieval.
        start = perf_counter()
        query_embedding = embedding_service.embed_query(case.question)
        candidate_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=CANDIDATE_K,
        )
        dense_ms = (perf_counter() - start) * 1000

        if len(candidate_results) != CANDIDATE_K:
            raise RuntimeError(
                f"{case.case_id}: expected {CANDIDATE_K} candidates, "
                f"got {len(candidate_results)}"
            )

        candidate_docs = document_ids(candidate_results)

        # With candidate_k == top_k, dense final ranking is the
        # first-stage candidate ordering.
        dense_docs = candidate_docs[:TOP_K]

        # Score EXACTLY the same candidate objects.
        start = perf_counter()
        scores = reranker.reranker.score(
            case.question,
            [r["document"] for r in candidate_results],
        )
        ce_ms = (perf_counter() - start) * 1000

        if len(scores) != len(candidate_results):
            raise RuntimeError(
                f"{case.case_id}: {len(scores)} scores for "
                f"{len(candidate_results)} candidates"
            )

        ranked = sorted(
            zip(scores, candidate_results),
            key=lambda pair: pair[0],
            reverse=True,
        )

        ce_docs = document_ids([r for _, r in ranked[:TOP_K]])

        dense_metrics = metrics(case.expected_documents, dense_docs)
        ce_metrics = metrics(case.expected_documents, ce_docs)

        dense_diag = analyze_retrieval(
            case.expected_documents,
            dense_docs,
        )
        ce_diag = analyze_retrieval(
            case.expected_documents,
            ce_docs,
        )

        classification = classify(
            dense_docs,
            ce_docs,
            case.expected_documents,
        )

        transition = (
            f"{dense_diag.failure_type} -> "
            f"{ce_diag.failure_type}"
        )

        counts[classification] = counts.get(classification, 0) + 1
        transitions[transition] = transitions.get(transition, 0) + 1

        for key in aggregate_dense:
            aggregate_dense[key] += dense_metrics[key]
            aggregate_ce[key] += ce_metrics[key]

        dense_total_ms += dense_ms
        ce_total_ms += ce_ms

        cases.append({
            "case_id": case.case_id,
            "question": case.question,
            "expected_documents": case.expected_documents,
            "classification": classification,
            "candidate_pool": candidate_docs,
            "dense": {
                "documents": dense_docs,
                "failure_type": dense_diag.failure_type,
                "metrics": dense_metrics,
            },
            "crossencoder": {
                "documents": ce_docs,
                "failure_type": ce_diag.failure_type,
                "metrics": ce_metrics,
                "scores": [
                    {
                        "document_id": r["metadata"]["document_id"],
                        "score": float(score),
                    }
                    for score, r in zip(scores, candidate_results)
                ],
            },
        })

        print(
            f"[{number}/{len(dataset.cases)}] "
            f"{case.case_id} -> {classification}"
        )

    n = len(dataset.cases)

    for key in aggregate_dense:
        aggregate_dense[key] /= n
        aggregate_ce[key] /= n

    output = {
        "experiment": "dense_vs_crossencoder_forensic",
        "dataset_size": n,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "same_candidate_pool": True,
        "classification_counts": counts,
        "failure_transitions": transitions,
        "aggregate_metrics": {
            "dense": aggregate_dense,
            "crossencoder": aggregate_ce,
            "delta": {
                key: aggregate_ce[key] - aggregate_dense[key]
                for key in aggregate_dense
            },
        },
        "mean_dense_first_stage_latency_ms": dense_total_ms / n,
        "mean_crossencoder_latency_ms": ce_total_ms / n,
        "cases": cases,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n==============================================")
    print("DENSE vs CROSSENCODER FORENSIC ANALYSIS")
    print("==============================================")

    print("\nClassification:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")

    print("\nFailure transitions:")
    for key, value in sorted(transitions.items()):
        print(f"  {key}: {value}")

    print("\nAggregate metrics:")
    for key in aggregate_dense:
        delta = aggregate_ce[key] - aggregate_dense[key]
        print(
            f"  {key}: "
            f"{aggregate_dense[key]:.4f} -> "
            f"{aggregate_ce[key]:.4f} "
            f"({delta:+.4f})"
        )

    print(
        f"\nDense latency: "
        f"{dense_total_ms / n:.2f} ms"
    )
    print(
        f"CrossEncoder latency: "
        f"{ce_total_ms / n:.2f} ms"
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
