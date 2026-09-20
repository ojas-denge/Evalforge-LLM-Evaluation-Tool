from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset
from app.retrieval.retriever import Retriever


CASE_IDS = [
    "cross_003",
    "routing_006",
    "cross_001",
    "evaluation_strategy_003",
    "deployment_004",
    "embedding_004",
    "architecture_003",
    "cost_005",
    "context_004",
    "structured_004",
]

TOP_K = 5
OUTPUT_PATH = Path("logs/g3c_retrieval_checkpoint.json")


def metrics(retrieval, expected_documents):
    expected = set(expected_documents)
    ranks = [
        doc.rank
        for doc in retrieval.results
        if doc.document_id in expected
    ]

    hit = {}
    for k in (1, 3, 5):
        hit[k] = int(any(rank <= k for rank in ranks))

    recall = (
        len({doc.document_id for doc in retrieval.results if doc.document_id in expected})
        / len(expected)
        if expected
        else 0.0
    )

    mrr = 1.0 / min(ranks) if ranks else 0.0

    return {
        "hit_at_1": hit[1],
        "hit_at_3": hit[3],
        "hit_at_5": hit[5],
        "recall_at_5": recall,
        "mrr": mrr,
        "expected_ranks": ranks,
    }


def serialize(retrieval):
    return {
        "latency_ms": retrieval.latency_ms,
        "results": [
            {
                "rank": doc.rank,
                "document_id": doc.document_id,
                "chunk_id": doc.chunk_id,
                "distance": doc.distance,
                "text": doc.text,
            }
            for doc in retrieval.results
        ],
    }


def main():
    dataset = EvaluationDataset.load()
    by_id = {case.case_id: case for case in dataset.cases}

    missing = [cid for cid in CASE_IDS if cid not in by_id]
    if missing:
        raise RuntimeError(f"Missing case IDs: {missing}")

    dense = Retriever(
        reranking_enabled=False,
        hybrid_retrieval_enabled=False,
        candidate_k=TOP_K,
    )

    crossencoder = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=TOP_K,
        reranker_candidate_k=TOP_K,
    )

    cases = []

    for index, case_id in enumerate(CASE_IDS, start=1):
        case = by_id[case_id]
        expected = case.expected_documents

        dense_result = dense.retrieve(case.question, top_k=TOP_K)
        ce_result = crossencoder.retrieve(case.question, top_k=TOP_K)

        dense_docs = [d.document_id for d in dense_result.results]
        ce_docs = [d.document_id for d in ce_result.results]

        cases.append({
            "case_id": case_id,
            "question": case.question,
            "expected_documents": expected,
            "dense": {
                "metrics": metrics(dense_result, expected),
                "retrieval": serialize(dense_result),
            },
            "dense_crossencoder": {
                "metrics": metrics(ce_result, expected),
                "retrieval": serialize(ce_result),
            },
            "diagnostics": {
                "document_set_changed": set(dense_docs) != set(ce_docs),
                "order_changed": dense_docs != ce_docs,
                "expected_rank_changed": (
                    metrics(dense_result, expected)["expected_ranks"]
                    != metrics(ce_result, expected)["expected_ranks"]
                ),
                "expected_documents_missing_dense": [
                    doc for doc in expected if doc not in dense_docs
                ],
                "expected_documents_missing_crossencoder": [
                    doc for doc in expected if doc not in ce_docs
                ],
            },
        })

        print(f"[{index}/{len(CASE_IDS)}] {case_id}")

    summary = {
        "dataset_size": len(CASE_IDS),
        "top_k": TOP_K,
        "document_set_changed_cases": sum(
            c["diagnostics"]["document_set_changed"] for c in cases
        ),
        "order_changed_cases": sum(
            c["diagnostics"]["order_changed"] for c in cases
        ),
        "expected_rank_changed_cases": sum(
            c["diagnostics"]["expected_rank_changed"] for c in cases
        ),
        "missing_expected_changed_cases": sum(
            bool(c["diagnostics"]["expected_documents_missing_dense"])
            != bool(c["diagnostics"]["expected_documents_missing_crossencoder"])
            for c in cases
        ),
    }

    output = {
        "experiment": "g3c_retrieval_checkpoint",
        "version": "0.1",
        "purpose": "Validate retrieval perturbations for the final G3C candidate pool before generation.",
        "summary": summary,
        "cases": cases,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nG3C retrieval checkpoint complete")
    print(json.dumps(summary, indent=2))
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
