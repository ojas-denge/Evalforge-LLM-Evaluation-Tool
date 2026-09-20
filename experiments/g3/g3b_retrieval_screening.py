"""
EvalForge G3B — Retrieval-Only Screening
-----------------------------------------
Screens the selected 16 G3B candidates with:
  A) Dense top-5
  B) Dense + CrossEncoder top-5

NO LLM calls.
NO answer generation.
NO production/default changes.

Outputs:
  logs/g3b_retrieval_screening.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.dataset import EvaluationDataset
from app.retrieval.retriever import Retriever


SELECTED_CASES = [
    "evaluation_strategy_003",
    "cross_003",
    "routing_006",
    "golden_004",
    "architecture_003",
    "context_004",
    "deployment_004",
    "embedding_004",
    "reliability_004",
    "retrieval_004",
    "cost_001",
    "cost_002",
    "latency_001",
    "validation_003",
    "validation_005",
    "reliability_002",
]

OUTPUT_PATH = Path("logs/g3b_retrieval_screening.json")


def doc_signature(doc: Any) -> dict[str, Any]:
    return {
        "rank": doc.rank,
        "chunk_id": doc.chunk_id,
        "document_id": doc.document_id,
        "distance": doc.distance,
        "text": doc.text,
    }


def expected_rank_info(results: list[Any], expected_docs: list[str]) -> dict[str, Any]:
    ranks = {}
    for expected in expected_docs:
        matching = [d.rank for d in results if d.document_id == expected]
        ranks[expected] = min(matching) if matching else None

    present = [doc for doc in expected_docs if ranks[doc] is not None]
    missing = [doc for doc in expected_docs if ranks[doc] is None]

    return {
        "expected_ranks": ranks,
        "expected_present": present,
        "expected_missing": missing,
        "all_expected_present": len(missing) == 0,
    }


def retrieval_metrics(results: list[Any], expected_docs: list[str]) -> dict[str, Any]:
    expected = set(expected_docs)

    def hit(k: int) -> float:
        return float(any(d.document_id in expected for d in results[:k]))

    retrieved = {d.document_id for d in results[:5]}
    recall5 = len(expected & retrieved) / len(expected) if expected else 0.0

    reciprocal = 0.0
    for d in results:
        if d.document_id in expected:
            reciprocal = 1.0 / d.rank
            break

    return {
        "hit_at_1": hit(1),
        "hit_at_3": hit(3),
        "hit_at_5": hit(5),
        "recall_at_5": recall5,
        "mrr": reciprocal,
    }


def serialize_condition(retrieval: Any, expected_docs: list[str]) -> dict[str, Any]:
    results = retrieval.results
    return {
        "latency_ms": retrieval.latency_ms,
        "metrics": retrieval_metrics(results, expected_docs),
        "expected": expected_rank_info(results, expected_docs),
        "results": [doc_signature(d) for d in results],
    }


def main() -> None:
    dataset = EvaluationDataset.load()
    cases = {case.case_id: case for case in dataset.cases}

    missing_cases = [cid for cid in SELECTED_CASES if cid not in cases]
    if missing_cases:
        raise RuntimeError(f"Selected cases missing from dataset: {missing_cases}")

    dense = Retriever(
        reranking_enabled=False,
        hybrid_retrieval_enabled=False,
        candidate_k=5,
    )

    crossencoder = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=5,
        reranker_candidate_k=5,
    )

    output: dict[str, Any] = {
        "experiment": "g3b_retrieval_only_screening",
        "dataset_size": len(SELECTED_CASES),
        "cases": [],
        "summary": {},
    }

    for index, case_id in enumerate(SELECTED_CASES, start=1):
        case = cases[case_id]
        expected_docs = list(case.expected_documents)

        print(f"[{index:02d}/{len(SELECTED_CASES)}] {case_id}")

        dense_result = dense.retrieve(case.question, top_k=5)
        ce_result = crossencoder.retrieve(case.question, top_k=5)

        dense_metrics = retrieval_metrics(dense_result.results, expected_docs)
        ce_metrics = retrieval_metrics(ce_result.results, expected_docs)

        dense_ids = [d.document_id for d in dense_result.results]
        ce_ids = [d.document_id for d in ce_result.results]

        dense_expected = expected_rank_info(dense_result.results, expected_docs)
        ce_expected = expected_rank_info(ce_result.results, expected_docs)

        dense_set = set(dense_ids)
        ce_set = set(ce_ids)

        record = {
            "case_id": case_id,
            "question": case.question,
            "category": case.category,
            "expected_documents": expected_docs,
            "dense": serialize_condition(dense_result, expected_docs),
            "dense_crossencoder": serialize_condition(ce_result, expected_docs),
            "comparison": {
                "document_set_changed": dense_set != ce_set,
                "dense_only_documents": sorted(dense_set - ce_set),
                "crossencoder_only_documents": sorted(ce_set - dense_set),
                "order_changed": dense_ids != ce_ids,
                "expected_rank_changed": {
                    doc: dense_expected["expected_ranks"].get(doc)
                    != ce_expected["expected_ranks"].get(doc)
                    for doc in expected_docs
                },
                "missing_expected_changed": (
                    dense_expected["expected_missing"]
                    != ce_expected["expected_missing"]
                ),
                "metrics_delta": {
                    metric: ce_metrics[metric] - dense_metrics[metric]
                    for metric in (
                        "hit_at_1",
                        "hit_at_3",
                        "hit_at_5",
                        "recall_at_5",
                        "mrr",
                    )
                },
            },
        }

        output["cases"].append(record)

    # Aggregate retrieval-only diagnostics.
    def avg(condition: str, metric: str) -> float:
        vals = [
            c[condition]["metrics"][metric]
            for c in output["cases"]
        ]
        return sum(vals) / len(vals) if vals else 0.0

    output["summary"] = {
        "dense": {
            metric: avg("dense", metric)
            for metric in (
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
                "recall_at_5",
                "mrr",
            )
        },
        "dense_crossencoder": {
            metric: avg("dense_crossencoder", metric)
            for metric in (
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
                "recall_at_5",
                "mrr",
            )
        },
        "document_set_changed_cases": sum(
            c["comparison"]["document_set_changed"] for c in output["cases"]
        ),
        "order_changed_cases": sum(
            c["comparison"]["order_changed"] for c in output["cases"]
        ),
        "expected_rank_changed_cases": sum(
            any(c["comparison"]["expected_rank_changed"].values())
            for c in output["cases"]
        ),
        "missing_expected_changed_cases": sum(
            c["comparison"]["missing_expected_changed"]
            for c in output["cases"]
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== G3B RETRIEVAL-ONLY SCREENING COMPLETE ===")
    print(f"Cases: {len(SELECTED_CASES)}")
    print(f"Output: {OUTPUT_PATH}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
