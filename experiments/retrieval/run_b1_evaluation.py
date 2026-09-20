import argparse
import json
from collections import Counter

from app.evaluation import EvaluationDataset
from app.evaluation.runner import Evaluator
from app.retrieval.retriever import Retriever


MODES = {
    "dense": {
        "hybrid_retrieval_enabled": False,
        "reranking_enabled": False,
        "lexical_only": False,
    },
    "bm25": {
        "hybrid_retrieval_enabled": False,
        "reranking_enabled": False,
        "lexical_only": True,
    },
    "hybrid": {
        "hybrid_retrieval_enabled": True,
        "reranking_enabled": False,
        "lexical_only": False,
    },
    "dense_reranked": {
        "hybrid_retrieval_enabled": False,
        "reranking_enabled": True,
        "lexical_only": False,
    },
    "hybrid_reranked": {
        "hybrid_retrieval_enabled": True,
        "reranking_enabled": True,
        "lexical_only": False,
    },
}


def evaluate_mode(
    dataset: EvaluationDataset,
    mode: str,
    candidate_k: int | None = None,
) -> dict:
    settings = MODES[mode]

    retriever = Retriever(
        hybrid_retrieval_enabled=settings["hybrid_retrieval_enabled"],
        reranking_enabled=settings["reranking_enabled"],
        lexical_only=settings["lexical_only"],
        candidate_k=candidate_k,
    )

    evaluator = Evaluator(retriever=retriever)
    run = evaluator.evaluate_dataset(dataset)

    diagnostics = Counter(
        result.failure_type
        for result in run.results
    )

    return {
        "dataset_size": run.dataset_size,
        "mode": retriever.mode,
        "candidate_k": candidate_k,
        "hit_at_1": run.mean_hit_at_1,
        "hit_at_3": run.mean_hit_at_3,
        "hit_at_5": run.mean_hit_at_5,
        "recall_at_1": run.mean_recall_at_1,
        "recall_at_3": run.mean_recall_at_3,
        "recall_at_5": run.mean_recall_at_5,
        "mrr": run.mean_mrr,
        "mean_retrieval_latency_ms": run.mean_retrieval_latency_ms,
        "diagnostics": dict(diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EvalForge B1 retrieval evaluation."
    )

    parser.add_argument(
        "--mode",
        choices=[*MODES, "all"],
        default="dense",
        help="Retrieval mode to evaluate.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate pool size before final ranking/reranking.",
    )

    args = parser.parse_args()

    dataset = EvaluationDataset.load()

    if args.mode == "all":
        results = [
            evaluate_mode(
                dataset=dataset,
                mode=mode,
                candidate_k=args.candidate_k,
            )
            for mode in MODES
        ]
        print(json.dumps(results, default=dict))
        return

    result = evaluate_mode(
        dataset=dataset,
        mode=args.mode,
        candidate_k=args.candidate_k,
    )

    print(json.dumps(result, default=dict))


if __name__ == "__main__":
    main()
