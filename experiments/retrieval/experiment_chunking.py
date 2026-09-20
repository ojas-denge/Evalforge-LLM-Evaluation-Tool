"""
EvalForge chunking ablation.

Isolated retrieval experiment:
- Does NOT modify app settings or the production Chroma collection.
- Rebuilds an ephemeral Chroma index for each chunking configuration.
- Keeps embedding model, dense retrieval, final top_k, and evaluation dataset fixed.
- Produces aggregate metrics plus per-case results for later diagnosis.

Usage:
    python scripts/experiment_chunking.py

Optional:
    python scripts/experiment_chunking.py --chunk-sizes 60 120 240
    python scripts/experiment_chunking.py --overlaps 0 15 30 60
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from statistics import mean

import chromadb

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.evaluation.diagnostics import analyze_retrieval
from app.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.models.evaluation import EvaluationResult
from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever


DOCUMENTS_PATH = Path("data/documents")
DATASET_PATH = Path("data/evaluation_cases.json")
DEFAULT_CHUNK_SIZES = [60, 120, 240]
DEFAULT_OVERLAPS = [0, 15, 30, 60]
FINAL_TOP_K = 5


class EphemeralVectorStore:
    """Minimal VectorStore-compatible adapter backed by an isolated Chroma DB."""

    def __init__(self, chunks, embeddings, root: Path, collection_name: str) -> None:
        self.root = root
        self.client = chromadb.PersistentClient(path=str(root))
        self.collection = self.client.create_collection(name=collection_name)

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def all_chunks(self) -> list[dict]:
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": None,
            }
            for chunk_id, document, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]

    def count(self) -> int:
        return self.collection.count()


def load_chunked_corpus(chunk_size: int, chunk_overlap: int):
    chunks = []

    for path in sorted(DOCUMENTS_PATH.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue

        document_id = path.relative_to(DOCUMENTS_PATH).as_posix()
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        chunks.extend(
            chunk_text(
                text=text,
                document_id=document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def evaluate_configuration(
    dataset: EvaluationDataset,
    embedding_service: EmbeddingService,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    chunks = load_chunked_corpus(chunk_size, chunk_overlap)

    temp_root = Path(tempfile.mkdtemp(prefix="evalforge_chunking_"))

    try:
        embeddings = embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )

        vector_store = EphemeralVectorStore(
            chunks=chunks,
            embeddings=embeddings,
            root=temp_root,
            collection_name="documents",
        )

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            reranking_enabled=False,
            hybrid_retrieval_enabled=False,
            lexical_only=False,
        )

        results: list[EvaluationResult] = []

        for index, case in enumerate(dataset.cases, start=1):
            result = retriever.retrieve(
                query=case.question,
                top_k=FINAL_TOP_K,
            )

            retrieved_documents = [
                item.document_id for item in result.results
            ]

            diagnostic = analyze_retrieval(
                case.expected_documents,
                retrieved_documents,
            )

            results.append(
                EvaluationResult(
                    case_id=case.case_id,
                    question=case.question,
                    expected_documents=case.expected_documents,
                    failure_type=diagnostic.failure_type,
                    relevant_documents_found=diagnostic.relevant_documents_found,
                    first_relevant_rank=diagnostic.first_relevant_rank,
                    missing_documents=diagnostic.missing_documents,
                    confounding_documents=diagnostic.confounding_documents,
                    hit_at_1=hit_at_k(case.expected_documents, retrieved_documents, 1),
                    hit_at_3=hit_at_k(case.expected_documents, retrieved_documents, 3),
                    hit_at_5=hit_at_k(case.expected_documents, retrieved_documents, 5),
                    recall_at_1=recall_at_k(case.expected_documents, retrieved_documents, 1),
                    recall_at_3=recall_at_k(case.expected_documents, retrieved_documents, 3),
                    recall_at_5=recall_at_k(case.expected_documents, retrieved_documents, 5),
                    mrr=reciprocal_rank(
                        case.expected_documents,
                        retrieved_documents,
                    ),
                    retrieval_latency_ms=result.latency_ms,
                    retrieved_evidence=[
                        {
                            "rank": item.rank,
                            "chunk_id": item.chunk_id,
                            "document_id": item.document_id,
                            "distance": item.distance,
                            "text": item.text,
                        }
                        for item in result.results
                    ],
                )
            )

            if index % 25 == 0 or index == len(dataset.cases):
                print(
                    f"    {chunk_size}/{chunk_overlap}: "
                    f"{index}/{len(dataset.cases)}"
                )

        failures = {}
        for result in results:
            failures[result.failure_type] = failures.get(
                result.failure_type, 0
            ) + 1

        latencies = [result.retrieval_latency_ms for result in results]
        sorted_latencies = sorted(latencies)
        p95_index = min(
            len(sorted_latencies) - 1,
            max(0, int(0.95 * len(sorted_latencies)) - 1),
        )

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": len(
                {chunk.document_id for chunk in chunks}
            ),
            "chunk_count": len(chunks),
            "mean_chunk_words": mean(
                len(chunk.text.split()) for chunk in chunks
            ),
            "final_top_k": FINAL_TOP_K,
            "retrieval_mode": retriever.mode,
            "embedding_model": get_settings().embedding_model,
            "metrics": {
                "hit_at_1": mean(r.hit_at_1 for r in results),
                "hit_at_3": mean(r.hit_at_3 for r in results),
                "hit_at_5": mean(r.hit_at_5 for r in results),
                "recall_at_1": mean(r.recall_at_1 for r in results),
                "recall_at_3": mean(r.recall_at_3 for r in results),
                "recall_at_5": mean(r.recall_at_5 for r in results),
                "mrr": mean(r.mrr for r in results),
                "mean_retrieval_latency_ms": mean(latencies),
                "p95_retrieval_latency_ms": sorted_latencies[p95_index],
            },
            "failure_counts": failures,
            "cases": [result.model_dump() for result in results],
        }

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def compare_cases(baseline: dict, candidate: dict) -> dict:
    baseline_cases = {
        item["case_id"]: item for item in baseline["cases"]
    }
    candidate_cases = {
        item["case_id"]: item for item in candidate["cases"]
    }

    changes = []

    for case_id in baseline_cases:
        base = baseline_cases[case_id]
        cand = candidate_cases[case_id]

        base_docs = [
            item["document_id"] for item in base["retrieved_evidence"]
        ]
        cand_docs = [
            item["document_id"] for item in cand["retrieved_evidence"]
        ]

        if (
            base_docs != cand_docs
            or base["failure_type"] != cand["failure_type"]
        ):
            changes.append(
                {
                    "case_id": case_id,
                    "baseline_failure": base["failure_type"],
                    "candidate_failure": cand["failure_type"],
                    "baseline_documents": base_docs,
                    "candidate_documents": cand_docs,
                    "baseline_mrr": base["mrr"],
                    "candidate_mrr": cand["mrr"],
                }
            )

    return {
        "changed_cases": len(changes),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated EvalForge chunking ablation."
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_CHUNK_SIZES,
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=int,
        default=DEFAULT_OVERLAPS,
    )
    parser.add_argument(
        "--output",
        default="logs/chunking_ablation.json",
    )
    args = parser.parse_args()

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(DOCUMENTS_PATH)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    dataset = EvaluationDataset.load(str(DATASET_PATH))

    settings = get_settings()
    embedding_service = EmbeddingService()

    configurations = []

    # Stage A: size sweep with overlap fixed at 25% of chunk size.
    for size in args.chunk_sizes:
        overlap = size // 4
        if overlap >= size:
            raise ValueError(f"Invalid overlap for chunk size {size}")
        configurations.append((size, overlap, "size_sweep"))

    # Stage B: overlap sweep at the current baseline chunk size.
    for overlap in args.overlaps:
        if overlap >= settings.chunk_size:
            raise ValueError(
                f"Overlap {overlap} must be smaller than baseline "
                f"chunk size {settings.chunk_size}"
            )
        pair = (settings.chunk_size, overlap, "overlap_sweep")
        if pair not in configurations:
            configurations.append(pair)

    print("=" * 80)
    print("EVALFORGE CHUNKING ABLATION")
    print("=" * 80)
    print(f"Dataset: {len(dataset)} cases")
    print(f"Documents: {DOCUMENTS_PATH}")
    print(f"Baseline: chunk_size={settings.chunk_size}, "
          f"chunk_overlap={settings.chunk_overlap}")
    print(f"Final top_k: {FINAL_TOP_K}")
    print(f"Embedding: {settings.embedding_model}")
    print("Retrieval: dense, no reranking")
    print()

    results = []

    for chunk_size, overlap, sweep in configurations:
        print(
            f"[run] {sweep}: "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )

        result = evaluate_configuration(
            dataset=dataset,
            embedding_service=embedding_service,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        result["sweep"] = sweep
        results.append(result)

        print(
            "    "
            f"chunks={result['chunk_count']} | "
            f"MRR={result['metrics']['mrr']:.4f} | "
            f"Hit@1={result['metrics']['hit_at_1']:.4f} | "
            f"Recall@5={result['metrics']['recall_at_5']:.4f} | "
            f"latency={result['metrics']['mean_retrieval_latency_ms']:.2f}ms"
        )
        print()

    baseline = next(
        (
            result
            for result in results
            if result["chunk_size"] == settings.chunk_size
            and result["chunk_overlap"] == settings.chunk_overlap
        ),
        None,
    )

    comparisons = {}
    if baseline is not None:
        for result in results:
            key = f"{result['chunk_size']}/{result['chunk_overlap']}"
            comparisons[key] = compare_cases(
                baseline,
                result,
            )

    payload = {
        "experiment": "chunking_ablation",
        "dataset_size": len(dataset),
        "documents_path": str(DOCUMENTS_PATH),
        "baseline": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "retrieval_mode": "dense",
            "final_top_k": FINAL_TOP_K,
        },
        "results": results,
        "baseline_comparisons": comparisons,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'config':<14}"
        f"{'chunks':>8}"
        f"{'Hit@1':>10}"
        f"{'Recall@5':>11}"
        f"{'MRR':>10}"
        f"{'mean ms':>12}"
    )

    for result in results:
        key = (
            f"{result['chunk_size']}/{result['chunk_overlap']}"
        )
        metrics = result["metrics"]
        print(
            f"{key:<14}"
            f"{result['chunk_count']:>8}"
            f"{metrics['hit_at_1']:>10.4f}"
            f"{metrics['recall_at_5']:>11.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['mean_retrieval_latency_ms']:>12.2f}"
        )

    print()
    print(f"Raw results: {output_path}")


if __name__ == "__main__":
    main()
