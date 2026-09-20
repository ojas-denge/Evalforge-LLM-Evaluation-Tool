from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.models.evaluation import RetrievedEvidence
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


OUTPUT_PATH = Path(
    "logs/generation_reranking_ablation.json"
)

TOP_K = 5
CANDIDATE_K = 5
JUDGE_BATCH_SIZE = 5


def build_generator(settings, tracer):
    return OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=(
            settings.llm_structured_output_mode
        ),
    )


def serialize_evidence(results):
    return [
        {
            "rank": result.rank,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "distance": result.distance,
            "text": result.text,
        }
        for result in results
    ]


def generate_case(
    *,
    case,
    retriever,
    generator,
):
    retrieval = retriever.retrieve(
        query=case.question,
        top_k=TOP_K,
    )

    generation_request = GenerationRequest(
        question=case.question,
        context=retrieval.results,
        model=None,
        temperature=0.0,
    )

    generation = generator.generate(
        generation_request
    )

    return {
        "retrieval_latency_ms": retrieval.latency_ms,
        "generation_latency_ms": generation.latency_ms,
        "answer": generation.answer,
        "input_tokens": generation.usage.input_tokens,
        "output_tokens": generation.usage.output_tokens,
        "total_tokens": generation.usage.total_tokens,
        "estimated_cost_usd": (
            generation.estimated_cost_usd
        ),
        "evidence": serialize_evidence(
            retrieval.results
        ),
    }


def build_judge_case(case, result):
    evidence = [
        RetrievedEvidence(
            rank=item["rank"],
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            distance=item["distance"],
            text=item["text"],
        )
        for item in result["evidence"]
    ]

    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_topics": case.expected_topics,
        "generated_answer": result["answer"] or "",
        "retrieved_evidence": evidence,
    }


def run_condition(
    *,
    name,
    dataset,
    generator,
    retriever,
):
    print(
        f"\n=== {name} ===",
        flush=True,
    )

    results = {}

    for index, case in enumerate(
        dataset.cases,
        start=1,
    ):
        print(
            f"\r[{name}] {index}/{len(dataset)}",
            end="",
            flush=True,
        )

        started = time.perf_counter()

        result = generate_case(
            case=case,
            retriever=retriever,
            generator=generator,
        )

        result["wall_time_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        results[case.case_id] = result

    print(
        f"\r[{name}] {len(dataset)}/{len(dataset)}",
        flush=True,
    )

    return results


def judge_condition(
    *,
    name,
    dataset,
    results,
    answer_judge,
):
    print(
        f"\n=== Judging {name} ===",
        flush=True,
    )

    verdicts = {}

    for batch_start in range(
        0,
        len(dataset.cases),
        JUDGE_BATCH_SIZE,
    ):
        batch = dataset.cases[
            batch_start:
            batch_start + JUDGE_BATCH_SIZE
        ]

        judge_cases = [
            build_judge_case(
                case,
                results[case.case_id],
            )
            for case in batch
        ]

        batch_number = (
            batch_start // JUDGE_BATCH_SIZE
        ) + 1

        total_batches = (
            (
                len(dataset.cases)
                + JUDGE_BATCH_SIZE
                - 1
            )
            // JUDGE_BATCH_SIZE
        )

        print(
            f"\r[judge {name}] "
            f"{batch_number}/{total_batches}",
            end="",
            flush=True,
        )

        batch_verdicts = answer_judge.batch_judge(
            judge_cases
        )

        if len(batch_verdicts) != len(batch):
            raise RuntimeError(
                f"{name}: expected "
                f"{len(batch)} judge verdicts, "
                f"received {len(batch_verdicts)}"
            )

        for case, verdict in zip(
            batch,
            batch_verdicts,
            strict=True,
        ):
            verdicts[case.case_id] = verdict

    print(
        f"\r[judge {name}] "
        f"{total_batches}/{total_batches}",
        flush=True,
    )

    return verdicts


def main():
    settings = get_settings()
    tracer = Tracer()

    dataset = EvaluationDataset.load()

    print(
        f"EvalForge G1 — "
        f"Generation impact of reranking",
        flush=True,
    )
    print(
        f"Dataset: {len(dataset)} cases",
        flush=True,
    )
    print(
        f"Top-K: {TOP_K}",
        flush=True,
    )
    print(
        f"Candidate-K: {CANDIDATE_K}",
        flush=True,
    )

    generator = build_generator(
        settings,
        tracer,
    )

    judge_generator = build_generator(
        settings,
        tracer,
    )

    answer_judge = AnswerJudge(
        generator=judge_generator,
    )

    dense_retriever = Retriever(
        tracer=tracer,
        mode="dense",
        reranking_enabled=False,
        candidate_k=CANDIDATE_K,
    )

    reranked_retriever = Retriever(
        tracer=tracer,
        mode="dense_reranked",
        reranking_enabled=True,
        candidate_k=CANDIDATE_K,
        reranker_candidate_k=CANDIDATE_K,
    )

    experiment_started = time.perf_counter()

    dense_results = run_condition(
        name="DENSE",
        dataset=dataset,
        generator=generator,
        retriever=dense_retriever,
    )

    reranked_results = run_condition(
        name="DENSE+CE",
        dataset=dataset,
        generator=generator,
        retriever=reranked_retriever,
    )

    dense_verdicts = judge_condition(
        name="DENSE",
        dataset=dataset,
        results=dense_results,
        answer_judge=answer_judge,
    )

    reranked_verdicts = judge_condition(
        name="DENSE+CE",
        dataset=dataset,
        results=reranked_results,
        answer_judge=answer_judge,
    )

    cases = []

    for case in dataset.cases:
        case_id = case.case_id

        dense = dense_results[case_id]
        reranked = reranked_results[case_id]

        dense_docs = [
            item["document_id"]
            for item in dense["evidence"]
        ]

        reranked_docs = [
            item["document_id"]
            for item in reranked["evidence"]
        ]

        dense_judge = dense_verdicts[case_id]
        reranked_judge = reranked_verdicts[case_id]

        retrieval_changed = (
            dense_docs != reranked_docs
        )

        dense_correct = (
            dense_judge.result.answer_correct
            if dense_judge.result
            else None
        )

        reranked_correct = (
            reranked_judge.result.answer_correct
            if reranked_judge.result
            else None
        )

        dense_grounded = (
            dense_judge.result.answer_grounded
            if dense_judge.result
            else None
        )

        reranked_grounded = (
            reranked_judge.result.answer_grounded
            if reranked_judge.result
            else None
        )

        if (
            dense_correct is True
            and reranked_correct is True
        ):
            answer_outcome = "BOTH_CORRECT"

        elif (
            dense_correct is False
            and reranked_correct is True
        ):
            answer_outcome = "RERANKING_IMPROVEMENT"

        elif (
            dense_correct is True
            and reranked_correct is False
        ):
            answer_outcome = "RERANKING_REGRESSION"

        elif (
            dense_correct is False
            and reranked_correct is False
        ):
            answer_outcome = "BOTH_INCORRECT"

        else:
            answer_outcome = "UNRESOLVED"

        if retrieval_changed:
            propagation = (
                answer_outcome
            )
        else:
            propagation = "RETRIEVAL_UNCHANGED"

        cases.append(
            {
                "case_id": case_id,
                "question": case.question,
                "retrieval_changed": retrieval_changed,
                "dense_documents": dense_docs,
                "reranked_documents": reranked_docs,
                "dense_answer": dense["answer"],
                "reranked_answer": reranked["answer"],
                "dense_judge": (
                    dense_judge.model_dump(
                        mode="json"
                    )
                ),
                "reranked_judge": (
                    reranked_judge.model_dump(
                        mode="json"
                    )
                ),
                "dense_answer_correct": dense_correct,
                "reranked_answer_correct": (
                    reranked_correct
                ),
                "dense_answer_grounded": (
                    dense_grounded
                ),
                "reranked_answer_grounded": (
                    reranked_grounded
                ),
                "answer_outcome": answer_outcome,
                "propagation": propagation,
                "dense_retrieval_latency_ms": (
                    dense["retrieval_latency_ms"]
                ),
                "reranked_retrieval_latency_ms": (
                    reranked[
                        "retrieval_latency_ms"
                    ]
                ),
                "dense_generation_latency_ms": (
                    dense["generation_latency_ms"]
                ),
                "reranked_generation_latency_ms": (
                    reranked[
                        "generation_latency_ms"
                    ]
                ),
                "dense_input_tokens": (
                    dense["input_tokens"]
                ),
                "reranked_input_tokens": (
                    reranked["input_tokens"]
                ),
                "dense_output_tokens": (
                    dense["output_tokens"]
                ),
                "reranked_output_tokens": (
                    reranked["output_tokens"]
                ),
                "dense_cost_usd": (
                    dense["estimated_cost_usd"]
                ),
                "reranked_cost_usd": (
                    reranked[
                        "estimated_cost_usd"
                    ]
                ),
            }
        )

    experiment = {
        "experiment": (
            "generation_reranking_ablation"
        ),
        "created_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "dataset_size": len(dataset),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "conditions": {
            "A": (
                "dense -> top_k=5 -> LLM"
            ),
            "B": (
                "dense -> candidate_k=5 -> "
                "CrossEncoder -> top_k=5 -> LLM"
            ),
        },
        "generator": {
            "provider": "openai-compatible",
            "model": settings.llm_model,
            "temperature": 0.0,
        },
        "cases": cases,
        "wall_time_ms": (
            time.perf_counter()
            - experiment_started
        ) * 1000.0,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            experiment,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}",
        flush=True,
    )

    tracer.flush()


if __name__ == "__main__":
    main()