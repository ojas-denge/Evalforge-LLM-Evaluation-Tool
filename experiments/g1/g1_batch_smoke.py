from __future__ import annotations

import json

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


BATCH_SIZE = 5

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["case_id", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are answering multiple independent questions.

For each case:
- answer the question using ONLY that case's retrieved context;
- do not use context belonging to another case;
- preserve the case_id exactly;
- provide one answer per case.

Return only the requested structured output.
"""


def retrieve_cases(cases, retriever):
    results = {}

    for case in cases:
        retrieval = retriever.retrieve(
            query=case.question,
            top_k=5,
        )

        results[case.case_id] = retrieval

    return results


def build_batch_input(cases, retrievals):
    blocks = []

    for case in cases:
        retrieval = retrievals[case.case_id]

        context = "\n\n".join(
            (
                f"[rank={doc.rank} | "
                f"document={doc.document_id} | "
                f"chunk={doc.chunk_id}]\n"
                f"{doc.text}"
            )
            for doc in retrieval.results
        )

        blocks.append(
            f"""CASE ID: {case.case_id}

QUESTION:
{case.question}

RETRIEVED CONTEXT:
{context}
"""
        )

    return "\n\n====================\n\n".join(blocks)


def validate_batch(cases, output):
    if not isinstance(output, dict):
        raise RuntimeError("Batch output is not an object.")

    answers = output.get("answers")

    if not isinstance(answers, list):
        raise RuntimeError("Batch output has no answers list.")

    expected_ids = [case.case_id for case in cases]
    returned_ids = [item.get("case_id") for item in answers]

    if len(returned_ids) != len(expected_ids):
        raise RuntimeError(
            f"Expected {len(expected_ids)} answers; "
            f"received {len(returned_ids)}."
        )

    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError(
            f"Duplicate case IDs returned: {returned_ids}"
        )

    if set(returned_ids) != set(expected_ids):
        raise RuntimeError(
            "Case ID mismatch.\n"
            f"Expected: {expected_ids}\n"
            f"Returned: {returned_ids}"
        )

    if returned_ids != expected_ids:
        raise RuntimeError(
            "Case ordering mismatch.\n"
            f"Expected: {expected_ids}\n"
            f"Returned: {returned_ids}"
        )

    for item in answers:
        if not isinstance(item.get("answer"), str):
            raise RuntimeError(
                f"Missing/invalid answer for {item['case_id']}"
            )


def run_condition(name, cases, retriever, generator):
    print(f"\n[{name}] retrieving 5 cases...", flush=True)

    retrievals = retrieve_cases(
        cases,
        retriever,
    )

    print(
        f"[{name}] generating ONE batch request...",
        flush=True,
    )

    request = GenerationRequest(
        question=build_batch_input(
            cases,
            retrievals,
        ),
        context=[],
        temperature=0.0,
        system_prompt=SYSTEM_PROMPT,
        response_schema=BATCH_SCHEMA,
    )

    result = generator.generate(request)

    if result.structured_output is None:
        raise RuntimeError(
            f"{name}: no structured output returned."
        )

    validate_batch(
        cases,
        result.structured_output,
    )

    print(
        f"[{name}] PASS — "
        f"{len(cases)}/{len(cases)} answers",
        flush=True,
    )

    print(
        f"[{name}] tokens="
        f"{result.usage.total_tokens}, "
        f"latency={result.latency_ms:.2f} ms",
        flush=True,
    )

    return {
        "generation": result.structured_output,
        "retrievals": retrievals,
    }


def main():
    settings = get_settings()
    tracer = Tracer()

    dataset = EvaluationDataset.load()
    cases = dataset.cases[:BATCH_SIZE]

    generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=(
            settings.llm_structured_output_mode
        ),
    )

    dense = Retriever(
        reranking_enabled=False,
        hybrid_retrieval_enabled=False,
        candidate_k=5,
        tracer=tracer,
    )

    crossencoder = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=5,
        reranker_candidate_k=5,
        tracer=tracer,
    )

    print("EvalForge G1 — 5-case batch smoke test")
    print(
        "Cases:",
        [case.case_id for case in cases],
        flush=True,
    )

    dense_result = run_condition(
        "DENSE",
        cases,
        dense,
        generator,
    )

    ce_result = run_condition(
        "DENSE+CE",
        cases,
        crossencoder,
        generator,
    )

    print("\nRetrieval comparison", flush=True)

    for case in cases:
        case_id = case.case_id

        dense_docs = [
            doc.document_id
            for doc in dense_result["retrievals"][
                case_id
            ].results
        ]

        ce_docs = [
            doc.document_id
            for doc in ce_result["retrievals"][
                case_id
            ].results
        ]

        print(
            f"{case_id}: "
            f"{'CHANGED' if dense_docs != ce_docs else 'UNCHANGED'}",
            flush=True,
        )

    output = {
        "experiment": "g1_batch_smoke",
        "batch_size": BATCH_SIZE,
        "case_ids": [case.case_id for case in cases],
        "dense": dense_result["generation"],
        "dense_crossencoder": ce_result["generation"],
    }

    path = "logs/g1_batch_smoke.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"\nSaved: {path}", flush=True)

    tracer.flush()


if __name__ == "__main__":
    main()