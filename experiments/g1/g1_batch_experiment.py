from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


BATCH_SIZE = 5
TOP_K = 5
CANDIDATE_K = 5

OUTPUT_PATH = Path("logs/g1_batch_experiment.json")
CHECKPOINT_PATH = Path("logs/g1_batch_checkpoint.json")

# Gemini free-tier generation quota is currently 15 requests/minute/model.
# G1 intentionally uses normal requests rather than Gemini Batch API, so a
# 429 must be treated as a transient quota event and retried after cooldown.
MAX_429_RETRIES = 3
RATE_LIMIT_COOLDOWN_SECONDS = 65


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


SYSTEM_PROMPT = """You are answering multiple independent evaluation cases.

For each case:
- Answer the question using ONLY that case's retrieved context.
- Never use context belonging to another case.
- Preserve the case_id exactly.
- Return exactly one answer for every case.
- Do not omit, duplicate, rename, or reorder case IDs.

Return only the requested structured output.
"""


def retrieve_cases(cases, retriever):
    results = {}

    for case in cases:
        results[case.case_id] = retriever.retrieve(
            query=case.question,
            top_k=TOP_K,
        )

    return results


def serialize_retrieval(retrieval):
    return {
        "query": retrieval.query,
        "latency_ms": retrieval.latency_ms,
        "results": [
            {
                "rank": doc.rank,
                "chunk_id": doc.chunk_id,
                "document_id": doc.document_id,
                "text": doc.text,
                "distance": doc.distance,
            }
            for doc in retrieval.results
        ],
    }


def serialize_generation(result):
    return {
        "structured_output": result.structured_output,
        "model": result.model,
        "provider": result.provider,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "estimated_cost_usd": result.estimated_cost_usd,
        "latency_ms": result.latency_ms,
        "finish_reason": result.finish_reason,
        "structured_output_status": result.structured_output_status,
    }


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
        raise RuntimeError("Batch output has no valid 'answers' list.")

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
                f"Missing/invalid answer for {item.get('case_id')}"
            )


def generate_with_rate_limit_retry(generator, request, label):
    """Generate one request, recovering from Gemini per-minute 429s.

    The experiment is checkpointed at batch boundaries, so a transient quota
    failure should not terminate the entire run. We deliberately keep the
    same model/provider/request payload and only add cooldown + retry logic.
    """

    for attempt in range(MAX_429_RETRIES + 1):
        try:
            return generator.generate(request)

        except httpx.HTTPStatusError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            if status_code != 429:
                raise

            if attempt >= MAX_429_RETRIES:
                print(
                    f"[{label}] 429 quota limit persists after "
                    f"{MAX_429_RETRIES} retries; aborting.",
                    flush=True,
                )
                raise

            retry_number = attempt + 1

            # Gemini may expose Retry-After, but the free-tier quota response
            # does not always provide it. Use it when valid; otherwise use a
            # conservative 65-second cooldown to cross the RPM window.
            retry_after = None

            if exc.response is not None:
                header_value = exc.response.headers.get("retry-after")
                if header_value:
                    try:
                        retry_after = float(header_value)
                    except ValueError:
                        retry_after = None

            cooldown = max(
                RATE_LIMIT_COOLDOWN_SECONDS,
                retry_after or 0,
            )

            print(
                f"[{label}] 429 rate/quota limit; "
                f"retrying ({retry_number}/{MAX_429_RETRIES}) "
                f"after {cooldown:.0f}s cooldown...",
                flush=True,
            )

            time.sleep(cooldown)

    raise RuntimeError(
        f"{label}: unreachable rate-limit retry state."
    )


def run_condition(name, cases, retriever, generator):
    print(
        f"\n[{name}] retrieving {len(cases)} cases...",
        flush=True,
    )

    retrievals = retrieve_cases(cases, retriever)

    print(
        f"[{name}] generating ONE batch request...",
        flush=True,
    )

    request = GenerationRequest(
        question=build_batch_input(cases, retrievals),
        context=[],
        temperature=0.0,
        system_prompt=SYSTEM_PROMPT,
        response_schema=BATCH_SCHEMA,
    )

    result = generate_with_rate_limit_retry(
        generator,
        request,
        name,
    )

    if result.structured_output is None:
        raise RuntimeError(
            f"{name}: no structured output returned."
        )

    validate_batch(cases, result.structured_output)

    print(
        f"[{name}] PASS — {len(cases)}/{len(cases)} answers",
        flush=True,
    )

    print(
        f"[{name}] tokens={result.usage.total_tokens}, "
        f"latency={result.latency_ms:.2f} ms",
        flush=True,
    )

    return {
        "generation": serialize_generation(result),
        "retrievals": {
            case_id: serialize_retrieval(retrieval)
            for case_id, retrieval in retrievals.items()
        },
    }


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {
            "completed_batches": [],
            "batches": {},
        }

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid G1 checkpoint format.")

    checkpoint.setdefault("completed_batches", [])
    checkpoint.setdefault("batches", {})

    return checkpoint


def save_checkpoint(completed_batches, batches):
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": "g1_batch_experiment",
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "completed_batches": sorted(
            completed_batches,
            key=int,
        ),
        "batches": batches,
    }

    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    temp_path.replace(CHECKPOINT_PATH)


def main():
    settings = get_settings()
    tracer = Tracer()

    dataset = EvaluationDataset.load()
    cases = dataset.cases

    if not cases:
        raise RuntimeError("Evaluation dataset is empty.")

    batches = [
        cases[i : i + BATCH_SIZE]
        for i in range(0, len(cases), BATCH_SIZE)
    ]

    generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=settings.llm_structured_output_mode,
    )

    dense = Retriever(
        reranking_enabled=False,
        hybrid_retrieval_enabled=False,
        candidate_k=CANDIDATE_K,
        tracer=tracer,
    )

    crossencoder = Retriever(
        reranking_enabled=True,
        hybrid_retrieval_enabled=False,
        candidate_k=CANDIDATE_K,
        reranker_candidate_k=CANDIDATE_K,
        tracer=tracer,
    )

    print(
        "\nEvalForge G1 — Full 75-case batch experiment",
        flush=True,
    )
    print(f"Cases: {len(cases)}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Batches: {len(batches)}", flush=True)
    print(f"Top-K: {TOP_K}", flush=True)
    print(f"Candidate-K: {CANDIDATE_K}", flush=True)
    print(
        f"Generation requests: {len(batches) * 2}",
        flush=True,
    )

    checkpoint = load_checkpoint()

    completed_batches = set(
        checkpoint.get("completed_batches", [])
    )

    batch_results = checkpoint.get("batches", {})

    if completed_batches:
        print(
            f"Checkpoint found: "
            f"{len(completed_batches)}/{len(batches)} "
            f"batches completed.",
            flush=True,
        )

    for batch_index, batch in enumerate(batches):
        batch_key = str(batch_index)

        if batch_key in completed_batches:
            print(
                f"\n[BATCH {batch_index + 1}/{len(batches)}] "
                "SKIP — already checkpointed",
                flush=True,
            )
            continue

        case_ids = [
            case.case_id
            for case in batch
        ]

        print(
            "\n" + "=" * 70,
            flush=True,
        )
        print(
            f"BATCH {batch_index + 1}/{len(batches)}",
            flush=True,
        )
        print(
            f"Cases: {case_ids}",
            flush=True,
        )
        print(
            "=" * 70,
            flush=True,
        )

        dense_result = run_condition(
            "DENSE",
            batch,
            dense,
            generator,
        )

        ce_result = run_condition(
            "DENSE+CE",
            batch,
            crossencoder,
            generator,
        )

        batch_results[batch_key] = {
            "batch_index": batch_index,
            "case_ids": case_ids,
            "dense": dense_result,
            "dense_crossencoder": ce_result,
        }

        completed_batches.add(batch_key)

        save_checkpoint(
            completed_batches,
            batch_results,
        )

        print(
            f"[BATCH {batch_index + 1}/{len(batches)}] "
            "CHECKPOINT SAVED",
            flush=True,
        )

    expected_batch_keys = {
        str(i)
        for i in range(len(batches))
    }

    if completed_batches != expected_batch_keys:
        missing = sorted(
            expected_batch_keys - completed_batches,
            key=int,
        )

        raise RuntimeError(
            "Experiment did not complete.\n"
            f"Missing batches: {missing}\n"
            "Checkpoint preserved for resume."
        )

    output = {
        "experiment": "g1_batch_experiment",
        "dataset_size": len(cases),
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "generation_request_count": len(batches) * 2,
        "batches": {
            key: batch_results[key]
            for key in sorted(
                batch_results,
                key=int,
            )
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    tracer.flush()

    print(
        "\n" + "=" * 70,
        flush=True,
    )
    print(
        "G1 COMPLETE",
        flush=True,
    )
    print(
        f"Saved: {OUTPUT_PATH}",
        flush=True,
    )
    print(
        f"Cases: {len(cases)}",
        flush=True,
    )
    print(
        f"Batches: {len(batches)}",
        flush=True,
    )
    print(
        f"Generation requests: {len(batches) * 2}",
        flush=True,
    )
    print(
        "=" * 70,
        flush=True,
    )


if __name__ == "__main__":
    main()
