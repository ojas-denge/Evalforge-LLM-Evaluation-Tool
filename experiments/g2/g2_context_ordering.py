from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.observability.tracing import Tracer


BATCH_SIZE = 5
TOP_K = 5
CONDITIONS = (
    "dense",
    "dense_crossencoder",
    "expected_first",
    "oracle",
)

SOURCE_PATH = Path("logs/g1_batch_experiment.json")
FORENSIC_PATH = Path("logs/dense_vs_crossencoder_forensic.json")

OUTPUT_PATH = Path("logs/g2_context_ordering.json")
CHECKPOINT_PATH = Path("logs/g2_context_ordering_checkpoint.json")

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


def load_sources():
    if not SOURCE_PATH.exists():
        raise RuntimeError(f"Missing source artifact: {SOURCE_PATH}")
    if not FORENSIC_PATH.exists():
        raise RuntimeError(f"Missing forensic artifact: {FORENSIC_PATH}")

    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        g1 = json.load(f)

    with FORENSIC_PATH.open("r", encoding="utf-8") as f:
        forensic = json.load(f)

    forensic_cases = {
        item["case_id"]: item
        for item in forensic["cases"]
    }

    return g1, forensic_cases


def flatten_batches(g1):
    cases = []

    for batch_key in sorted(g1["batches"], key=int):
        batch = g1["batches"][batch_key]
        for case_id in batch["case_ids"]:
            dense_retrieval = batch["dense"]["retrievals"][case_id]
            ce_retrieval = batch["dense_crossencoder"]["retrievals"][case_id]

            cases.append(
                {
                    "case_id": case_id,
                    "query": dense_retrieval["query"],
                    "dense": dense_retrieval,
                    "dense_crossencoder": ce_retrieval,
                }
            )

    return cases


def reorder_results(results, expected_documents, mode):
    """Reorder only the existing top-5 evidence. Never add evidence."""
    items = [dict(item) for item in results]

    if mode == "dense":
        ordered = items

    elif mode == "dense_crossencoder":
        ordered = items

    elif mode == "expected_first":
        if not expected_documents:
            ordered = items
        else:
            target = expected_documents[0]
            matching = [x for x in items if x["document_id"] == target]
            remaining = [x for x in items if x["document_id"] != target]
            ordered = matching + remaining

    elif mode == "oracle":
        expected_set = set(expected_documents)
        relevant = [
            x for x in items
            if x["document_id"] in expected_set
        ]
        remaining = [
            x for x in items
            if x["document_id"] not in expected_set
        ]

        # Preserve the benchmark's expected-document ordering where
        # multiple expected documents exist.
        rank_map = {
            doc_id: index
            for index, doc_id in enumerate(expected_documents)
        }
        relevant.sort(key=lambda x: rank_map[x["document_id"]])
        ordered = relevant + remaining

    else:
        raise ValueError(f"Unknown condition: {mode}")

    # Reassign ranks after the controlled reorder.
    for rank, item in enumerate(ordered[:TOP_K], start=1):
        item["rank"] = rank

    return ordered[:TOP_K]


def build_condition_retrieval(case, forensic_case, condition):
    if condition == "dense":
        source = case["dense"]
    elif condition == "dense_crossencoder":
        source = case["dense_crossencoder"]
    else:
        # Both controlled-ordering conditions use the SAME Dense top-5
        # evidence set. Only the ordering changes.
        source = case["dense"]

    expected_documents = forensic_case["expected_documents"]

    results = reorder_results(
        source["results"],
        expected_documents,
        condition,
    )

    return {
        "query": source["query"],
        "latency_ms": source["latency_ms"],
        "results": results,
    }


def build_batch_input(cases, retrievals):
    blocks = []

    for case in cases:
        retrieval = retrievals[case["case_id"]]

        context = "\n\n".join(
            (
                f"[rank={doc['rank']} | "
                f"document={doc['document_id']} | "
                f"chunk={doc['chunk_id']}]\n"
                f"{doc['text']}"
            )
            for doc in retrieval["results"]
        )

        blocks.append(
            f"""CASE ID: {case["case_id"]}

QUESTION:
{case["query"]}

RETRIEVED CONTEXT:
{context}
"""
        )

    return "\n\n====================\n\n".join(blocks)


def validate_batch(case_ids, output):
    if not isinstance(output, dict):
        raise RuntimeError("Batch output is not an object.")

    answers = output.get("answers")
    if not isinstance(answers, list):
        raise RuntimeError("Batch output has no answers list.")

    returned_ids = [item.get("case_id") for item in answers]

    if len(returned_ids) != len(case_ids):
        raise RuntimeError(
            f"Expected {len(case_ids)} answers; received {len(returned_ids)}."
        )

    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError(f"Duplicate case IDs returned: {returned_ids}")

    if set(returned_ids) != set(case_ids):
        raise RuntimeError(
            f"Case ID mismatch. Expected={case_ids}, returned={returned_ids}"
        )

    if returned_ids != case_ids:
        raise RuntimeError(
            f"Case ordering mismatch. Expected={case_ids}, returned={returned_ids}"
        )

    for item in answers:
        if not isinstance(item.get("answer"), str):
            raise RuntimeError(f"Missing/invalid answer for {item['case_id']}")


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


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {"completed_batches": [], "batches": {}}

    with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("completed_batches", [])
    data.setdefault("batches", {})
    return data


def save_checkpoint(completed_batches, batches):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "g2_context_ordering",
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "conditions": list(CONDITIONS),
        "completed_batches": sorted(completed_batches, key=int),
        "batches": batches,
    }

    temp = CHECKPOINT_PATH.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temp.replace(CHECKPOINT_PATH)


def is_retryable(exc):
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "quota" in text
        or "resource exhausted" in text
        or "timeout" in text
    )


def run_condition(condition, batch, forensic_cases, generator):
    retrievals = {
        case["case_id"]: build_condition_retrieval(
            case,
            forensic_cases[case["case_id"]],
            condition,
        )
        for case in batch
    }

    case_ids = [case["case_id"] for case in batch]

    request = GenerationRequest(
        question=build_batch_input(batch, retrievals),
        context=[],
        temperature=0.0,
        system_prompt=SYSTEM_PROMPT,
        response_schema=BATCH_SCHEMA,
    )

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            result = generator.generate(request)

            if result.structured_output is None:
                raise RuntimeError(
                    f"{condition}: no structured output returned."
                )

            validate_batch(case_ids, result.structured_output)

            print(
                f"  [{condition}] PASS "
                f"{len(case_ids)}/{len(case_ids)} | "
                f"tokens={result.usage.total_tokens} | "
                f"latency={result.latency_ms:.0f} ms",
                flush=True,
            )

            return {
                "generation": serialize_generation(result),
                "retrievals": retrievals,
            }

        except Exception as exc:
            if attempt == max_attempts or not is_retryable(exc):
                raise

            delay = 45
            print(
                f"  [{condition}] retryable error; "
                f"retrying ({attempt}/{max_attempts - 1}) "
                f"after {delay}s: {exc}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable retry state.")


def main():
    g1, forensic_cases = load_sources()
    cases = flatten_batches(g1)

    if len(cases) != 75:
        raise RuntimeError(f"Expected 75 cases; found {len(cases)}.")

    missing = [
        case["case_id"]
        for case in cases
        if case["case_id"] not in forensic_cases
    ]
    if missing:
        raise RuntimeError(
            f"Missing forensic records for {len(missing)} cases: {missing}"
        )

    batches = [
        cases[i:i + BATCH_SIZE]
        for i in range(0, len(cases), BATCH_SIZE)
    ]

    settings = get_settings()
    tracer = Tracer()

    generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=settings.llm_structured_output_mode,
    )

    print("=" * 72)
    print("EvalForge G2 — Context Ordering Experiment")
    print("=" * 72)
    print(f"Cases: {len(cases)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches: {len(batches)}")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print(f"Generation requests: {len(batches) * len(CONDITIONS)}")
    print()
    print("Oracle and Expected-first use the SAME Dense top-5 evidence set.")
    print("No condition adds documents that were not already retrieved.")
    print("=" * 72)

    checkpoint = load_checkpoint()
    completed = set(str(x) for x in checkpoint["completed_batches"])
    batch_results = checkpoint["batches"]

    for batch_index, batch in enumerate(batches):
        key = str(batch_index)

        if key in completed:
            print(
                f"\n[BATCH {batch_index + 1}/{len(batches)}] "
                "SKIP — already checkpointed",
                flush=True,
            )
            continue

        case_ids = [case["case_id"] for case in batch]

        print(
            f"\n[BATCH {batch_index + 1}/{len(batches)}] "
            f"{case_ids}",
            flush=True,
        )

        results = {}

        for condition in CONDITIONS:
            results[condition] = run_condition(
                condition,
                batch,
                forensic_cases,
                generator,
            )

        batch_results[key] = {
            "batch_index": batch_index,
            "case_ids": case_ids,
            **results,
        }

        completed.add(key)
        save_checkpoint(completed, batch_results)

        print(
            f"[BATCH {batch_index + 1}/{len(batches)}] "
            "CHECKPOINT SAVED",
            flush=True,
        )

    expected_keys = {str(i) for i in range(len(batches))}
    if completed != expected_keys:
        missing = sorted(expected_keys - completed, key=int)
        raise RuntimeError(
            f"Experiment incomplete. Missing batches: {missing}. "
            "Checkpoint preserved for resume."
        )

    output = {
        "experiment": "g2_context_ordering",
        "dataset_size": len(cases),
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "conditions": list(CONDITIONS),
        "generation_request_count": len(batches) * len(CONDITIONS),
        "methodology": {
            "dense": "Original Dense top-5 from G1.",
            "dense_crossencoder": "Original Dense+CrossEncoder top-5 from G1.",
            "expected_first": (
                "Same Dense top-5 evidence set; first expected document "
                "moved to rank 1 when present."
            ),
            "oracle": (
                "Same Dense top-5 evidence set; all expected documents "
                "moved to the front in benchmark expected-document order."
            ),
        },
        "batches": {
            key: batch_results[key]
            for key in sorted(batch_results, key=int)
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    tracer.flush()

    print()
    print("=" * 72)
    print("G2 GENERATION COMPLETE")
    print("=" * 72)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Cases: {len(cases)}")
    print(f"Batches: {len(batches)}")
    print(f"Generation requests: {len(batches) * len(CONDITIONS)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
