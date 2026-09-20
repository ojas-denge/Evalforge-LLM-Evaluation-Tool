"""
EvalForge G3B — End-to-End Retrieval Ranking Propagation Experiment

Uses ONLY the selected retrieval-screening artifact as the retrieval source:
    logs/g3b_retrieval_screening.json

Conditions:
    A) Dense top-5 context
    B) Dense + CrossEncoder top-5 context

Selected 10 cases:
    cross_003
    routing_006
    evaluation_strategy_003
    golden_004
    context_004
    deployment_004
    embedding_004
    retrieval_004
    validation_005
    reliability_002

Generation:
    2 batches x 5 cases x 2 conditions = 4 LLM requests

Judge:
    2 batches x 5 cases x 2 conditions = 4 LLM requests

Strict batch integrity validation.
Checkpoint after each completed condition/batch.
No production/default changes.

Outputs:
    logs/g3b_end_to_end.json
    logs/g3b_end_to_end_checkpoint.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.generation.base import GenerationRequest
from app.generation.factory import create_generator
from app.models.evaluation import RetrievedEvidence
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


SCREENING_PATH = Path("logs/g3b_retrieval_screening.json")
OUTPUT_PATH = Path("logs/g3b_end_to_end.json")
CHECKPOINT_PATH = Path("logs/g3b_end_to_end_checkpoint.json")

CASE_IDS = [
    "cross_003",
    "routing_006",
    "evaluation_strategy_003",
    "golden_004",
    "context_004",
    "deployment_004",
    "embedding_004",
    "retrieval_004",
    "validation_005",
    "reliability_002",
]

BATCH_SIZE = 5
MAX_RETRIES = 3
RETRY_COOLDOWN_SECONDS = 65

ANSWER_SCHEMA = {
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_retrieved_evidence(case_record: dict[str, Any], condition: str) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            rank=item["rank"],
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            distance=item["distance"],
            text=item["text"],
        )
        for item in case_record[condition]["results"]
    ]


def build_batch_prompt(
    cases: list[dict[str, Any]],
    condition: str,
) -> str:
    blocks: list[str] = []

    for index, case in enumerate(cases, start=1):
        evidence = case[condition]["results"]

        evidence_text = "\n\n".join(
            (
                f"[Evidence {item['rank']} | {item['document_id']}]\n"
                f"{item['text']}"
            )
            for item in evidence
        )

        blocks.append(
            f"CASE {index}\n"
            f"case_id: {case['case_id']}\n"
            f"question: {case['question']}\n\n"
            f"Retrieved evidence:\n{evidence_text}"
        )

    return (
        "Answer each independent case using only its question and retrieved evidence. "
        "Do not merge cases. Do not invent evidence that is not present. "
        "Return exactly one answer for every supplied case, preserving the supplied case order. "
        "The answer should directly address the question and synthesize the retrieved evidence "
        "when multiple documents are relevant.\n\n"
        + "\n\n==========\n\n".join(blocks)
    )


SYSTEM_PROMPT = (
    "You are the answer-generation component of EvalForge. "
    "For each case, answer the user's question using the retrieved context supplied for that case. "
    "Remain grounded in the supplied evidence. "
    "When multiple retrieved documents are relevant, synthesize them rather than discussing them separately. "
    "Return the required JSON schema exactly."
)


def validate_answers(
    raw: Any,
    expected_case_ids: list[str],
) -> list[dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("Structured output is not an object")

    answers = raw.get("answers")
    if not isinstance(answers, list):
        raise ValueError("Structured output missing answers list")

    if len(answers) != len(expected_case_ids):
        raise ValueError(
            f"Expected {len(expected_case_ids)} answers, got {len(answers)}"
        )

    ids = [item.get("case_id") for item in answers]
    if ids != expected_case_ids:
        raise ValueError(
            f"Case ID/order mismatch. Expected {expected_case_ids}, got {ids}"
        )

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")

    for item in answers:
        if not isinstance(item.get("answer"), str):
            raise ValueError(f"Invalid answer for {item.get('case_id')}")

    return answers


def retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "quota" in text:
        return True
    if isinstance(exc, httpx.ReadTimeout):
        return True
    return False


def generate_batch(
    generator: Any,
    cases: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    expected_ids = [c["case_id"] for c in cases]
    prompt = build_batch_prompt(cases, condition)

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()

        try:
            result = generator.generate(
                GenerationRequest(
                    question=prompt,
                    context=[],
                    temperature=0.0,
                    system_prompt=SYSTEM_PROMPT,
                    response_schema=ANSWER_SCHEMA,
                    metadata={
                        "experiment": "g3b_end_to_end",
                        "condition": condition,
                        "case_ids": expected_ids,
                    },
                )
            )

            structured = result.structured_output
            answers = validate_answers(structured, expected_ids)

            return {
                "case_ids": expected_ids,
                "answers": answers,
                "generation": {
                    "model": result.model,
                    "provider": result.provider,
                    "usage": (
                        result.usage.model_dump()
                        if hasattr(result.usage, "model_dump")
                        else result.usage
                    ),
                    "estimated_cost_usd": result.estimated_cost_usd,
                    "latency_ms": result.latency_ms,
                    "finish_reason": result.finish_reason,
                    "structured_output_status": result.structured_output_status,
                },
                "wall_latency_ms": (time.perf_counter() - started) * 1000,
                "attempts": attempt,
            }

        except Exception as exc:
            last_exc = exc

            if attempt >= MAX_RETRIES or not retryable(exc):
                raise

            print(
                f"    retryable generation error; "
                f"retrying ({attempt}/{MAX_RETRIES - 1})"
            )
            time.sleep(RETRY_COOLDOWN_SECONDS)

    raise RuntimeError(f"Generation failed: {last_exc}")

def json_safe(value: Any) -> Any:
    """Recursively convert Pydantic/model objects into JSON-safe values."""
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)

def make_judge_case(
    case: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "expected_answer": case["_expected_answer"],
        "expected_topics": case["_expected_topics"],
        "generated_answer": answer,
        "retrieved_evidence": [
            RetrievedEvidence(
                rank=item["rank"],
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                distance=item["distance"],
                text=item["text"],
            )
            for item in case["_judge_evidence"]
        ],
    }


def main() -> None:
    settings = get_settings()
    tracer = Tracer()
    generator = create_generator(settings, tracer=tracer)

    dataset = EvaluationDataset.load()
    dataset_cases = {case.case_id: case for case in dataset.cases}

    screening = load_json(SCREENING_PATH)
    screening_cases = {
        case["case_id"]: case for case in screening["cases"]
    }

    missing = [cid for cid in CASE_IDS if cid not in screening_cases]
    if missing:
        raise RuntimeError(f"Cases missing from screening artifact: {missing}")

    missing_dataset = [cid for cid in CASE_IDS if cid not in dataset_cases]
    if missing_dataset:
        raise RuntimeError(f"Cases missing from evaluation dataset: {missing_dataset}")

    all_batches = batches(CASE_IDS, BATCH_SIZE)

    checkpoint: dict[str, Any] = {
        "generation": {
            "dense": {},
            "dense_crossencoder": {},
        },
        "judging": {
            "dense": {},
            "dense_crossencoder": {},
        },
    }

    if CHECKPOINT_PATH.exists():
        checkpoint = load_json(CHECKPOINT_PATH)
        print(f"Resuming from {CHECKPOINT_PATH}")

    generation_output: dict[str, Any] = {
        "dense": {},
        "dense_crossencoder": {},
    }

    for condition in ("dense", "dense_crossencoder"):
        for batch_index, batch_ids in enumerate(all_batches):
            key = str(batch_index)

            if key in checkpoint["generation"][condition]:
                generation_output[condition][key] = checkpoint["generation"][condition][key]
                print(f"[generation] {condition} batch {batch_index + 1}/{len(all_batches)} — checkpoint")
                continue

            case_records = [screening_cases[cid] for cid in batch_ids]

            print(
                f"[generation] {condition} "
                f"batch {batch_index + 1}/{len(all_batches)}: "
                f"{batch_ids}"
            )

            result = generate_batch(generator, case_records, condition)

            generation_output[condition][key] = result
            checkpoint["generation"][condition][key] = result
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.write_text(
                json.dumps(json_safe(checkpoint), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Prepare condition-specific judge inputs.
    judge_inputs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "dense": {},
        "dense_crossencoder": {},
    }

    for condition in ("dense", "dense_crossencoder"):
        for batch_index, batch_ids in enumerate(all_batches):
            answers = {
                item["case_id"]: item["answer"]
                for item in generation_output[condition][str(batch_index)]["answers"]
            }

            batch_cases = []

            for cid in batch_ids:
                source = screening_cases[cid]
                dataset_case = dataset_cases[cid]

                enriched = dict(source)
                enriched["_expected_answer"] = dataset_case.expected_answer
                enriched["_expected_topics"] = dataset_case.expected_topics
                enriched["_judge_evidence"] = source[condition]["results"]

                batch_cases.append(
                    make_judge_case(enriched, answers[cid])
                )

            judge_inputs[condition][str(batch_index)] = batch_cases

    # Rate-limit-safe wrapper around the existing AnswerJudge generator.
    class SafeGenerator:
        def __init__(self, inner: Any):
            self.inner = inner

        def generate(self, request: GenerationRequest) -> Any:
            last_exc: Exception | None = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return self.inner.generate(request)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= MAX_RETRIES or not retryable(exc):
                        raise
                    print(
                        f"    retryable judge error; "
                        f"retrying ({attempt}/{MAX_RETRIES - 1})"
                    )
                    time.sleep(RETRY_COOLDOWN_SECONDS)

            raise RuntimeError(f"Judge failed: {last_exc}")

    judge = AnswerJudge(generator=SafeGenerator(generator))

    judged_output: dict[str, Any] = {
        "dense": {},
        "dense_crossencoder": {},
    }

    for condition in ("dense", "dense_crossencoder"):
        for batch_index, batch_ids in enumerate(all_batches):
            key = str(batch_index)

            if key in checkpoint["judging"][condition]:
                judged_output[condition][key] = checkpoint["judging"][condition][key]
                print(f"[judge] {condition} batch {batch_index + 1}/{len(all_batches)} — checkpoint")
                continue

            print(
                f"[judge] {condition} "
                f"batch {batch_index + 1}/{len(all_batches)}: "
                f"{batch_ids}"
            )

            verdicts = judge.batch_judge(judge_inputs[condition][key])

            serialized = []
            for cid, verdict in zip(batch_ids, verdicts):
                if verdict.result is None:
                    raise RuntimeError(f"Missing judge result for {cid}")

                serialized.append(
                    {
                        "case_id": cid,
                        "status": verdict.status.value
                        if hasattr(verdict.status, "value")
                        else str(verdict.status),
                        "result": verdict.result.model_dump()
                        if hasattr(verdict.result, "model_dump")
                        else verdict.result,
                        "judge_model": verdict.judge_model,
                        "judge_provider": verdict.judge_provider,
                        "judge_latency_ms": verdict.judge_latency_ms,
                        "judge_tokens": verdict.judge_tokens,
                        "judge_cost_usd": verdict.judge_cost_usd,
                        "structured_output_valid": verdict.structured_output_valid,
                        "judge_failure": verdict.judge_failure,
                        "judge_metadata": verdict.judge_metadata,
                    }
                )

            judged_output[condition][key] = {"verdicts": serialized}
            checkpoint["judging"][condition][key] = judged_output[condition][key]

            CHECKPOINT_PATH.write_text(
                json.dumps(json_safe(checkpoint), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Flatten into per-case records and calculate propagation categories.
    by_case: dict[str, dict[str, Any]] = {
        cid: {
            "case_id": cid,
            "question": dataset_cases[cid].question,
        }
        for cid in CASE_IDS
    }

    for condition in ("dense", "dense_crossencoder"):
        for batch_index in range(len(all_batches)):
            key = str(batch_index)
            answers = {
                item["case_id"]: item["answer"]
                for item in generation_output[condition][key]["answers"]
            }
            verdicts = {
                item["case_id"]: item
                for item in judged_output[condition][key]["verdicts"]
            }

            for cid in all_batches[batch_index]:
                by_case[cid][condition] = {
                    "answer": answers[cid],
                    "judge": verdicts[cid],
                }

    for cid in CASE_IDS:
        dense_correct = by_case[cid]["dense"]["judge"]["result"]["answer_correct"]
        ce_correct = by_case[cid]["dense_crossencoder"]["judge"]["result"]["answer_correct"]

        dense_rank = screening_cases[cid]["dense"]["expected"]["expected_ranks"]
        ce_rank = screening_cases[cid]["dense_crossencoder"]["expected"]["expected_ranks"]

        if dense_correct and not ce_correct:
            outcome = "ANSWER_REGRESSION"
        elif not dense_correct and ce_correct:
            outcome = "ANSWER_IMPROVEMENT"
        else:
            outcome = "ANSWER_UNCHANGED"

        by_case[cid]["retrieval"] = {
            "dense_metrics": screening_cases[cid]["dense"]["metrics"],
            "crossencoder_metrics": screening_cases[cid]["dense_crossencoder"]["metrics"],
            "dense_expected_ranks": dense_rank,
            "crossencoder_expected_ranks": ce_rank,
            "document_set_changed": screening_cases[cid]["comparison"]["document_set_changed"],
            "order_changed": screening_cases[cid]["comparison"]["order_changed"],
            "expected_rank_changed": screening_cases[cid]["comparison"]["expected_rank_changed"],
            "metrics_delta": screening_cases[cid]["comparison"]["metrics_delta"],
        }
        by_case[cid]["propagation_outcome"] = outcome

    output = {
        "experiment": "g3b_end_to_end",
        "dataset_size": len(CASE_IDS),
        "batch_size": BATCH_SIZE,
        "generation_requests": 4,
        "judge_requests": 4,
        "cases": list(by_case.values()),
    }

    counts = {
        "ANSWER_IMPROVEMENT": 0,
        "ANSWER_UNCHANGED": 0,
        "ANSWER_REGRESSION": 0,
    }

    for record in output["cases"]:
        counts[record["propagation_outcome"]] += 1

    output["summary"] = {
        "propagation_outcomes": counts,
        "dense_correct": sum(
            r["dense"]["judge"]["result"]["answer_correct"]
            for r in output["cases"]
        ),
        "crossencoder_correct": sum(
            r["dense_crossencoder"]["judge"]["result"]["answer_correct"]
            for r in output["cases"]
        ),
        "dense_grounded": sum(
            r["dense"]["judge"]["result"]["answer_grounded"]
            for r in output["cases"]
        ),
        "crossencoder_grounded": sum(
            r["dense_crossencoder"]["judge"]["result"]["answer_grounded"]
            for r in output["cases"]
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
    json.dumps(json_safe(output), indent=2, ensure_ascii=False),
    encoding="utf-8",
    )

    CHECKPOINT_PATH.unlink(missing_ok=True)

    try:
        tracer.flush()
    except Exception:
        pass

    print("\n=== G3B END-TO-END COMPLETE ===")
    print(f"Output: {OUTPUT_PATH}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
