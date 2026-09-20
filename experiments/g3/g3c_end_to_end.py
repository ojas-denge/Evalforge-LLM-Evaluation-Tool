from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.generation.base import GenerationRequest
from app.generation.factory import create_generator
from app.core.config import Settings
from app.models.evaluation import RetrievedEvidence

CASE_IDS = [
    "cross_003",
    "routing_006",
    "cross_001",
    "evaluation_strategy_003",
    "deployment_004",
    "embedding_004",
    "context_004",
    "cost_005",
]

BATCH_SIZE = 4
MAX_RETRIES = 3
COOLDOWN_SECONDS = 65

RETRIEVAL_PATH = Path("logs/g3c_retrieval_checkpoint.json")
OUTPUT_PATH = Path("logs/g3c_end_to_end.json")
CHECKPOINT_PATH = Path("logs/g3c_end_to_end_checkpoint.json")

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


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    ) or any(
        token in text
        for token in ("429", "rate limit", "quota", "resource exhausted")
    )


def build_context(condition: dict[str, Any]) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            rank=d["rank"],
            chunk_id=d["chunk_id"],
            document_id=d["document_id"],
            distance=d.get("distance"),
            text=d["text"],
        )
        for item in condition["retrieval"]["results"]
    ]


def build_batch_prompt(cases: list[dict[str, Any]], condition: str) -> str:
    blocks = []
    for case in cases:
        blocks.append(
            f"""CASE_ID: {case['case_id']}
QUESTION:
{case['question']}

RETRIEVED CONTEXT ({condition}):
"""
            + "\n".join(
                f"[rank={d['rank']} document={d['document_id']}]"
                for d in case[condition]["retrieval"]["results"]
            )
        )
    return (
        "Answer each case independently. Return exactly one answer object "
        "for every CASE_ID, preserving input order. Use only the supplied "
        "retrieved context. Do not invent missing evidence.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def generate_batch(generator, cases, condition):
    prompt = build_batch_prompt(cases, condition)
    request = GenerationRequest(
        question=prompt,
        context=[],
        temperature=0.0,
        system_prompt=(
            "You are the EvalForge evaluation generator. "
            "Answer each case independently and return strict JSON."
        ),
        response_schema=ANSWER_SCHEMA,
        metadata={"experiment": "g3c_end_to_end", "condition": condition},
    )

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generator.generate(request)
            payload = result.structured_output
            if isinstance(payload, str):
                payload = json.loads(payload)

            answers = payload["answers"]
            ids = [item["case_id"] for item in answers]
            expected = [case["case_id"] for case in cases]

            if ids != expected:
                raise ValueError(
                    f"Batch integrity failure: expected {expected}, got {ids}"
                )

            return {
                "answers": answers,
                "generation": json_safe(result),
            }
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES or not retryable(exc):
                raise
            print(f"[generation] {condition}: retryable error; retrying ({attempt}/{MAX_RETRIES - 1})")
            time.sleep(COOLDOWN_SECONDS)

    raise last_exc


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {
        "dense": {},
        "dense_crossencoder": {},
        "judged": {"dense": {}, "dense_crossencoder": {}},
    }


def save_checkpoint(state):
    CHECKPOINT_PATH.write_text(
        json.dumps(json_safe(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    retrieval = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))
    cases = retrieval["cases"]
    by_id = {case["case_id"]: case for case in cases}
    cases = [by_id[cid] for cid in CASE_IDS]

    settings = Settings()
    generator = create_generator(settings)
    judge = AnswerJudge(generator=generator)

    state = load_checkpoint()

    batches = [
        cases[i:i + BATCH_SIZE]
        for i in range(0, len(cases), BATCH_SIZE)
    ]

    for condition in ("dense", "dense_crossencoder"):
        for index, batch in enumerate(batches, start=1):
            if all(
                case["case_id"] in state[condition]
                for case in batch
            ):
                continue

            print(
                f"[generation] {condition} batch {index}/{len(batches)}: "
                f"{[c['case_id'] for c in batch]}"
            )
            state[condition].update(
                {
                    case["case_id"]: result
                    for case, result in zip(
                        batch,
                        generate_batch(generator, batch, condition)["answers"],
                    )
                }
            )
            save_checkpoint(state)

    # Judge independently by condition, preserving the same case batches.
    dataset = EvaluationDataset.load()
    dataset_by_id = {case.case_id: case for case in dataset.cases}

    for condition in ("dense", "dense_crossencoder"):
        for index, batch in enumerate(batches, start=1):
            ids = [case["case_id"] for case in batch]
            if all(cid in state["judged"][condition] for cid in ids):
                continue

            judge_cases = []
            for cid in ids:
                source = by_id[cid]
                eval_case = dataset_by_id[cid]
                judge_cases.append({
                    "case_id": cid,
                    "question": eval_case.question,
                    "expected_answer": eval_case.expected_answer,
                    "expected_topics": eval_case.expected_topics,
                    "generated_answer": state[condition][cid]["answer"],
                    "retrieved_evidence": [
                        RetrievedEvidence(
                            rank=d["rank"],
                            chunk_id=d["chunk_id"],
                            document_id=d["document_id"],
                            distance=d.get("distance"),
                            text=d["text"],
                        )
                        for d in source[condition]["retrieval"]["results"]
                    ],
                })

            print(f"[judge] {condition} batch {index}/{len(batches)}: {ids}")
            verdicts = judge.batch_judge(judge_cases)

            for cid, verdict in zip(ids, verdicts):
                state["judged"][condition][cid] = json_safe(verdict)

            save_checkpoint(state)

    output = {
        "experiment": "g3c_end_to_end",
        "version": "0.1",
        "dataset_size": len(CASE_IDS),
        "batch_size": BATCH_SIZE,
        "generation_requests": len(batches) * 2,
        "judge_requests": len(batches) * 2,
        "conditions": {
            "dense": state["dense"],
            "dense_crossencoder": state["dense_crossencoder"],
        },
        "judged": state["judged"],
    }

    correct = {"dense": 0, "dense_crossencoder": 0}
    grounded = {"dense": 0, "dense_crossencoder": 0}

    for condition in correct:
        for cid in CASE_IDS:
            verdict = state["judged"][condition][cid]
            result = verdict.get("result") or {}
            if result.get("answer_correct"):
                correct[condition] += 1
            if result.get("answer_grounded"):
                grounded[condition] += 1

    unchanged = improvements = regressions = 0
    for cid in CASE_IDS:
        d = bool(
            (state["judged"]["dense"][cid].get("result") or {}).get("answer_correct")
        )
        c = bool(
            (state["judged"]["dense_crossencoder"][cid].get("result") or {}).get("answer_correct")
        )
        if d == c:
            unchanged += 1
        elif not d and c:
            improvements += 1
        else:
            regressions += 1

    output["summary"] = {
        "dense_correct": correct["dense"],
        "crossencoder_correct": correct["dense_crossencoder"],
        "dense_grounded": grounded["dense"],
        "crossencoder_grounded": grounded["dense_crossencoder"],
        "propagation_outcomes": {
            "ANSWER_IMPROVEMENT": improvements,
            "ANSWER_UNCHANGED": unchanged,
            "ANSWER_REGRESSION": regressions,
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(json_safe(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    CHECKPOINT_PATH.unlink(missing_ok=True)

    print("\nG3C end-to-end complete")
    print(json.dumps(output["summary"], indent=2))
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
