from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from dataclasses import asdict, is_dataclass
from app.core.config import Settings
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.dataset import EvaluationDataset
from app.generation.factory import create_generator
from app.models.evaluation import RetrievedEvidence
from app.observability.tracing import Tracer


INPUT_PATH = Path("logs/g2_context_ordering.json")
OUTPUT_PATH = Path("logs/g2_judged.json")
CHECKPOINT_PATH = Path("logs/g2_judge_checkpoint.json")

CONDITIONS = (
    "dense",
    "dense_crossencoder",
    "expected_first",
    "oracle",
)

BATCH_SIZE = 5
MAX_RETRIES = 3
RATE_LIMIT_COOLDOWN_SECONDS = 65.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("g2_judge")


class RateLimitSafeGenerator:
    """Wrap a generator so AnswerJudge never falls back into uncontrolled
    individual calls after a batch-level 429.
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def generate(self, request: Any) -> Any:
        for attempt in range(MAX_RETRIES):
            try:
                return self.generator.generate(request)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status != 429 or attempt == MAX_RETRIES - 1:
                    raise

                retry_after = exc.response.headers.get("Retry-After")
                delay = RATE_LIMIT_COOLDOWN_SECONDS
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass

                logger.warning(
                    "429 from judge provider; sleeping %.1fs before retry (%d/%d)",
                    delay,
                    attempt + 1,
                    MAX_RETRIES - 1,
                )
                time.sleep(delay)

        raise RuntimeError("unreachable")


def load_input() -> dict[str, Any]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    batches = raw.get("batches")
    if not isinstance(batches, dict):
        raise RuntimeError("G2 artifact does not contain a 'batches' object")

    flattened_cases: dict[str, dict[str, Any]] = {}

    for batch_key, batch in batches.items():
        case_ids = batch.get("case_ids", [])

        for case_id in case_ids:
            case_entry = {
                "case_id": case_id,
            }

            for condition in CONDITIONS:
                condition_data = batch.get(condition)
                if not isinstance(condition_data, dict):
                    raise RuntimeError(
                        f"{case_id}: missing condition '{condition}' "
                        f"in batch {batch_key}"
                    )

                generation = condition_data.get("generation")
                retrievals = condition_data.get("retrievals")

                if not isinstance(generation, dict):
                    raise RuntimeError(
                        f"{case_id}: invalid generation data "
                        f"for {condition}"
                    )

                if not isinstance(retrievals, dict):
                    raise RuntimeError(
                        f"{case_id}: invalid retrieval data "
                        f"for {condition}"
                    )

                answers = (
                    generation
                    .get("structured_output", {})
                    .get("answers")
                )

                if not isinstance(answers, list):
                    raise RuntimeError(
                        f"{condition}: missing structured answers "
                        f"in batch {batch_key}"
                    )

                matching_answers = [
                    item
                    for item in answers
                    if item.get("case_id") == case_id
                ]

                if len(matching_answers) != 1:
                    raise RuntimeError(
                        f"{case_id}: expected exactly one generated answer "
                        f"for {condition}, found {len(matching_answers)}"
                    )

                if case_id not in retrievals:
                    raise RuntimeError(
                        f"{case_id}: missing retrieval for {condition}"
                    )

                # Generation metadata is batch-level, while the answer
                # itself is case-level.
                case_entry[condition] = {
                    "generation": {
                        "answer": matching_answers[0]["answer"],
                        "model": generation.get("model"),
                        "provider": generation.get("provider"),
                        "usage": generation.get("usage"),
                        "estimated_cost_usd": generation.get(
                            "estimated_cost_usd"
                        ),
                        "latency_ms": generation.get("latency_ms"),
                        "finish_reason": generation.get("finish_reason"),
                        "structured_output_status": generation.get(
                            "structured_output_status"
                        ),
                    },
                    "retrieval": retrievals[case_id],
                }

            if case_id in flattened_cases:
                raise RuntimeError(
                    f"Duplicate case ID across G2 batches: {case_id}"
                )

            flattened_cases[case_id] = case_entry

    cases = list(flattened_cases.values())

    expected_size = raw.get("dataset_size")
    if expected_size is not None and len(cases) != expected_size:
        raise RuntimeError(
            f"G2 artifact declares {expected_size} cases, "
            f"but flattened {len(cases)}"
        )

    return {
        "cases": cases,
        "metadata": {
            "dataset_size": raw.get("dataset_size"),
            "batch_size": raw.get("batch_size"),
            "generation_request_count": raw.get(
                "generation_request_count"
            ),
            "conditions": raw.get("conditions"),
        },
    }


def load_dataset() -> dict[str, Any]:
    dataset = EvaluationDataset.load()
    return {case.case_id: case for case in dataset.cases}


def evidence_from_json(retrieval: dict[str, Any]) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            rank=item["rank"],
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            distance=item.get("distance"),
            text=item["text"],
        )
        for item in retrieval.get("results", [])
    ]


def make_judge_case(
    case_id: str,
    case_ref: dict[str, Any],
    generated: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "question": case_ref.question,
        "expected_answer": case_ref.expected_answer,
        "expected_topics": case_ref.expected_topics,
        "generated_answer": generated["answer"],
        "retrieved_evidence": evidence_from_json(retrieval),
    }


def validate_verdicts(
    verdicts: list[Any],
    expected_ids: list[str],
) -> None:
    if len(verdicts) != len(expected_ids):
        raise RuntimeError(
            f"Judge returned {len(verdicts)} verdicts; expected {len(expected_ids)}"
        )

    for case_id, verdict in zip(expected_ids, verdicts):
        status = getattr(verdict, "status", None)
        result = getattr(verdict, "result", None)
        valid = getattr(verdict, "structured_output_valid", False)

        if getattr(status, "value", None) != "valid":
            raise RuntimeError(f"{case_id}: invalid judge status: {status}")
        if result is None:
            raise RuntimeError(f"{case_id}: missing judge result")
        if not valid:
            raise RuntimeError(f"{case_id}: structured output not valid")


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {"conditions": {}}
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if hasattr(value, "value") and not isinstance(
        value, (str, int, float, bool)
    ):
        return _json_safe(value.value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def save_checkpoint(state: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp = CHECKPOINT_PATH.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(state),
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp.replace(CHECKPOINT_PATH)

def main() -> None:
    payload = load_input()
    dataset = load_dataset()

    cases = payload.get("cases", [])
    if len(cases) != 75:
        raise RuntimeError(f"Expected 75 G2 cases, found {len(cases)}")

    # G2 stores condition outputs under each case. Validate the full generation
    # artifact before spending any judge calls.
    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("Duplicate case IDs in G2 artifact")

    for condition in CONDITIONS:
        for case in cases:
            if condition not in case:
                raise RuntimeError(f"{case['case_id']}: missing condition {condition}")
            generated = case[condition].get("generation")
            retrieval = case[condition].get("retrieval")
            if not generated or not retrieval:
                raise RuntimeError(
                    f"{case['case_id']}: incomplete {condition} generation/retrieval"
                )

    settings = Settings()
    tracer = Tracer()
    generator = create_generator(settings, tracer=tracer)
    safe_generator = RateLimitSafeGenerator(generator)
    judge = AnswerJudge(generator=safe_generator)

    checkpoint = load_checkpoint()
    results: dict[str, Any] = checkpoint.setdefault("conditions", {})

    total_batches = len(cases) // BATCH_SIZE

    for condition in CONDITIONS:
        if condition in results and len(results[condition]) == len(cases):
            logger.info("%s already complete; skipping", condition)
            continue

        condition_results: dict[str, Any] = results.setdefault(condition, {})

        for batch_start in range(0, len(cases), BATCH_SIZE):
            batch = cases[batch_start : batch_start + BATCH_SIZE]
            batch_ids = [case["case_id"] for case in batch]

            if all(case_id in condition_results for case_id in batch_ids):
                continue

            judge_cases = []
            for case in batch:
                case_id = case["case_id"]
                ref = dataset.get(case_id)
                if ref is None:
                    raise RuntimeError(f"{case_id}: missing dataset reference")

                generated = case[condition]["generation"]
                retrieval = case[condition]["retrieval"]

                judge_cases.append(
                    make_judge_case(
                        case_id=case_id,
                        case_ref=ref,
                        generated=generated,
                        retrieval=retrieval,
                    )
                )

            batch_number = batch_start // BATCH_SIZE + 1
            logger.info(
                "Judging %s batch %d/%d (%s)",
                condition,
                batch_number,
                total_batches,
                ", ".join(batch_ids),
            )

            verdicts = judge.batch_judge(judge_cases)
            validate_verdicts(verdicts, batch_ids)

            for case_id, verdict in zip(batch_ids, verdicts):
                result = verdict.result
                condition_results[case_id] = {
                    "answer_correct": result.answer_correct,
                    "answer_grounded": result.answer_grounded,
                    "topics_covered": result.topics_covered,
                    "topics_missing": result.topics_missing,
                    "unsupported_claims": result.unsupported_claims,
                    "reasoning": result.reasoning,
                    "judge_model": verdict.judge_model,
                    "judge_provider": verdict.judge_provider,
                    "judge_latency_ms": verdict.judge_latency_ms,
                    "judge_tokens": (
                        verdict.judge_tokens.model_dump()
                        if hasattr(verdict.judge_tokens, "model_dump")
                        else verdict.judge_tokens
                    ),
                    "judge_cost_usd": verdict.judge_cost_usd,
                    "structured_output_valid": verdict.structured_output_valid,
                    "judge_failure": verdict.judge_failure,
                    "judge_metadata": verdict.judge_metadata,
                }

            save_checkpoint(checkpoint)

    incomplete = [
        condition
        for condition in CONDITIONS
        if len(results.get(condition, {})) != len(cases)
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete conditions: {incomplete}")

    output = {
        "experiment": "g2_context_ordering_judged",
        "source": str(INPUT_PATH),
        "case_count": len(cases),
        "batch_size": BATCH_SIZE,
        "conditions": list(CONDITIONS),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(output),
            f,
            indent=2,
            ensure_ascii=False,
        )
    tmp.replace(OUTPUT_PATH)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    tracer.flush()

    logger.info("G2 judging complete: %d cases × %d conditions", len(cases), len(CONDITIONS))
    logger.info("Output: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
