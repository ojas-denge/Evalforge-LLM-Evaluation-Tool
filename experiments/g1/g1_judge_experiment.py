from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.judge_result import JudgeStatus
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.models.evaluation import RetrievedEvidence
from app.observability.tracing import Tracer


INPUT_PATH = Path("logs/g1_batch_experiment.json")
OUTPUT_PATH = Path("logs/g1_judged.json")
CHECKPOINT_PATH = Path("logs/g1_judge_checkpoint.json")

BATCH_SIZE = 5
MAX_429_RETRIES = 3
RATE_LIMIT_COOLDOWN_SECONDS = 65


class RateLimitSafeGenerator:
    """Wrap the real generator so AnswerJudge.batch_judge can recover from 429s."""

    def __init__(self, generator):
        self.generator = generator

    def generate(self, request):
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                return self.generator.generate(request)

            except httpx.HTTPStatusError as exc:
                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )

                if status != 429:
                    raise

                if attempt >= MAX_429_RETRIES:
                    print(
                        "[JUDGE] 429 persists after "
                        f"{MAX_429_RETRIES} retries; aborting.",
                        flush=True,
                    )
                    raise

                retry_number = attempt + 1

                retry_after = None
                if exc.response is not None:
                    value = exc.response.headers.get("retry-after")
                    if value:
                        try:
                            retry_after = float(value)
                        except ValueError:
                            retry_after = None

                cooldown = max(
                    RATE_LIMIT_COOLDOWN_SECONDS,
                    retry_after or 0,
                )

                print(
                    "[JUDGE] 429 rate/quota limit; "
                    f"retrying ({retry_number}/{MAX_429_RETRIES}) "
                    f"after {cooldown:.0f}s cooldown...",
                    flush=True,
                )

                time.sleep(cooldown)

        raise RuntimeError("Unreachable rate-limit retry state.")


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {
            "completed_dense_batches": [],
            "completed_dense_crossencoder_batches": [],
            "batches": {},
        }

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError("Invalid G1 judge checkpoint.")

    data.setdefault("completed_dense_batches", [])
    data.setdefault("completed_dense_crossencoder_batches", [])
    data.setdefault("batches", {})

    return data


def save_checkpoint(checkpoint):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2)

    temp.replace(CHECKPOINT_PATH)


def build_case_map():
    dataset = EvaluationDataset.load()

    return {
        case.case_id: case
        for case in dataset.cases
    }


def evidence_from_json(items):
    return [
        RetrievedEvidence(
            rank=item["rank"],
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            distance=item.get("distance"),
            text=item["text"],
        )
        for item in items
    ]


def extract_condition_cases(
    batch,
    condition_name,
    case_map,
):
    cases = []

    for case_id in batch["case_ids"]:
        source = batch[condition_name]

        answer_items = (
            source["generation"]
            ["structured_output"]
            ["answers"]
        )

        answer_by_id = {
            item["case_id"]: item["answer"]
            for item in answer_items
        }

        if case_id not in answer_by_id:
            raise RuntimeError(
                f"{condition_name}: missing generated answer "
                f"for {case_id}"
            )

        retrieval = source["retrievals"].get(case_id)

        if retrieval is None:
            raise RuntimeError(
                f"{condition_name}: missing retrieval "
                f"for {case_id}"
            )

        reference_case = case_map.get(case_id)

        if reference_case is None:
            raise RuntimeError(
                f"Dataset does not contain case {case_id}"
            )

        cases.append(
            {
                "case_id": case_id,
                "question": reference_case.question,
                "expected_answer": reference_case.expected_answer,
                "expected_topics": reference_case.expected_topics,
                "generated_answer": answer_by_id[case_id],
                "retrieved_evidence": evidence_from_json(
                    retrieval["results"]
                ),
            }
        )

    return cases


def serialize_verdict(verdict):
    result = verdict.result

    return {
        "status": verdict.status.value,
        "structured_output_valid": verdict.structured_output_valid,
        "judge_model": verdict.judge_model,
        "judge_provider": verdict.judge_provider,
        "judge_latency_ms": verdict.judge_latency_ms,
        "judge_tokens": {
            "input_tokens": verdict.judge_tokens.input_tokens,
            "output_tokens": verdict.judge_tokens.output_tokens,
            "total_tokens": verdict.judge_tokens.total_tokens,
        },
        "judge_cost_usd": verdict.judge_cost_usd,
        "judge_failure": verdict.judge_failure,
        "result": (
            {
                "answer_correct": result.answer_correct,
                "answer_grounded": result.answer_grounded,
                "topics_covered": result.topics_covered,
                "topics_missing": result.topics_missing,
                "unsupported_claims": result.unsupported_claims,
                "reasoning": result.reasoning,
            }
            if result is not None
            else None
        ),
    }


def validate_verdicts(case_ids, verdicts, condition_name):
    if len(verdicts) != len(case_ids):
        raise RuntimeError(
            f"{condition_name}: expected {len(case_ids)} verdicts, "
            f"received {len(verdicts)}"
        )

    invalid = [
        verdict
        for verdict in verdicts
        if verdict.status != JudgeStatus.VALID
        or verdict.result is None
        or not verdict.structured_output_valid
    ]

    if invalid:
        details = [
            {
                "status": verdict.status.value,
                "failure": verdict.judge_failure,
            }
            for verdict in invalid
        ]
        raise RuntimeError(
            f"{condition_name}: invalid judge verdicts: {details}"
        )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing G1 generation artifact: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        experiment = json.load(file)

    batches = experiment["batches"]
    batch_keys = sorted(batches, key=int)

    if len(batch_keys) != 15:
        raise RuntimeError(
            f"Expected 15 batches, found {len(batch_keys)}"
        )

    case_map = build_case_map()

    settings = get_settings()
    tracer = Tracer()

    base_generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=settings.llm_structured_output_mode,
    )

    safe_generator = RateLimitSafeGenerator(base_generator)
    judge = AnswerJudge(safe_generator)

    checkpoint = load_checkpoint()

    completed_dense = set(
        checkpoint["completed_dense_batches"]
    )
    completed_ce = set(
        checkpoint["completed_dense_crossencoder_batches"]
    )
    batch_results = checkpoint["batches"]

    print("\nEvalForge G1 — AnswerJudge phase", flush=True)
    print(f"Cases: 75", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Batches: {len(batch_keys)}", flush=True)
    print("Judge requests planned: 30", flush=True)

    for position, batch_key in enumerate(batch_keys, start=1):
        batch = batches[batch_key]
        case_ids = batch["case_ids"]

        print("\n" + "=" * 70, flush=True)
        print(
            f"JUDGE BATCH {position}/15",
            flush=True,
        )
        print(f"Cases: {case_ids}", flush=True)
        print("=" * 70, flush=True)

        batch_results.setdefault(
            batch_key,
            {
                "batch_index": batch["batch_index"],
                "case_ids": case_ids,
            },
        )

        if batch_key not in completed_dense:
            print(
                "[DENSE] judging 5 cases...",
                flush=True,
            )

            dense_cases = extract_condition_cases(
                batch,
                "dense",
                case_map,
            )

            verdicts = judge.batch_judge(dense_cases)

            validate_verdicts(
                case_ids,
                verdicts,
                "DENSE",
            )

            batch_results[batch_key]["dense"] = [
                serialize_verdict(verdict)
                for verdict in verdicts
            ]

            completed_dense.add(batch_key)
            checkpoint["completed_dense_batches"] = sorted(
                completed_dense,
                key=int,
            )
            checkpoint["batches"] = batch_results
            save_checkpoint(checkpoint)

            print(
                "[DENSE] PASS — 5/5 valid verdicts",
                flush=True,
            )
        else:
            print(
                "[DENSE] SKIP — already checkpointed",
                flush=True,
            )

        if batch_key not in completed_ce:
            print(
                "[DENSE+CE] judging 5 cases...",
                flush=True,
            )

            ce_cases = extract_condition_cases(
                batch,
                "dense_crossencoder",
                case_map,
            )

            verdicts = judge.batch_judge(ce_cases)

            validate_verdicts(
                case_ids,
                verdicts,
                "DENSE+CE",
            )

            batch_results[batch_key]["dense_crossencoder"] = [
                serialize_verdict(verdict)
                for verdict in verdicts
            ]

            completed_ce.add(batch_key)
            checkpoint["completed_dense_crossencoder_batches"] = sorted(
                completed_ce,
                key=int,
            )
            checkpoint["batches"] = batch_results
            save_checkpoint(checkpoint)

            print(
                "[DENSE+CE] PASS — 5/5 valid verdicts",
                flush=True,
            )
        else:
            print(
                "[DENSE+CE] SKIP — already checkpointed",
                flush=True,
            )

    if len(completed_dense) != 15 or len(completed_ce) != 15:
        raise RuntimeError(
            "G1 judge phase incomplete; checkpoint preserved."
        )

    output = {
        "experiment": "g1_answer_judging",
        "source": str(INPUT_PATH),
        "dataset_size": 75,
        "batch_size": BATCH_SIZE,
        "judge_request_count": 30,
        "batches": {
            key: batch_results[key]
            for key in batch_keys
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    tracer.flush()

    print("\n" + "=" * 70, flush=True)
    print("G1 ANSWER JUDGING COMPLETE", flush=True)
    print(f"Saved: {OUTPUT_PATH}", flush=True)
    print("Cases judged: 75 Dense + 75 Dense+CE", flush=True)
    print("Judge requests: 30", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
