from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
import argparse
from pathlib import Path

from app.core.config import get_settings
from app.evaluation import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.judge_reliability import compute_judge_reliability
from app.evaluation.runner import (
    Evaluator,
    JUDGE_BATCH_SIZE,
    MAX_RETRIES,
    RATE_LIMIT_COOLDOWN_SECONDS,
)
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.observability.tracing import Tracer


CHECKPOINT_DIR = Path(".evalforge_checkpoints")
LOG_DIR = Path("logs")


def _redact(value):
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(
                    marker in key.lower()
                    for marker in (
                        "api_key",
                        "apikey",
                        "password",
                        "secret",
                        "token",
                    )
                )
                else _redact(item)
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_redact(item) for item in value]

    return value


def _safe_settings_dump(settings):
    try:
        data = settings.model_dump(mode="json")
    except AttributeError:
        try:
            data = dict(settings)
        except Exception:
            data = {
                key: value
                for key, value in vars(settings).items()
                if not key.startswith("_")
            }

    return _redact(data)


def _find_existing_log(run_id: str) -> Path | None:
    if not LOG_DIR.exists():
        return None

    matches = sorted(
        LOG_DIR.glob(f"evaluation_*_{run_id}.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return matches[0] if matches else None


def _create_log_path(run_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    return LOG_DIR / f"evaluation_{timestamp}_{run_id}.log"


def _write_log_header(
    log_path: Path,
    *,
    run_id: str,
    settings,
    dataset_size: int,
    evaluator: Evaluator,
    resumed: bool,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", encoding="utf-8") as log:
        if file_exists:
            log.write("\n\n")
            log.write("=" * 100 + "\n")
            log.write("EVALFORGE RUN RESUMED\n")
            log.write("=" * 100 + "\n")
            log.write(
                f"Resumed at: {datetime.now(timezone.utc).isoformat()}\n"
            )
            log.write(f"Run ID: {run_id}\n")
            log.write("\n")
        else:
            log.write("=" * 100 + "\n")
            log.write("EVALFORGE DETAILED EVALUATION LOG\n")
            log.write("=" * 100 + "\n\n")
            log.write(f"Run ID: {run_id}\n")
            log.write(
                f"Started at: {datetime.now(timezone.utc).isoformat()}\n"
            )
            log.write(f"Dataset size: {dataset_size}\n")
            log.write(f"Resumed: {resumed}\n")
            log.write("\n")

        log.write("=" * 100 + "\n")
        log.write("EVALUATION PARAMETERS\n")
        log.write("=" * 100 + "\n")

        log.write(
            json.dumps(
                {
                    "settings": _safe_settings_dump(settings),
                    "retriever": {
                        "mode": evaluator.retriever.mode,
                        "top_k": 5,
                        "candidate_k": evaluator.retriever.candidate_k,
                        "reranking_enabled": (
                            evaluator.retriever.reranking_enabled
                        ),
                        "reranker_candidate_k": (
                            evaluator.retriever.reranker_candidate_k
                        ),
                        "hybrid_retrieval_enabled": (
                            evaluator.retriever.hybrid_retrieval_enabled
                        ),
                    },
                    "evaluation": {
                        "judge_enabled": (
                            evaluator.answer_judge is not None
                        ),
                        "judge_batch_size": JUDGE_BATCH_SIZE,
                        "rate_limit_cooldown_seconds": (
                            RATE_LIMIT_COOLDOWN_SECONDS
                        ),
                        "max_retries": (
                            MAX_RETRIES
                        ),
                    },
                },
                indent=2,
                default=str,
            )
        )
        log.write("\n\n")


def _append_results_log(
    log_path: Path,
    *,
    run,
    settings,
    judge_reliability,
):
    with log_path.open("a", encoding="utf-8") as log:
        log.write("=" * 100 + "\n")
        log.write("CASE RESULTS\n")
        log.write("=" * 100 + "\n\n")

        for result in run.results:
            log.write("-" * 100 + "\n")
            log.write(f"Case: {result.case_id}\n")
            log.write(f"Question: {result.question}\n")
            log.write(
                "Expected documents: "
                + (
                    ", ".join(result.expected_documents)
                    or "None"
                )
                + "\n"
            )

            if result.generated_answer is not None:
                log.write(
                    "Generated answer:\n"
                    f"{result.generated_answer}\n"
                )

            log.write(
                f"Topic coverage: {result.topic_coverage}\n"
            )
            log.write(
                f"Failure type: {result.failure_type}\n"
            )
            log.write(
                "Relevant evidence found: "
                f"{result.relevant_documents_found}\n"
            )
            log.write(
                "First relevant rank: "
                f"{result.first_relevant_rank}\n"
            )
            log.write(
                "Missing documents: "
                + (
                    ", ".join(result.missing_documents)
                    or "None"
                )
                + "\n"
            )
            log.write(
                "Confounding documents: "
                + (
                    ", ".join(result.confounding_documents)
                    or "None"
                )
                + "\n"
            )
            log.write(
                f"Hit@1: {result.hit_at_1}\n"
                f"Hit@3: {result.hit_at_3}\n"
                f"Hit@5: {result.hit_at_5}\n"
                f"Recall@1: {result.recall_at_1}\n"
                f"Recall@3: {result.recall_at_3}\n"
                f"Recall@5: {result.recall_at_5}\n"
                f"MRR: {result.mrr}\n"
                f"Retrieval latency: "
                f"{result.retrieval_latency_ms} ms\n"
            )

            log.write("\nRetrieved evidence:\n")

            for evidence in result.retrieved_evidence:
                log.write(
                    f"  Rank {evidence.rank} | "
                    f"Document={evidence.document_id} | "
                    f"Chunk={evidence.chunk_id} | "
                    f"Distance={evidence.distance}\n"
                )
                log.write(
                    f"    Text: {evidence.text}\n"
                )

            if result.judge_verdict is not None:
                verdict = result.judge_verdict

                log.write("\nJudge:\n")
                log.write(
                    f"  Status: {verdict.status.value}\n"
                )
                log.write(
                    "  Structured output valid: "
                    f"{verdict.structured_output_valid}\n"
                )
                log.write(
                    "  Judge attempts: "
                    f"{verdict.judge_metadata.get('judge_attempt', 1)}\n"
                )
                log.write(
                    "  Judge recovered: "
                    f"{verdict.judge_metadata.get('judge_recovered', False)}\n"
                )

                if verdict.judge_metadata.get(
                    "initial_failure_type"
                ):
                    log.write(
                        "  Initial failure type: "
                        f"{verdict.judge_metadata['initial_failure_type']}\n"
                    )

                if verdict.result is not None:
                    log.write(
                        "  Answer correct: "
                        f"{verdict.result.answer_correct}\n"
                    )
                    log.write(
                        "  Answer grounded: "
                        f"{verdict.result.answer_grounded}\n"
                    )
                    log.write(
                        "  Topics covered: "
                        f"{verdict.result.topics_covered}\n"
                    )
                    log.write(
                        "  Topics missing: "
                        f"{verdict.result.topics_missing}\n"
                    )
                    log.write(
                        "  Unsupported claims: "
                        f"{verdict.result.unsupported_claims}\n"
                    )
                    log.write(
                        "  Reasoning: "
                        f"{verdict.result.reasoning}\n"
                    )

            log.write("\n")

        log.write("=" * 100 + "\n")
        log.write("AGGREGATE METRICS\n")
        log.write("=" * 100 + "\n")
        log.write(f"Mean Hit@1: {run.mean_hit_at_1:.6f}\n")
        log.write(f"Mean Hit@3: {run.mean_hit_at_3:.6f}\n")
        log.write(f"Mean Hit@5: {run.mean_hit_at_5:.6f}\n")
        log.write(
            f"Mean Recall@1: {run.mean_recall_at_1:.6f}\n"
        )
        log.write(
            f"Mean Recall@3: {run.mean_recall_at_3:.6f}\n"
        )
        log.write(
            f"Mean Recall@5: {run.mean_recall_at_5:.6f}\n"
        )
        log.write(f"Mean MRR: {run.mean_mrr:.6f}\n")
        log.write(
            "Mean topic coverage: "
            + (
                f"{run.mean_topic_coverage:.6f}\n"
                if run.mean_topic_coverage is not None
                else "N/A\n"
            )
        )
        log.write(
            f"Mean retrieval latency: "
            f"{run.mean_retrieval_latency_ms:.6f} ms\n"
        )

        log.write("\n")
        log.write("=" * 100 + "\n")
        log.write("JUDGE RELIABILITY\n")
        log.write("=" * 100 + "\n")
        log.write(
            f"Evaluated cases: "
            f"{judge_reliability.total_cases}\n"
        )
        log.write(
            f"Final validity rate: "
            f"{judge_reliability.structured_output_validity_rate:.6f}\n"
        )
        log.write(
            f"Recovery rate: "
            f"{judge_reliability.recovery_rate:.6f}\n"
        )
        log.write(
            f"Failure rate: "
            f"{judge_reliability.failure_rate:.6f}\n"
        )
        log.write(
            f"Unrecovered failures: "
            f"{judge_reliability.failure_count}\n"
        )

        log.write("\n")
        log.write("=" * 100 + "\n")
        log.write("RUN PROVENANCE\n")
        log.write("=" * 100 + "\n")
        log.write(f"Run ID: {run.run_id}\n")
        log.write(f"Created at: {run.created_at}\n")
        log.write(
            f"Retriever: {run.retrieval_config.mode}\n"
        )
        log.write(
            f"Top K: {run.retrieval_config.top_k}\n"
        )
        log.write(
            f"Candidate K: {run.retrieval_config.candidate_k}\n"
        )
        log.write(
            "Reranking: "
            f"{run.retrieval_config.reranking_enabled}\n"
        )
        log.write(
            "Reranker K: "
            f"{run.retrieval_config.reranker_candidate_k}\n"
        )
        log.write(
            "Hybrid: "
            f"{run.retrieval_config.hybrid_retrieval_enabled}\n"
        )
        log.write(
            f"Application model: {settings.llm_model}\n"
        )
        log.write(
            f"Judge model: {settings.llm_model}\n"
        )
        log.write(
            "Structured output: "
            f"{settings.llm_structured_output_mode}\n"
        )

        log.write("\n")
        log.write("=" * 100 + "\n")
        log.write(
            f"Completed at: "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        log.write("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the EvalForge evaluation."
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume a failed/incomplete evaluation run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    tracer = Tracer()

    print("EvalForge Evaluation", flush=True)
    print("────────────────────────────────────────", flush=True)

    print("[1/4] Loading dataset...", flush=True)

    dataset = EvaluationDataset.load()

    print(
        f"       {len(dataset)} cases",
        flush=True,
    )

    application_generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=settings.llm_structured_output_mode,
    )

    judge_generator = OpenAICompatibleGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "",
        default_model=settings.llm_model,
        tracer=tracer,
        structured_output_mode=settings.llm_structured_output_mode,
    )

    answer_judge = AnswerJudge(
        generator=judge_generator,
    )

    evaluator = Evaluator(
        generator=application_generator,
        tracer=tracer,
        answer_judge=answer_judge,
    )

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if args.resume:
        requested_checkpoint = (
            CHECKPOINT_DIR / f"{args.resume}.json"
        )

        if not requested_checkpoint.exists():
            raise SystemExit(
                f"ERROR: no checkpoint found for run "
                f"{args.resume}"
            )

        run_id = args.resume
        resumed = True

        print(
            f"[2/4] Resuming run {run_id}",
            flush=True,
        )

    else:
        run_id = str(uuid.uuid4())
        resumed = False

        print(
            "[2/4] Starting new run",
            flush=True,
        )

    existing_log = _find_existing_log(run_id)

    if existing_log is not None:
        log_path = existing_log
    else:
        log_path = _create_log_path(run_id)

    _write_log_header(
        log_path,
        run_id=run_id,
        settings=settings,
        dataset_size=len(dataset),
        evaluator=evaluator,
        resumed=resumed,
    )

    print(
        "[3/4] Running evaluation...",
        flush=True,
    )

    try:
        run = evaluator.evaluate_dataset(
            dataset,
            run_id=run_id,
        )
    except Exception as exc:
        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log:
            log.write("\n")
            log.write("=" * 100 + "\n")
            log.write(
                f"FAILED at: "
                f"{datetime.now(timezone.utc).isoformat()}\n"
            )
            log.write(f"Run ID: {run_id}\n")
            log.write(
                f"Exception type: "
                f"{type(exc).__name__}\n"
            )
            log.write(
                f"Exception: {exc}\n"
            )
            log.write("=" * 100 + "\n")

        tracer.flush()

        print(
            f"\n[ERROR] Evaluation failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        print(
            f"Run ID       {run_id}",
            flush=True,
        )
        print(
            f"Detailed log {log_path}",
            flush=True,
        )

        raise SystemExit(1) from exc

    judge_results = [
        {
            "case_id": result.case_id,
            "judge_verdict": (
                result.judge_verdict.model_dump(
                    mode="json"
                )
                if result.judge_verdict is not None
                else None
            ),
        }
        for result in run.results
    ]

    judge_reliability = compute_judge_reliability(
        judge_results
    )

    _append_results_log(
        log_path,
        run=run,
        settings=settings,
        judge_reliability=judge_reliability,
    )

    print(
        "[4/4] Evaluation completed",
        flush=True,
    )

    print()
    print("Results")
    print("────────────────────────────────────────")
    print(f"Cases        {run.dataset_size}")
    print(f"Hit@1        {run.mean_hit_at_1:.3f}")
    print(f"Hit@3        {run.mean_hit_at_3:.3f}")
    print(f"Hit@5        {run.mean_hit_at_5:.3f}")
    print(f"Recall@1     {run.mean_recall_at_1:.3f}")
    print(f"Recall@3     {run.mean_recall_at_3:.3f}")
    print(f"Recall@5     {run.mean_recall_at_5:.3f}")
    print(f"MRR          {run.mean_mrr:.3f}")
    print(
        "Topic cov.   "
        + (
            f"{run.mean_topic_coverage:.3f}"
            if run.mean_topic_coverage is not None
            else "N/A"
        )
    )
    print(
        f"Latency      "
        f"{run.mean_retrieval_latency_ms:.2f} ms"
    )

    print()
    print("Judge")
    print("────────────────────────────────────────")
    print(
        f"Validity     "
        f"{judge_reliability.structured_output_validity_rate:.1%}"
    )
    print(
        f"Failures     "
        f"{judge_reliability.failure_rate:.1%}"
    )

    print()
    print(f"Run ID       {run.run_id}")
    print(f"Detailed log {log_path}")

    tracer.flush()


if __name__ == "__main__":
    main()
