from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.db.repository import EvaluationRepository
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.diagnostics import analyze_retrieval
from app.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    topic_coverage,
)
from app.generation.base import GenerationRequest
from app.generation.factory import create_generator
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievedEvidence,
    RetrievalConfig,
    JudgeConfig,
)
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


JUDGE_BATCH_SIZE = 5

RATE_LIMIT_COOLDOWN_SECONDS = 60

# Maximum retries after the initial request.
# A request can therefore be attempted at most
# MAX_RETRIES + 1 times.
MAX_RETRIES = 2

CHECKPOINT_DIR = Path(".evalforge_checkpoints")


def _is_retryable_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True

    response = getattr(exc, "response", None)

    if response is not None and getattr(
        response,
        "status_code",
        None,
    ) == 429:
        return True

    message = str(exc).lower()

    if any(
        marker in message
        for marker in (
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota",
            "resource exhausted",
        )
    ):
        return True

    # Transient HTTP/network timeout.
    try:
        import httpx

        if isinstance(exc, httpx.ReadTimeout):
            return True
    except ImportError:
        pass

    return False


def _cooldown(seconds: int) -> None:
    print(
        f"[cooldown] {seconds}s",
        flush=True,
    )
    time.sleep(seconds)


def _checkpoint_path(run_id: str) -> Path:
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    return CHECKPOINT_DIR / f"{run_id}.json"


def _save_checkpoint(
    path: Path,
    *,
    run_id: str,
    phase: str,
    completed_case_ids: set[str],
    results: dict[str, EvaluationResult],
    retry_state: dict | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "phase": phase,
        "completed_case_ids": sorted(
            completed_case_ids
        ),
        "results": [
            result.model_dump(mode="json")
            for result in results.values()
        ],
        "retry_state": retry_state or {},
    }

    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _load_checkpoint(path: Path):
    if not path.exists():
        return None

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    results = {
        item["case_id"]: EvaluationResult.model_validate(
            item
        )
        for item in payload.get(
            "results",
            [],
        )
    }

    return (
        payload["run_id"],
        payload["phase"],
        set(
            payload.get(
                "completed_case_ids",
                [],
            )
        ),
        results,
        payload.get("retry_state", {}),
    )


class Evaluator:
    def __init__(
        self,
        retriever=None,
        generator=None,
        tracer=None,
        repository=None,
        answer_judge: AnswerJudge | None = None,
    ):
        self.tracer = tracer or Tracer()

        self.retriever = (
            retriever
            or Retriever(
                tracer=self.tracer
            )
        )

        self.generator = (
            generator
            or create_generator(
                settings=get_settings(),
                tracer=self.tracer,
            )
        )

        self.repository = (
            repository
            or EvaluationRepository()
        )

        self.answer_judge = answer_judge

    def _build_retrieval_config(self):
        return RetrievalConfig(
            mode=self.retriever.mode,
            top_k=5,
            candidate_k=self.retriever.candidate_k,
            reranking_enabled=(
                self.retriever.reranking_enabled
            ),
            reranker_candidate_k=(
                self.retriever.reranker_candidate_k
            ),
            hybrid_retrieval_enabled=(
                self.retriever.hybrid_retrieval_enabled
            ),
        )

    def evaluate_case(
        self,
        case,
        evaluation_run_id: str | None = None,
    ):
        trace_metadata = {
            "evaluation_run_id": evaluation_run_id,
            "case_id": case.case_id,
        }

        retrieval = self.retriever.retrieve(
            query=case.question,
            top_k=5,
            trace_metadata=trace_metadata,
        )

        retrieved_evidence = [
            RetrievedEvidence(
                rank=result.rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                distance=result.distance,
                text=result.text,
            )
            for result in retrieval.results
        ]

        retrieved_documents = [
            evidence.document_id
            for evidence in retrieved_evidence
        ]

        diagnostic = analyze_retrieval(
            case.expected_documents,
            retrieved_documents,
        )

        generation_request = GenerationRequest(
            question=case.question,
            context=retrieval.results,
            model=None,
            temperature=0.0,
            metadata=trace_metadata,
        )

        generation = self.generator.generate(
            generation_request
        )

        case_topic_coverage = None

        if case.expected_topics:
            case_topic_coverage = topic_coverage(
                case.expected_topics,
                generation.answer,
            )

        return EvaluationResult(
            case_id=case.case_id,
            question=case.question,
            expected_documents=case.expected_documents,
            generated_answer=generation.answer,
            topic_coverage=case_topic_coverage,
            answer_judge=None,
            judge_verdict=None,
            retrieved_evidence=retrieved_evidence,
            failure_type=diagnostic.failure_type,
            relevant_documents_found=(
                diagnostic.relevant_documents_found
            ),
            first_relevant_rank=(
                diagnostic.first_relevant_rank
            ),
            missing_documents=(
                diagnostic.missing_documents
            ),
            confounding_documents=(
                diagnostic.confounding_documents
            ),
            hit_at_1=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                1,
            ),
            hit_at_3=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                3,
            ),
            hit_at_5=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                5,
            ),
            recall_at_1=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                1,
            ),
            recall_at_3=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                3,
            ),
            recall_at_5=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                5,
            ),
            mrr=reciprocal_rank(
                case.expected_documents,
                retrieved_documents,
            ),
            retrieval_latency_ms=(
                retrieval.latency_ms
            ),
        )

    def evaluate_dataset(
        self,
        dataset,
        run_id: str | None = None,
    ):
        run_id = run_id or str(uuid4())

        with self.tracer.trace(
            name="evaluation_run",
            input={
                "dataset_size": len(dataset)
            },
            metadata={
                "evaluation_run_id": run_id,
                "retriever_mode": self.retriever.mode,
                "retrieval_top_k": 5,
                "answer_judge_enabled": (
                    self.answer_judge is not None
                ),
                "judge_batch_size": JUDGE_BATCH_SIZE,
                "max_retries": (
                    MAX_RETRIES
                ),
            },
        ) as observation:
            run = self._evaluate_dataset(
                dataset,
                run_id=run_id,
            )

            self.repository.save_run(run)

            checkpoint_path = _checkpoint_path(
                run.run_id
            )

            if checkpoint_path.exists():
                checkpoint_path.unlink()

            if observation is not None:
                observation.update(
                    output={
                        "dataset_size": (
                            run.dataset_size
                        ),
                        "mean_hit_at_1": (
                            run.mean_hit_at_1
                        ),
                        "mean_hit_at_3": (
                            run.mean_hit_at_3
                        ),
                        "mean_hit_at_5": (
                            run.mean_hit_at_5
                        ),
                        "mean_recall_at_1": (
                            run.mean_recall_at_1
                        ),
                        "mean_recall_at_3": (
                            run.mean_recall_at_3
                        ),
                        "mean_recall_at_5": (
                            run.mean_recall_at_5
                        ),
                        "mean_mrr": run.mean_mrr,
                        "mean_topic_coverage": (
                            run.mean_topic_coverage
                        ),
                    }
                )

            return run

    def _evaluate_dataset(
        self,
        dataset,
        run_id: str | None = None,
    ):
        run_id = run_id or str(uuid4())

        checkpoint_path = _checkpoint_path(
            run_id
        )

        checkpoint = _load_checkpoint(
            checkpoint_path
        )

        results_by_case_id = {}
        completed_generation_ids = set()
        completed_judge_ids = set()

        retry_state = {
            "generation": {},
            "judge": {},
        }

        if checkpoint is not None:
            (
                _,
                phase,
                completed_case_ids,
                checkpoint_results,
                checkpoint_retry_state,
            ) = checkpoint

            results_by_case_id.update(
                checkpoint_results
            )

            retry_state.update(
                checkpoint_retry_state
            )

            if phase == "generation":
                completed_generation_ids = (
                    completed_case_ids
                )

            elif phase == "judge":
                completed_generation_ids = {
                    case.case_id
                    for case in dataset.cases
                    if case.case_id
                    in results_by_case_id
                }

                completed_judge_ids = (
                    completed_case_ids
                )

            print(
                f"[resume] {phase} checkpoint "
                f"({len(completed_case_ids)} complete)",
                flush=True,
            )

        total_cases = len(dataset.cases)

        # -----------------------------------------------------
        # Generation phase
        # -----------------------------------------------------

        for index, case in enumerate(
            dataset.cases,
            start=1,
        ):
            if (
                case.case_id
                in completed_generation_ids
            ):
                continue

            attempts = int(
                retry_state["generation"].get(
                    case.case_id,
                    0,
                )
            )

            while True:
                try:
                    print(
                        f"\r[generation] {index}/{total_cases}",
                        end="",
                        flush=True,
                    )

                    result = self.evaluate_case(
                        case,
                        evaluation_run_id=run_id,
                    )

                    results_by_case_id[
                        case.case_id
                    ] = result

                    completed_generation_ids.add(
                        case.case_id
                    )

                    retry_state["generation"].pop(
                        case.case_id,
                        None,
                    )

                    _save_checkpoint(
                        checkpoint_path,
                        run_id=run_id,
                        phase="generation",
                        completed_case_ids=(
                            completed_generation_ids
                        ),
                        results=results_by_case_id,
                        retry_state=retry_state,
                    )

                    break

                except Exception as exc:
                    if not _is_retryable_error(
                        exc
                    ):
                        raise

                    if attempts >= MAX_RETRIES:
                        print(
                            "\n[generation] "
                            "retry limit "
                            f"reached ({MAX_RETRIES}); "
                            "checkpoint preserved.",
                            flush=True,
                        )
                        raise

                    attempts += 1

                    print(
                        "\n[generation] "
                        f"retryable error; retrying "
                        f"({attempts}/{MAX_RETRIES})",
                        flush=True,
                    )

                    retry_state["generation"][
                        case.case_id
                    ] = attempts

                    _save_checkpoint(
                        checkpoint_path,
                        run_id=run_id,
                        phase="generation",
                        completed_case_ids=(
                            completed_generation_ids
                        ),
                        results=results_by_case_id,
                        retry_state=retry_state,
                    )

                    _cooldown(
                        RATE_LIMIT_COOLDOWN_SECONDS
                    )

        print(
            f"\r[generation] {total_cases}/{total_cases}",
            flush=True,
        )

        results = [
            results_by_case_id[
                case.case_id
            ]
            for case in dataset.cases
        ]

        # -----------------------------------------------------
        # Judge phase
        # -----------------------------------------------------

        if self.answer_judge is not None:
            total_batches = (
                (
                    len(dataset.cases)
                    + JUDGE_BATCH_SIZE
                    - 1
                )
                // JUDGE_BATCH_SIZE
            )

            for batch_number, batch_start in enumerate(
                range(
                    0,
                    len(dataset.cases),
                    JUDGE_BATCH_SIZE,
                ),
                start=1,
            ):
                batch_cases = dataset.cases[
                    batch_start:
                    batch_start
                    + JUDGE_BATCH_SIZE
                ]

                pending_cases = [
                    case
                    for case in batch_cases
                    if case.case_id
                    not in completed_judge_ids
                ]

                if not pending_cases:
                    continue

                judge_cases = []

                for case in pending_cases:
                    result = (
                        results_by_case_id[
                            case.case_id
                        ]
                    )

                    judge_cases.append(
                        {
                            "case_id": case.case_id,
                            "question": case.question,
                            "expected_answer": (
                                case.expected_answer
                            ),
                            "expected_topics": (
                                case.expected_topics
                            ),
                            "generated_answer": (
                                result.generated_answer
                                or ""
                            ),
                            "retrieved_evidence": (
                                result.retrieved_evidence
                            ),
                        }
                    )

                attempts = int(
                    retry_state["judge"].get(
                        str(batch_number),
                        0,
                    )
                )

                while True:
                    try:
                        print(
                            f"\r[judge] "
                            f"{batch_number}/{total_batches}",
                            end="",
                            flush=True,
                        )

                        with self.tracer.judge(
                            name="judge_batch",
                            input={
                                "case_count": len(judge_cases),
                            },
                            metadata={
                                "evaluation_run_id": run_id,
                                "batch_number": batch_number,
                                "case_ids": [
                                    item["case_id"]
                                    for item in judge_cases
                                ],
                            },
                        ):
                            verdicts = (
                                self.answer_judge.batch_judge(
                                    judge_cases
                                )
                            )

                        if len(verdicts) != len(
                            judge_cases
                        ):
                            raise RuntimeError(
                                "Judge returned an "
                                "unexpected number "
                                "of verdicts."
                            )

                        for case, verdict in zip(
                            pending_cases,
                            verdicts,
                            strict=True,
                        ):
                            result = (
                                results_by_case_id[
                                    case.case_id
                                ]
                            )

                            result.answer_judge = (
                                verdict.result
                            )

                            result.judge_verdict = (
                                verdict
                            )

                            completed_judge_ids.add(
                                case.case_id
                            )

                        retry_state["judge"].pop(
                            str(batch_number),
                            None,
                        )

                        _save_checkpoint(
                            checkpoint_path,
                            run_id=run_id,
                            phase="judge",
                            completed_case_ids=(
                                completed_judge_ids
                            ),
                            results=(
                                results_by_case_id
                            ),
                            retry_state=retry_state,
                        )

                        break

                    except Exception as exc:
                        if not _is_retryable_error(
                            exc
                        ):
                            raise

                        if attempts >= MAX_RETRIES:
                            print(
                                "\n[judge] "
                                "retry limit "
                                f"reached ({MAX_RETRIES}); "
                                "checkpoint preserved.",
                                flush=True,
                            )
                            raise

                        attempts += 1

                        print(
                            "\n[judge] "
                            f"retryable error; retrying "
                            f"({attempts}/{MAX_RETRIES})",
                            flush=True,
                        )

                        retry_state["judge"][
                            str(batch_number)
                        ] = attempts

                        _save_checkpoint(
                            checkpoint_path,
                            run_id=run_id,
                            phase="judge",
                            completed_case_ids=(
                                completed_judge_ids
                            ),
                            results=(
                                results_by_case_id
                            ),
                            retry_state=retry_state,
                        )

                        _cooldown(
                            RATE_LIMIT_COOLDOWN_SECONDS
                        )

            print(
                f"\r[judge] {total_batches}/{total_batches}",
                flush=True,
            )

            results = [
                results_by_case_id[
                    case.case_id
                ]
                for case in dataset.cases
            ]

        # -----------------------------------------------------
        # Aggregation
        # -----------------------------------------------------

        if results:
            mean_hit_at_1 = sum(
                result.hit_at_1
                for result in results
            ) / len(results)

            mean_hit_at_3 = sum(
                result.hit_at_3
                for result in results
            ) / len(results)

            mean_hit_at_5 = sum(
                result.hit_at_5
                for result in results
            ) / len(results)

            mean_recall_at_1 = sum(
                result.recall_at_1
                for result in results
            ) / len(results)

            mean_recall_at_3 = sum(
                result.recall_at_3
                for result in results
            ) / len(results)

            mean_recall_at_5 = sum(
                result.recall_at_5
                for result in results
            ) / len(results)

            mean_mrr = sum(
                result.mrr
                for result in results
            ) / len(results)

            topic_scores = [
                result.topic_coverage
                for result in results
                if result.topic_coverage
                is not None
            ]

            mean_topic_coverage = (
                sum(topic_scores)
                / len(topic_scores)
                if topic_scores
                else None
            )

            mean_retrieval_latency_ms = sum(
                result.retrieval_latency_ms
                for result in results
            ) / len(results)

        else:
            mean_hit_at_1 = 0.0
            mean_hit_at_3 = 0.0
            mean_hit_at_5 = 0.0
            mean_recall_at_1 = 0.0
            mean_recall_at_3 = 0.0
            mean_recall_at_5 = 0.0
            mean_mrr = 0.0
            mean_topic_coverage = None
            mean_retrieval_latency_ms = 0.0

        judge_config = None

        if self.answer_judge is not None:
            gen = getattr(
                self.answer_judge,
                "generator",
                None,
            )

            if (
                gen
                and getattr(
                    gen,
                    "model",
                    None,
                )
                and getattr(
                    gen,
                    "provider",
                    None,
                )
            ):
                judge_config = JudgeConfig(
                    model=gen.model,
                    provider=gen.provider,
                    temperature=getattr(
                        gen,
                        "temperature",
                        0.0,
                    ),
                    batch_size=JUDGE_BATCH_SIZE,
                )

        return EvaluationRun(
            run_id=run_id,
            created_at=datetime.now(
                timezone.utc
            ),
            dataset_size=len(dataset),
            retrieval_config=(
                self._build_retrieval_config()
            ),
            judge_config=judge_config,
            results=results,
            mean_hit_at_1=mean_hit_at_1,
            mean_hit_at_3=mean_hit_at_3,
            mean_hit_at_5=mean_hit_at_5,
            mean_recall_at_1=mean_recall_at_1,
            mean_recall_at_3=mean_recall_at_3,
            mean_recall_at_5=mean_recall_at_5,
            mean_mrr=mean_mrr,
            mean_topic_coverage=(
                mean_topic_coverage
            ),
            mean_retrieval_latency_ms=(
                mean_retrieval_latency_ms
            ),
        )
