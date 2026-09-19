import time
from typing import Any

from app.evaluation.judge_result import JudgeStatus, JudgeVerdict
from app.evaluation.structured_output import StructuredOutputValidator
from app.generation.base import (
    GenerationRequest,
    GenerationResult,
    Generator,
    StructuredOutputStatus,
)
from app.generation.usage import GenerationUsage
from app.models.evaluation import AnswerJudgeResult, RetrievedEvidence


ANSWER_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_correct": {"type": "boolean"},
        "answer_grounded": {"type": "boolean"},
        "topics_covered": {"type": "array", "items": {"type": "string"}},
        "topics_missing": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "answer_correct",
        "answer_grounded",
        "topics_covered",
        "topics_missing",
        "unsupported_claims",
        "reasoning",
    ],
    "additionalProperties": False,
}

BATCH_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": ANSWER_JUDGE_SCHEMA,
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

ANSWER_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for an LLM application.

Evaluate the generated answer against:
1. The user's question.
2. The expected answer.
3. The expected topics.
4. The retrieved evidence.

Determine:
- whether the generated answer is correct,
- whether it is grounded in the retrieved evidence,
- which expected topics are covered,
- which expected topics are missing,
- which claims are unsupported by the evidence.

Return only valid structured output matching the provided schema.
Do not invent evidence.
"""

BATCH_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for an LLM application.

Evaluate each case independently.

For every case, compare:
1. The user's question.
2. The expected answer.
3. The expected topics.
4. The generated answer.
5. The retrieved evidence.

Determine:
- whether the generated answer is correct,
- whether it is grounded in the retrieved evidence,
- which expected topics are covered,
- which expected topics are missing,
- which claims are unsupported by the evidence.

Return exactly one structured verdict for every case, in the same order as the input cases.

Return only valid structured output matching the provided schema.
Do not invent evidence.
"""

MAX_JUDGE_ATTEMPTS = 2


class AnswerJudge:
    def __init__(self, generator: Generator):
        self.generator = generator

    def judge(
        self,
        *,
        question: str,
        expected_answer: str,
        expected_topics: list[str],
        generated_answer: str,
        retrieved_evidence: list[RetrievedEvidence],
    ) -> JudgeVerdict:
        base_question = self._build_judge_input(
            question=question,
            expected_answer=expected_answer,
            expected_topics=expected_topics,
            generated_answer=generated_answer,
            retrieved_evidence=retrieved_evidence,
        )

        started = time.perf_counter()
        last_result: GenerationResult | None = None
        last_failure: str | None = None
        initial_failure: str | None = None
        initial_failure_type: str | None = None

        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            judge_input = base_question

            if last_failure is not None:
                judge_input = self._build_retry_input(
                    base_question=base_question,
                    failure=last_failure,
                )

            request = GenerationRequest(
                question=judge_input,
                context=[],
                temperature=0.0,
                system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
                response_schema=ANSWER_JUDGE_SCHEMA,
                metadata={
                    "judge_attempt": attempt,
                    "judge_max_attempts": MAX_JUDGE_ATTEMPTS,
                },
            )

            try:
                result = self.generator.generate(request)
            except Exception as exc:
                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000.0

                return JudgeVerdict(
                    result=None,
                    status=JudgeStatus.SYSTEM_ERROR,
                    judge_model="unknown",
                    judge_provider="unknown",
                    judge_latency_ms=elapsed_ms,
                    judge_tokens=GenerationUsage(),
                    judge_cost_usd=0.0,
                    structured_output_valid=False,
                    judge_failure=f"Judge generation failed: {exc}",
                    judge_metadata={
                        "judge_attempt": attempt,
                        "judge_max_attempts": MAX_JUDGE_ATTEMPTS,
                        "judge_recovered": False,
                        "initial_failure": initial_failure,
                        "initial_failure_type": initial_failure_type,
                    },
                )

            last_result = result
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            current_failure: str | None = None
            current_failure_type: str | None = None

            if result.structured_output_status == (
                StructuredOutputStatus.MISSING_CONTENT
            ):
                current_failure = "Judge generation missing content"
                current_failure_type = "missing_content"

            elif result.structured_output_status == (
                StructuredOutputStatus.PARSE_FAILED
            ):
                parse_error = result.metadata.get(
                    "structured_output_error"
                )

                current_failure = "Judge generation parsing failed"
                current_failure_type = "parse_failed"

                if parse_error:
                    current_failure = (
                        f"{current_failure}: {parse_error}"
                    )

            elif result.structured_output is None:
                current_failure = (
                    "Judge generation produced no structured output"
                )
                current_failure_type = "missing_content"

            else:
                validation = StructuredOutputValidator.validate(
                    result.structured_output,
                    AnswerJudgeResult,
                )

                if validation.valid:
                    validated = validation.value

                    if isinstance(
                        validated,
                        AnswerJudgeResult,
                    ):
                        judge_model, judge_provider = (
                            self._observed_provenance(result)
                        )

                        return JudgeVerdict(
                            result=validated,
                            status=JudgeStatus.VALID,
                            judge_model=judge_model,
                            judge_provider=judge_provider,
                            judge_latency_ms=elapsed_ms,
                            judge_tokens=result.usage,
                            judge_cost_usd=result.estimated_cost_usd,
                            structured_output_valid=True,
                            judge_failure=None,
                            judge_metadata={
                                **result.metadata,
                                "judge_attempt": attempt,
                                "judge_max_attempts": (
                                    MAX_JUDGE_ATTEMPTS
                                ),
                                "judge_recovered": (
                                    attempt > 1
                                ),
                                "initial_failure": initial_failure,
                        "initial_failure_type": initial_failure_type,
                    },
                        )

                    current_failure = (
                        "Judge schema validation produced an "
                        "unexpected model type"
                    )
                    current_failure_type = "schema_invalid"

                else:
                    current_failure = (
                        "Judge schema validation failed: "
                        f"{validation.errors}"
                    )
                    current_failure_type = "schema_invalid"

            if current_failure is not None:
                if initial_failure is None:
                    initial_failure = current_failure
                    initial_failure_type = current_failure_type

                last_failure = current_failure

            if attempt < MAX_JUDGE_ATTEMPTS:
                continue

            judge_model, judge_provider = (
                self._observed_provenance(last_result)
            )

            return JudgeVerdict(
                result=None,
                status=JudgeStatus.SYSTEM_ERROR,
                judge_model=judge_model,
                judge_provider=judge_provider,
                judge_latency_ms=elapsed_ms,
                judge_tokens=last_result.usage,
                judge_cost_usd=last_result.estimated_cost_usd,
                structured_output_valid=False,
                judge_failure=last_failure,
                judge_metadata={
                    **last_result.metadata,
                    "judge_attempt": attempt,
                    "judge_max_attempts": MAX_JUDGE_ATTEMPTS,
                    "judge_recovered": False,
                    "initial_failure": initial_failure,
                        "initial_failure_type": initial_failure_type,
                    },
            )

        raise RuntimeError("Unreachable judge state")

    def batch_judge(
        self,
        cases: list[dict[str, Any]],
    ) -> list[JudgeVerdict]:
        if not cases:
            return []

        request = GenerationRequest(
            question=self._build_batch_input(cases),
            context=[],
            temperature=0.0,
            system_prompt=BATCH_JUDGE_SYSTEM_PROMPT,
            response_schema=BATCH_JUDGE_SCHEMA,
        )

        started = time.perf_counter()

        try:
            result = self.generator.generate(request)
        except Exception as exc:
            message = str(exc).lower()

            is_rate_limit = (
                getattr(exc, "status_code", None) == 429
                or (
                    getattr(exc, "response", None) is not None
                    and getattr(
                        exc.response,
                        "status_code",
                        None,
                    ) == 429
                )
                or any(
                    marker in message
                    for marker in (
                        "429",
                        "rate limit",
                        "rate_limit",
                        "too many requests",
                        "quota",
                        "resource exhausted",
                    )
                )
            )

            if is_rate_limit:
                raise

            return [
                self.judge(
                    question=case["question"],
                    expected_answer=case["expected_answer"],
                    expected_topics=case.get("expected_topics", []),
                    generated_answer=case["generated_answer"],
                    retrieved_evidence=case.get(
                        "retrieved_evidence",
                        [],
                    ),
                )
                for case in cases
            ]

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.structured_output_status != (
            StructuredOutputStatus.PARSED
        ):
            return [
                self.judge(
                    question=case["question"],
                    expected_answer=case["expected_answer"],
                    expected_topics=case.get("expected_topics", []),
                    generated_answer=case["generated_answer"],
                    retrieved_evidence=case.get(
                        "retrieved_evidence",
                        [],
                    ),
                )
                for case in cases
            ]

        raw_verdicts = result.structured_output.get("verdicts")

        if not isinstance(raw_verdicts, list):
            return [
                self.judge(
                    question=case["question"],
                    expected_answer=case["expected_answer"],
                    expected_topics=case.get("expected_topics", []),
                    generated_answer=case["generated_answer"],
                    retrieved_evidence=case.get(
                        "retrieved_evidence",
                        [],
                    ),
                )
                for case in cases
            ]

        if len(raw_verdicts) != len(cases):
            return [
                self.judge(
                    question=case["question"],
                    expected_answer=case["expected_answer"],
                    expected_topics=case.get("expected_topics", []),
                    generated_answer=case["generated_answer"],
                    retrieved_evidence=case.get(
                        "retrieved_evidence",
                        [],
                    ),
                )
                for case in cases
            ]

        case_count = len(cases)

        per_case_usage = GenerationUsage(
            input_tokens=result.usage.input_tokens // case_count,
            output_tokens=result.usage.output_tokens // case_count,
        )

        per_case_cost = result.estimated_cost_usd / case_count

        judge_model, judge_provider = self._observed_provenance(result)

        verdicts = []

        for raw_verdict in raw_verdicts:
            validation = StructuredOutputValidator.validate(
                raw_verdict,
                AnswerJudgeResult,
            )

            if validation.valid and isinstance(
                validation.value,
                AnswerJudgeResult,
            ):
                verdicts.append(
                    JudgeVerdict(
                        result=validation.value,
                        status=JudgeStatus.VALID,
                        judge_model=judge_model,
                        judge_provider=judge_provider,
                        judge_latency_ms=elapsed_ms,
                        judge_tokens=per_case_usage,
                        judge_cost_usd=per_case_cost,
                        structured_output_valid=True,
                        judge_failure=None,
                        judge_metadata=result.metadata,
                    )
                )
            else:
                errors = validation.errors

                verdicts.append(
                    JudgeVerdict(
                        result=None,
                        status=JudgeStatus.SYSTEM_ERROR,
                        judge_model=judge_model,
                        judge_provider=judge_provider,
                        judge_latency_ms=elapsed_ms,
                        judge_tokens=per_case_usage,
                        judge_cost_usd=per_case_cost,
                        structured_output_valid=False,
                        judge_failure=(
                            "Judge schema validation failed: "
                            f"{errors}"
                        ),
                        judge_metadata=result.metadata,
                    )
                )

        return verdicts

    @staticmethod
    def _build_retry_input(
        *,
        base_question: str,
        failure: str,
    ) -> str:
        return (
            f"{base_question}\n\n"
            "RETRY INSTRUCTION:\n"
            "The previous judge response was rejected by the "
            "application validator.\n"
            f"Validation failure: {failure}\n\n"
            "Produce a new response that strictly conforms to "
            "the requested schema.\n"
            "Use exactly the required field names and types.\n"
            "Do not add extra fields.\n"
            "Return only the structured output."
        )

    @staticmethod
    def _observed_provenance(
        result: GenerationResult,
    ) -> tuple[str, str]:
        """Return observed provider provenance with safe fallbacks."""

        observation = result.observation

        if observation is not None:
            model = observation.actual_model or result.model
            provider = observation.actual_provider or result.provider

            return model, provider

        return result.model, result.provider

    @staticmethod
    def _build_judge_input(
        *,
        question: str,
        expected_answer: str,
        expected_topics: list[str],
        generated_answer: str,
        retrieved_evidence: list[RetrievedEvidence],
    ) -> str:
        evidence_text = "\n\n".join(
            (
                f"[Rank {item.rank}] "
                f"Document: {item.document_id} "
                f"Chunk: {item.chunk_id}\n"
                f"{item.text}"
            )
            for item in retrieved_evidence
        )

        return (
            f"QUESTION:\n{question}\n\n"
            f"EXPECTED ANSWER:\n{expected_answer}\n\n"
            f"EXPECTED TOPICS:\n{expected_topics}\n\n"
            f"GENERATED ANSWER:\n{generated_answer}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_text or '[none]'}"
        )

    @staticmethod
    def _build_batch_input(
        cases: list[dict[str, Any]],
    ) -> str:
        sections = []

        for index, case in enumerate(cases, start=1):
            evidence_text = "\n\n".join(
                (
                    f"[Rank {item.rank}] "
                    f"Document: {item.document_id} "
                    f"Chunk: {item.chunk_id}\n"
                    f"{item.text}"
                )
                for item in case.get("retrieved_evidence", [])
            )

            sections.append(
                f"Case {index}\n"
                f"QUESTION:\n{case['question']}\n\n"
                f"EXPECTED ANSWER:\n{case['expected_answer']}\n\n"
                f"EXPECTED TOPICS:\n"
                f"{case.get('expected_topics', [])}\n\n"
                f"GENERATED ANSWER:\n"
                f"{case['generated_answer']}\n\n"
                f"RETRIEVED EVIDENCE:\n"
                f"{evidence_text or '[none]'}"
            )

        return "\n\n---\n\n".join(sections)
