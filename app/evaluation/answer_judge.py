import time
from typing import Any

from app.evaluation.judge_result import JudgeStatus, JudgeVerdict
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
        request = GenerationRequest(
            question=self._build_judge_input(
                question=question,
                expected_answer=expected_answer,
                expected_topics=expected_topics,
                generated_answer=generated_answer,
                retrieved_evidence=retrieved_evidence,
            ),
            context=[],
            temperature=0.0,
            system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
            response_schema=ANSWER_JUDGE_SCHEMA,
        )

        started = time.perf_counter()

        try:
            result = self.generator.generate(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0

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
                judge_metadata={},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.structured_output_status == (
            StructuredOutputStatus.MISSING_CONTENT
        ):
            return self._system_error(
                result=result,
                elapsed_ms=elapsed_ms,
                failure="Judge generation missing content",
            )

        if result.structured_output_status == (
            StructuredOutputStatus.PARSE_FAILED
        ):
            parse_error = result.metadata.get(
                "structured_output_error"
            )

            failure = "Judge generation parsing failed"

            if parse_error:
                failure = f"{failure}: {parse_error}"

            return self._system_error(
                result=result,
                elapsed_ms=elapsed_ms,
                failure=failure,
            )

        if result.structured_output is None:
            return self._system_error(
                result=result,
                elapsed_ms=elapsed_ms,
                failure="Judge generation produced no structured output",
            )

        try:
            validated = AnswerJudgeResult.model_validate(
                result.structured_output
            )
        except Exception as exc:
            return self._system_error(
                result=result,
                elapsed_ms=elapsed_ms,
                failure=f"Judge schema validation failed: {exc}",
            )

        judge_model, judge_provider = self._observed_provenance(result)

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
            judge_metadata=result.metadata,
        )

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
        except Exception:
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
            try:
                validated = AnswerJudgeResult.model_validate(
                    raw_verdict
                )

                verdicts.append(
                    JudgeVerdict(
                        result=validated,
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

            except Exception as exc:
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
                            f"{exc}"
                        ),
                        judge_metadata=result.metadata,
                    )
                )

        return verdicts

    @staticmethod
    def _system_error(
        *,
        result: GenerationResult,
        elapsed_ms: float,
        failure: str,
    ) -> JudgeVerdict:
        judge_model, judge_provider = AnswerJudge._observed_provenance(
            result
        )

        return JudgeVerdict(
            result=None,
            status=JudgeStatus.SYSTEM_ERROR,
            judge_model=judge_model,
            judge_provider=judge_provider,
            judge_latency_ms=elapsed_ms,
            judge_tokens=result.usage,
            judge_cost_usd=result.estimated_cost_usd,
            structured_output_valid=False,
            judge_failure=failure,
            judge_metadata=result.metadata,
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
