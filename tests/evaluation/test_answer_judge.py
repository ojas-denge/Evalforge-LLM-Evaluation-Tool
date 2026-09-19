from app.evaluation.answer_judge import (
    ANSWER_JUDGE_SCHEMA,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    BATCH_JUDGE_SCHEMA,
    BATCH_JUDGE_SYSTEM_PROMPT,
    AnswerJudge,
)
from app.evaluation.judge_result import JudgeStatus
from app.generation.base import (
    GenerationRequest,
    GenerationResult,
    StructuredOutputStatus,
)
from app.generation.usage import GenerationUsage
from app.models.evaluation import RetrievedEvidence


def _evidence():
    return [
        RetrievedEvidence(
            rank=1,
            chunk_id="chunk-1",
            document_id="doc-1",
            distance=0.1,
            text="The target p95 latency is below 2500 milliseconds.",
        )
    ]


def _judge_output():
    return {
        "answer_correct": True,
        "answer_grounded": True,
        "topics_covered": ["latency"],
        "topics_missing": [],
        "unsupported_claims": [],
        "reasoning": "The answer is correct and grounded.",
    }


def _batch_cases(count):
    return [
        {
            "question": f"What is the latency target? {index}",
            "expected_answer": "The target is below 2500 milliseconds.",
            "expected_topics": ["latency"],
            "generated_answer": "The target is below 2500 milliseconds.",
            "retrieved_evidence": _evidence(),
        }
        for index in range(count)
    ]


class FakeJudgeGenerator:
    def __init__(
        self,
        structured_output,
        *,
        raise_error=False,
        structured_output_status=StructuredOutputStatus.PARSED,
        answer="raw judge output",
        metadata=None,
        responses=None,
    ):
        self.structured_output = structured_output
        self.raise_error = raise_error
        self.structured_output_status = structured_output_status
        self.answer = answer
        self.metadata = metadata or {}
        self.responses = responses
        self.call_count = 0
        self.requests = []
        self.last_request = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        self.last_request = request
        self.requests.append(request)

        if self.responses is not None:
            response = self.responses[self.call_count - 1]

            if isinstance(response, Exception):
                raise response

            return response

        if self.raise_error:
            raise RuntimeError("API failure")

        return GenerationResult(
            answer=self.answer,
            model="fake-judge",
            provider="fake",
            usage=GenerationUsage(
                input_tokens=10,
                output_tokens=20,
            ),
            estimated_cost_usd=0.05,
            latency_ms=25.0,
            structured_output=self.structured_output,
            structured_output_status=self.structured_output_status,
            metadata=self.metadata,
        )


def _generation_result(
    structured_output,
    *,
    structured_output_status=StructuredOutputStatus.PARSED,
    answer="raw judge output",
):
    return GenerationResult(
        answer=answer,
        model="fake-judge",
        provider="fake",
        usage=GenerationUsage(
            input_tokens=10,
            output_tokens=20,
        ),
        estimated_cost_usd=0.05,
        latency_ms=25.0,
        structured_output=structured_output,
        structured_output_status=structured_output_status,
    )


def test_answer_judge_returns_valid_semantic_verdict():
    generator = FakeJudgeGenerator(_judge_output())
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.result is not None
    assert verdict.result.answer_correct is True
    assert verdict.result.answer_grounded is True

    request = generator.last_request

    assert request.system_prompt == ANSWER_JUDGE_SYSTEM_PROMPT
    assert request.response_schema == ANSWER_JUDGE_SCHEMA
    assert request.temperature == 0.0


def test_answer_judge_handles_missing_structured_output():
    generator = FakeJudgeGenerator(
        None,
        structured_output_status=StructuredOutputStatus.MISSING_CONTENT,
    )
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="Q",
        expected_answer="A",
        expected_topics=[],
        generated_answer="A",
        retrieved_evidence=[],
    )

    assert verdict.status == JudgeStatus.SYSTEM_ERROR
    assert verdict.result is None
    assert verdict.structured_output_valid is False
    assert verdict.judge_failure is not None
    assert "missing content" in verdict.judge_failure.lower()
    assert verdict.judge_model == "fake-judge"
    assert verdict.judge_provider == "fake"


def test_answer_judge_handles_unparseable_structured_output():
    generator = FakeJudgeGenerator(
        None,
        structured_output_status=StructuredOutputStatus.PARSE_FAILED,
        answer='{"answer_correct": true',
        metadata={
            "structured_output_error": "Structured output is not valid JSON",
        },
    )
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="Q",
        expected_answer="A",
        expected_topics=[],
        generated_answer="A",
        retrieved_evidence=[],
    )

    assert verdict.status == JudgeStatus.SYSTEM_ERROR
    assert verdict.result is None
    assert verdict.structured_output_valid is False
    assert verdict.judge_failure is not None
    assert "parsing" in verdict.judge_failure.lower()

    assert (
        verdict.judge_metadata["structured_output_error"]
        == "Structured output is not valid JSON"
    )


def test_answer_judge_handles_schema_violation():
    generator = FakeJudgeGenerator(
        {
            "answer_correct": "yes",
            "answer_grounded": True,
            "topics_covered": [],
            "topics_missing": [],
            "unsupported_claims": [],
            "reasoning": "bad",
        }
    )
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="Q",
        expected_answer="A",
        expected_topics=[],
        generated_answer="A",
        retrieved_evidence=[],
    )

    assert verdict.status == JudgeStatus.SYSTEM_ERROR
    assert verdict.result is None
    assert verdict.structured_output_valid is False
    assert verdict.judge_failure is not None
    assert "boolean" in verdict.judge_failure.lower()


def test_answer_judge_handles_generation_failure():
    generator = FakeJudgeGenerator(None, raise_error=True)
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="Q",
        expected_answer="A",
        expected_topics=[],
        generated_answer="A",
        retrieved_evidence=[],
    )

    assert verdict.status == JudgeStatus.SYSTEM_ERROR
    assert verdict.result is None
    assert verdict.structured_output_valid is False
    assert "API failure" in verdict.judge_failure


def test_answer_judge_captures_metadata():
    generator = FakeJudgeGenerator(
        _judge_output(),
        metadata={
            "actual_model": "real-model",
            "actual_provider": "real-provider",
        },
    )
    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="Q",
        expected_answer="A",
        expected_topics=[],
        generated_answer="A",
        retrieved_evidence=[],
    )

    assert verdict.judge_model == "fake-judge"
    assert verdict.judge_provider == "fake"
    assert verdict.judge_latency_ms >= 0.0
    assert verdict.judge_tokens.input_tokens == 10
    assert verdict.judge_tokens.output_tokens == 20
    assert verdict.judge_cost_usd == 0.05
    assert verdict.judge_metadata["actual_model"] == "real-model"
    assert verdict.judge_metadata["actual_provider"] == "real-provider"


def test_batch_judge_builds_structured_request_and_returns_one_verdict_per_case():
    generator = FakeJudgeGenerator(
        {
            "verdicts": [
                _judge_output(),
                _judge_output(),
            ]
        }
    )
    judge = AnswerJudge(generator)

    verdicts = judge.batch_judge(_batch_cases(2))

    assert len(verdicts) == 2
    assert all(verdict.status == JudgeStatus.VALID for verdict in verdicts)
    assert all(verdict.result is not None for verdict in verdicts)
    assert all(verdict.result.answer_correct is True for verdict in verdicts)
    assert all(verdict.result.answer_grounded is True for verdict in verdicts)

    request = generator.last_request

    assert request.system_prompt == BATCH_JUDGE_SYSTEM_PROMPT
    assert request.response_schema == BATCH_JUDGE_SCHEMA
    assert request.temperature == 0.0
    assert "Case 1" in request.question
    assert "Case 2" in request.question


def test_batch_judge_preserves_shared_batch_metadata_per_case():
    generator = FakeJudgeGenerator(
        {
            "verdicts": [
                _judge_output(),
                _judge_output(),
            ]
        }
    )
    judge = AnswerJudge(generator)

    verdicts = judge.batch_judge(_batch_cases(2))

    assert all(verdict.judge_model == "fake-judge" for verdict in verdicts)
    assert all(verdict.judge_provider == "fake" for verdict in verdicts)
    assert all(verdict.judge_cost_usd == 0.025 for verdict in verdicts)
    assert all(verdict.judge_tokens.input_tokens == 5 for verdict in verdicts)
    assert all(verdict.judge_tokens.output_tokens == 10 for verdict in verdicts)


def test_batch_judge_falls_back_to_sequential_on_batch_failure():
    generator = FakeJudgeGenerator(
        None,
        raise_error=True,
    )
    judge = AnswerJudge(generator)

    verdicts = judge.batch_judge(_batch_cases(2))

    assert len(verdicts) == 2
    assert all(verdict.status == JudgeStatus.SYSTEM_ERROR for verdict in verdicts)
    assert all(verdict.result is None for verdict in verdicts)


def test_answer_judge_retries_after_schema_failure():
    first_result = _generation_result(
        {
            "correct": True,
            "grounded": True,
            "covered_topics": ["latency"],
            "missing_topics": [],
            "unsupported_claims": [],
        }
    )

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.result is not None
    assert verdict.result.answer_correct is True
    assert generator.call_count == 2

    assert generator.requests[0].response_schema == ANSWER_JUDGE_SCHEMA
    assert generator.requests[1].response_schema == ANSWER_JUDGE_SCHEMA

    assert (
        "schema" in generator.requests[1].question.lower()
        or "valid" in generator.requests[1].question.lower()
        or "contract" in generator.requests[1].question.lower()
    )


def test_answer_judge_retries_after_missing_content():
    first_result = _generation_result(
        None,
        structured_output_status=StructuredOutputStatus.MISSING_CONTENT,
        answer=None,
    )

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.result is not None
    assert verdict.result.answer_correct is True
    assert generator.call_count == 2

    assert (
        "missing content"
        in generator.requests[1].question.lower()
    )
# Keep the existing tests exactly as they are, and add this test
# after test_answer_judge_retries_after_missing_content.

def test_answer_judge_retries_after_parse_failure():
    first_result = _generation_result(
        None,
        structured_output_status=StructuredOutputStatus.PARSE_FAILED,
        answer='{"answer_correct": true',
    )

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.result is not None
    assert verdict.result.answer_correct is True
    assert generator.call_count == 2

    assert (
        "parsing"
        in generator.requests[1].question.lower()
        or "valid"
        in generator.requests[1].question.lower()
    )
def test_answer_judge_stops_after_max_retries():
    invalid_output = {
        "correct": True,
        "grounded": True,
        "covered_topics": ["latency"],
        "missing_topics": [],
        "unsupported_claims": [],
    }

    generator = FakeJudgeGenerator(
        None,
        responses=[
            _generation_result(invalid_output),
            _generation_result(invalid_output),
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.SYSTEM_ERROR
    assert verdict.result is None
    assert verdict.structured_output_valid is False
    assert generator.call_count == 2
    assert verdict.judge_metadata["judge_attempt"] == 2
    assert (
        verdict.judge_metadata["judge_max_attempts"] == 2
    )
def test_answer_judge_records_recovery_metadata():
    first_result = _generation_result(
        {
            "correct": True,
            "grounded": True,
            "covered_topics": ["latency"],
            "missing_topics": [],
            "unsupported_claims": [],
        }
    )

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.result is not None

    assert verdict.judge_metadata["judge_attempt"] == 2
    assert verdict.judge_metadata["judge_max_attempts"] == 2
    assert verdict.judge_metadata["judge_recovered"] is True
    assert verdict.judge_metadata["initial_failure"] is not None
    assert "schema validation failed" in (
        verdict.judge_metadata["initial_failure"].lower()
    )
def test_answer_judge_records_schema_failure_type_on_recovery():
    first_result = _generation_result(
        {
            "correct": True,
            "grounded": True,
            "covered_topics": ["latency"],
            "missing_topics": [],
            "unsupported_claims": [],
        }
    )

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.judge_metadata["initial_failure_type"] == "schema_invalid"
    assert verdict.judge_metadata["judge_recovered"] is True


def test_answer_judge_records_missing_content_failure_type():
    first_result = _generation_result(None)

    second_result = _generation_result(_judge_output())

    generator = FakeJudgeGenerator(
        None,
        responses=[
            first_result,
            second_result,
        ],
    )

    judge = AnswerJudge(generator)

    verdict = judge.judge(
        question="What is the latency target?",
        expected_answer="The target is below 2500 milliseconds.",
        expected_topics=["latency"],
        generated_answer="The target is below 2500 milliseconds.",
        retrieved_evidence=_evidence(),
    )

    assert verdict.status == JudgeStatus.VALID
    assert verdict.judge_metadata["initial_failure_type"] == "missing_content"
    assert verdict.judge_metadata["judge_recovered"] is True
