from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.runner import Evaluator
from app.generation.base import GenerationResult
from app.generation.usage import GenerationUsage
from app.models.evaluation import (
    AnswerJudgeResult,
    EvaluationCase,
    RetrievedEvidence,
)


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeTracer:
    def __init__(self):
        self.calls = []

    def trace(self, name, *, input=None, metadata=None):
        tracer = self

        class TraceContext:
            def __enter__(self):
                self.observation = FakeObservation()
                tracer.calls.append(
                    {
                        "name": name,
                        "input": input,
                        "metadata": metadata,
                        "observation": self.observation,
                    }
                )
                return self.observation

            def __exit__(self, exc_type, exc, tb):
                return False

        return TraceContext()


class FakeRepository:
    def __init__(self):
        self.saved_runs = []

    def save_run(self, run):
        self.saved_runs.append(run)

    def get_run(self, run_id):
        return next(
            (run for run in self.saved_runs if run.run_id == run_id),
            None,
        )

    def list_runs(self):
        return list(self.saved_runs)


class FakeRetriever:
    mode = "dense"
    candidate_k = None
    reranking_enabled = False
    reranker_candidate_k = 10
    hybrid_retrieval_enabled = False

    def retrieve(self, query, top_k, trace_metadata=None):
        return type(
            "RetrievalResult",
            (),
            {
                "results": [
                    RetrievedEvidence(
                        rank=1,
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        distance=0.05,
                        text=(
                            "EvalForge is an LLM evaluation and "
                            "observability platform."
                        ),
                    )
                ],
                "latency_ms": 12.5,
            },
        )()


class FakeGenerator:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)

        return GenerationResult(
            answer=(
                "EvalForge is an LLM evaluation and "
                "observability platform."
            ),
            model="test-model",
            provider="test-provider",
            usage=GenerationUsage(
                input_tokens=10,
                output_tokens=8,
            ),
            estimated_cost_usd=0.001,
            latency_ms=5.0,
            finish_reason="stop",
        )


class FakeAnswerJudge:
    def __init__(self):
        self.calls = []

    def judge(
        self,
        *,
        question,
        expected_answer,
        expected_topics,
        generated_answer,
        retrieved_evidence,
    ):
        self.calls.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_topics": expected_topics,
                "generated_answer": generated_answer,
                "retrieved_evidence": retrieved_evidence,
            }
        )

        from app.evaluation.judge_result import JudgeStatus, JudgeVerdict

        return JudgeVerdict(
            status=JudgeStatus.VALID,
            result=AnswerJudgeResult(
                answer_correct=True,
                answer_grounded=True,
                topics_covered=expected_topics,
                topics_missing=[],
                unsupported_claims=[],
                reasoning=(
                    "The generated answer matches the reference "
                    "and evidence."
                ),
            ),
            judge_model="test-model",
            judge_provider="test-provider",
            judge_latency_ms=5.0,
            judge_tokens=GenerationUsage(),
            judge_cost_usd=0.0,
            structured_output_valid=True,
            judge_failure=None,
        )


    def batch_judge(self, cases):
        return [
            self.judge(
                question=case["question"],
                expected_answer=case["expected_answer"],
                expected_topics=case["expected_topics"],
                generated_answer=case["generated_answer"],
                retrieved_evidence=case["retrieved_evidence"],
            )
            for case in cases
        ]

def make_case(expected_topics=None):
    return EvaluationCase(
        case_id="test-001",
        question="What is EvalForge?",
        expected_answer=(
            "EvalForge is an LLM evaluation and "
            "observability platform."
        ),
        expected_documents=["doc-1"],
        expected_topics=expected_topics or [],
        category="general",
        difficulty="easy",
    )


def make_dataset(case):
    return type(
        "FakeDataset",
        (),
        {
            "cases": [case],
            "__len__": lambda self: 1,
        },
    )()


def test_evaluate_dataset_emits_evaluation_run_observation():
    tracer = FakeTracer()
    retriever = FakeRetriever()
    generator = FakeGenerator()
    repository = FakeRepository()

    evaluator = Evaluator(
        retriever=retriever,
        generator=generator,
        tracer=tracer,
        repository=repository,
    )

    run = evaluator.evaluate_dataset(
        make_dataset(make_case())
    )

    assert run.dataset_size == 1
    assert run.results[0].case_id == "test-001"
    assert len(repository.saved_runs) == 1
    assert repository.saved_runs[0].run_id == run.run_id

    evaluation_calls = [
        call
        for call in tracer.calls
        if call["name"] == "evaluation_run"
    ]

    assert len(evaluation_calls) == 1


def test_evaluate_dataset_retries_generation_read_timeout(
    monkeypatch,
    tmp_path,
    capsys,
):
    import httpx

    monkeypatch.setattr(
        "app.evaluation.runner.CHECKPOINT_DIR",
        tmp_path,
    )

    cooldowns = []

    monkeypatch.setattr(
        "app.evaluation.runner._cooldown",
        lambda seconds: cooldowns.append(seconds),
    )

    class TimeoutThenSuccessGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1

            if self.calls == 1:
                raise httpx.ReadTimeout(
                    "simulated read timeout"
                )

            return GenerationResult(
                answer=(
                    "EvalForge is an LLM evaluation and "
                    "observability platform."
                ),
                model="test-model",
                provider="test-provider",
                usage=GenerationUsage(
                    input_tokens=10,
                    output_tokens=8,
                ),
                estimated_cost_usd=0.001,
                latency_ms=5.0,
                finish_reason="stop",
            )

    generator = TimeoutThenSuccessGenerator()

    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=generator,
        repository=FakeRepository(),
    )

    run = evaluator.evaluate_dataset(
        make_dataset(make_case()),
        run_id="read-timeout-smoke-test",
    )

    assert len(run.results) == 1
    assert generator.calls == 2
    assert cooldowns == [60]

    result = run.results[0]

    assert result.case_id == "test-001"
    assert result.generated_answer == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )

    output = capsys.readouterr().out

    assert (
        "[generation] retryable error; retrying (1/2)"
        in output
    )

    assert not (
        tmp_path / "read-timeout-smoke-test.json"
    ).exists()


def test_evaluator_generates_answer_from_retrieved_context():
    retriever = FakeRetriever()
    generator = FakeGenerator()

    evaluator = Evaluator(
        retriever=retriever,
        generator=generator,
    )

    evaluator.evaluate_case(
        make_case(
            expected_topics=[
                "LLM evaluation",
                "observability",
            ]
        )
    )

    assert len(generator.requests) == 1

    request = generator.requests[0]

    assert request.question == "What is EvalForge?"
    assert len(request.context) == 1
    assert request.context[0].document_id == "doc-1"


def test_evaluate_case_records_topic_coverage():
    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = evaluator.evaluate_case(
        make_case(
            expected_topics=[
                "LLM evaluation",
                "observability",
            ]
        )
    )

    assert result.generated_answer == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert result.topic_coverage == 1.0


def test_evaluate_case_topic_coverage_is_none_without_topics():
    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = evaluator.evaluate_case(
        make_case()
    )

    assert result.generated_answer == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert result.topic_coverage is None


def test_evaluate_dataset_records_answer_judge_result():
    answer_judge = FakeAnswerJudge()

    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        answer_judge=answer_judge,
        repository=FakeRepository(),
    )

    run = evaluator.evaluate_dataset(
        make_dataset(
            make_case(
                expected_topics=[
                    "LLM evaluation",
                    "observability",
                ]
            )
        ),
        run_id="judge-result-test",
    )

    assert len(answer_judge.calls) == 1

    call = answer_judge.calls[0]

    assert call["question"] == "What is EvalForge?"
    assert call["expected_answer"] == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert call["expected_topics"] == [
        "LLM evaluation",
        "observability",
    ]
    assert call["generated_answer"] == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert len(call["retrieved_evidence"]) == 1
    assert call["retrieved_evidence"][0].document_id == "doc-1"

    result = run.results[0]

    assert result.answer_judge is not None
    assert result.answer_judge.answer_correct is True
    assert result.answer_judge.answer_grounded is True
    assert result.answer_judge.topics_covered == [
        "LLM evaluation",
        "observability",
    ]
    assert result.answer_judge.topics_missing == []
    assert result.answer_judge.unsupported_claims == []

    assert result.judge_verdict is not None
    assert result.judge_verdict.structured_output_valid is True




def test_checkpoint_round_trip_preserves_retry_state(
    tmp_path,
):
    from app.evaluation.runner import (
        _load_checkpoint,
        _save_checkpoint,
    )

    checkpoint_path = tmp_path / "checkpoint.json"

    retry_state = {
        "generation": {
            "test-001": 1,
        },
        "judge": {
            "1": 2,
        },
    }

    result = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    ).evaluate_case(make_case())

    _save_checkpoint(
        checkpoint_path,
        run_id="retry-state-test",
        phase="generation",
        completed_case_ids=set(),
        results={
            result.case_id: result,
        },
        retry_state=retry_state,
    )

    loaded = _load_checkpoint(checkpoint_path)

    assert loaded is not None

    (
        run_id,
        phase,
        completed_case_ids,
        results,
        loaded_retry_state,
    ) = loaded

    assert run_id == "retry-state-test"
    assert phase == "generation"
    assert completed_case_ids == set()
    assert "test-001" in results
    assert loaded_retry_state == retry_state


def test_generation_resume_uses_persisted_retry_budget(
    monkeypatch,
    tmp_path,
):
    import httpx

    monkeypatch.setattr(
        "app.evaluation.runner.CHECKPOINT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "app.evaluation.runner._cooldown",
        lambda seconds: None,
    )

    checkpoint_path = tmp_path / "resume-retry-test.json"

    class AlwaysTimeoutGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            raise httpx.ReadTimeout(
                "simulated timeout"
            )

    generator = AlwaysTimeoutGenerator()

    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=generator,
        repository=FakeRepository(),
    )

    run_id = "resume-retry-test"

    try:
        evaluator.evaluate_dataset(
            make_dataset(make_case()),
            run_id=run_id,
        )
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError(
            "Expected evaluation to exhaust retries"
        )

    assert generator.calls == 3

    from app.evaluation.runner import _load_checkpoint

    checkpoint = _load_checkpoint(
        checkpoint_path
    )

    assert checkpoint is not None
    assert checkpoint[4]["generation"][
        "test-001"
    ] == 2


def test_generation_resume_does_not_reset_retry_budget(
    monkeypatch,
    tmp_path,
):
    import httpx

    monkeypatch.setattr(
        "app.evaluation.runner.CHECKPOINT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "app.evaluation.runner._cooldown",
        lambda seconds: None,
    )

    from app.evaluation.runner import (
        _save_checkpoint,
    )
    checkpoint_path = (
        tmp_path / "persisted-budget-test.json"
    )

    _save_checkpoint(
        checkpoint_path,
        run_id="persisted-budget-test",
        phase="generation",
        completed_case_ids=set(),
        results={},
        retry_state={
            "generation": {
                "test-001": 2,
            },
            "judge": {},
        },
    )

    class CountingTimeoutGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            raise httpx.ReadTimeout(
                "simulated timeout"
            )

    generator = CountingTimeoutGenerator()

    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=generator,
        repository=FakeRepository(),
    )

    try:
        evaluator.evaluate_dataset(
            make_dataset(make_case()),
            run_id="persisted-budget-test",
        )
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError(
            "Expected persisted retry budget to be exhausted"
        )

    assert generator.calls == 1
