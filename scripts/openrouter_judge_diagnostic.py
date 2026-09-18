from app.core.config import get_settings
from app.evaluation.answer_judge import AnswerJudge
from app.generation.factory import create_generator
from app.generation.base import GenerationRequest
from app.models.evaluation import RetrievedEvidence
from app.evaluation.answer_judge import ANSWER_JUDGE_SCHEMA, ANSWER_JUDGE_SYSTEM_PROMPT


def main() -> None:
    settings = get_settings()

    print("=" * 80)
    print("OPENROUTER JUDGE DIAGNOSTIC")
    print("=" * 80)
    print(f"Requested model: {settings.llm_model}")
    print(f"Structured:      {settings.llm_structured_output_mode}")
    print()

    generator = create_generator(settings)

    question = "What factors should be considered when selecting a model?"
    expected_answer = (
        "Model selection should consider task complexity, "
        "latency, reliability, and cost."
    )
    expected_topics = [
        "task complexity",
        "latency",
        "reliability",
        "cost",
    ]
    generated_answer = expected_answer

    evidence = (
        "[routing_001 | chunk=routing_001_001 | rank=1]\n"
        "Model selection should consider task complexity, "
        "latency, reliability, and cost."
    )

    request = GenerationRequest(
        question=(
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Expected topics:\n{expected_topics}\n\n"
            f"Generated answer:\n{generated_answer}\n\n"
            f"Retrieved evidence:\n{evidence}"
        ),
        context=[],
        model=None,
        temperature=0.0,
        system_prompt=ANSWER_JUDGE_SYSTEM_PROMPT,
        response_schema=ANSWER_JUDGE_SCHEMA,
    )

    try:
        result = generator.generate(request)

        print("GENERATION SUCCEEDED")
        print("-" * 80)
        print(f"Requested model: {result.metadata.get('requested_model')}")
        print(f"Actual model:    {result.metadata.get('actual_model')}")
        print(f"Actual provider: {result.metadata.get('actual_provider')}")
        print(f"Valid JSON:      {result.structured_output is not None}")
        print(f"Tokens:          {result.usage.total_tokens}")
        print(f"Latency:         {result.latency_ms:.2f} ms")
        print()
        print("RAW CONTENT:")
        print(result.answer)

    except Exception as exc:
        print("GENERATION FAILED")
        print("-" * 80)
        print(f"{type(exc).__name__}: {exc}")

    finally:
        generator.tracer.flush()

    print("=" * 80)


if __name__ == "__main__":
    main()
