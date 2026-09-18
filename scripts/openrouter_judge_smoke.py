from app.core.config import get_settings
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.judge_result import JudgeStatus
from app.generation.factory import create_generator
from app.models.evaluation import RetrievedEvidence


def main() -> None:
    settings = get_settings()

    print("=" * 80)
    print("OPENROUTER REAL JUDGE SMOKE TEST")
    print("=" * 80)
    print(f"Requested model: {settings.llm_model}")
    print(f"Structured:      {settings.llm_structured_output_mode}")
    print()

    generator = create_generator(settings)
    judge = AnswerJudge(generator=generator)

    question = (
        "What factors should be considered when selecting a model?"
    )

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

    generated_answer = (
        "Model selection should consider task complexity, "
        "latency, reliability, and cost."
    )

    retrieved_evidence = [
        RetrievedEvidence(
            rank=1,
            chunk_id="routing_001_001",
            document_id="routing_001",
            distance=0.1,
            text=(
                "Model selection should consider task complexity, "
                "latency, reliability, and cost."
            ),
        )
    ]

    try:
        verdict = judge.judge(
            question=question,
            expected_answer=expected_answer,
            expected_topics=expected_topics,
            generated_answer=generated_answer,
            retrieved_evidence=retrieved_evidence,
        )

        print("JUDGE COMPLETED")
        print("-" * 80)
        print(f"Status:             {verdict.status.value}")
        print(f"Valid:              {verdict.structured_output_valid}")
        print(f"Requested model:    {verdict.judge_model}")
        print(f"Adapter provider:   {verdict.judge_provider}")
        print(
            f"Actual model:       "
            f"{verdict.judge_metadata.get('actual_model')}"
        )
        print(
            f"Actual provider:    "
            f"{verdict.judge_metadata.get('actual_provider')}"
        )
        print(f"Tokens:             {verdict.judge_tokens.total_tokens}")
        print(f"Cost:               ${verdict.judge_cost_usd:.8f}")
        print(f"Latency:            {verdict.judge_latency_ms:.2f} ms")
        print(f"Failure:            {verdict.judge_failure}")

        if verdict.status == JudgeStatus.VALID:
            if verdict.result is None:
                print()
                print("INVARIANT VIOLATION: VALID judge has no result")
            else:
                print(f"Correct:            {verdict.result.answer_correct}")
                print(f"Grounded:           {verdict.result.answer_grounded}")
                print(f"Covered:            {verdict.result.topics_covered}")
                print(f"Missing:            {verdict.result.topics_missing}")
                print(
                    f"Unsupported:        "
                    f"{verdict.result.unsupported_claims}"
                )
                print(f"Reasoning:          {verdict.result.reasoning}")

        elif verdict.status == JudgeStatus.SYSTEM_ERROR:
            print()
            print("JUDGE SYSTEM ERROR")
            print("-" * 80)
            print(
                "The judge did not produce a usable semantic judgment."
            )
            print(f"Error:              {verdict.judge_failure}")

    except Exception as exc:
        print("JUDGE REQUEST FAILED")
        print("-" * 80)
        print(f"{type(exc).__name__}: {exc}")

    finally:
        generator.tracer.flush()

    print("=" * 80)


if __name__ == "__main__":
    main()
