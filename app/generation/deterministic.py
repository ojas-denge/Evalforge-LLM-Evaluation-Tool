from time import perf_counter

from app.generation.base import GenerationRequest, GenerationResult, Generator
from app.generation.observation import GenerationObservation
from app.generation.usage import GenerationUsage
from app.observability.tracing import Tracer


class DeterministicGenerator(Generator):
    """Deterministic generator used for local integration tests."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        start_time = perf_counter()

        model = request.model or "deterministic-test"

        with self.tracer.generation(
            name="generation",
            input={
                "question": request.question,
                "context": [
                    {
                        "chunk_id": document.chunk_id,
                        "document_id": document.document_id,
                        "rank": document.rank,
                        "text": document.text,
                    }
                    for document in request.context
                ],
            },
            metadata={
                "provider": "deterministic",
                "model": model,
            },
        ) as trace_observation:

            if request.context:
                answer = "\n\n".join(
                    document.text
                    for document in request.context
                )
            else:
                answer = "No relevant context was retrieved."

            latency_ms = (perf_counter() - start_time) * 1000

            usage = GenerationUsage(
                input_tokens=0,
                output_tokens=0,
            )

            generation_observation = GenerationObservation(
                requested_model=model,
                actual_model=model,
                actual_provider="deterministic",
                http_status=None,
                finish_reason="stop",
                content=answer,
                refusal=None,
                reasoning=None,
                reasoning_details=[],
                usage=usage,
                latency_ms=latency_ms,
                response_format_requested=None,
                raw_response=None,
                metadata={
                    "execution_type": "deterministic",
                    "structured_output_requested": (
                        request.response_schema is not None
                    ),
                },
            )

            result = GenerationResult(
                answer=answer,
                model=model,
                provider="deterministic",
                usage=usage,
                estimated_cost_usd=0.0,
                latency_ms=latency_ms,
                finish_reason="stop",
                observation=generation_observation,
            )

            if trace_observation is not None:
                trace_observation.update(
                    output={"answer": result.answer},
                    metadata={
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                        "estimated_cost_usd": result.estimated_cost_usd,
                        "latency_ms": result.latency_ms,
                        "finish_reason": result.finish_reason,
                    },
                )

            return result
