from json import JSONDecodeError, loads
from time import perf_counter
from typing import Any

import httpx

from app.generation.base import (
    GenerationRequest,
    GenerationResult,
    Generator,
    StructuredOutputStatus,
)
from app.generation.cost import CostCalculator
from app.generation.observation import GenerationObservation
from app.generation.pricing_registry import PricingRegistry
from app.generation.usage import GenerationUsage
from app.observability.tracing import Tracer


class OpenAICompatibleGenerator(Generator):
    """Generator for APIs implementing the OpenAI-compatible chat interface."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        default_model: str,
        tracer: Tracer | None = None,
        cost_calculator: CostCalculator | None = None,
        pricing_registry: PricingRegistry | None = None,
        structured_output_mode: str = "json_schema",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.tracer = tracer or Tracer()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.structured_output_mode = structured_output_mode.lower().strip()
        self.timeout = timeout

        if self.structured_output_mode not in {
            "json_schema",
            "json_object",
        }:
            raise ValueError(
                "Unsupported structured output mode: "
                f"{structured_output_mode}"
            )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        model = request.model or self.default_model

        messages: list[dict[str, str]] = []

        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        context = "\n\n".join(
            (
                f"[{document.document_id} | "
                f"chunk={document.chunk_id} | "
                f"rank={document.rank}]\n"
                f"{document.text}"
            )
            for document in request.context
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{request.question}\n\n"
                    f"Retrieved context:\n{context}"
                ),
            }
        )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        response_format = None

        if request.response_schema is not None:
            response_format = self._build_response_format(
                request.response_schema
            )
            payload["response_format"] = response_format

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = perf_counter()

        with self.tracer.generation(
            name="generation",
            input={
                "question": request.question,
                "model": model,
                "context_count": len(request.context),
                "structured_output": (
                    request.response_schema is not None
                ),
                "structured_output_mode": self.structured_output_mode,
            },
            metadata={
                "provider": "openai-compatible",
                "model": model,
                "base_url": self.base_url,
                "structured_output_mode": self.structured_output_mode,
            },
        ) as trace_observation:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            latency_ms = (perf_counter() - start_time) * 1000.0

            actual_model = data.get("model") or model
            actual_provider = data.get("provider")

            choices = data.get("choices") or []

            if choices:
                choice = choices[0] or {}
                finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
            else:
                finish_reason = None
                message = {}

            raw_content = message.get("content")
            refusal = message.get("refusal")

            reasoning = message.get("reasoning")

            reasoning_details = message.get("reasoning_details")

            if not isinstance(reasoning_details, list):
                reasoning_details = []

            if raw_content is not None and not isinstance(
                raw_content,
                str,
            ):
                original_content_type = type(raw_content).__name__
                answer = str(raw_content)
            else:
                original_content_type = None
                answer = raw_content

            structured_output = None
            structured_output_status = (
                StructuredOutputStatus.NOT_REQUESTED
            )
            structured_output_error = None

            if request.response_schema is not None:
                if answer is None:
                    structured_output_status = (
                        StructuredOutputStatus.MISSING_CONTENT
                    )
                else:
                    try:
                        structured_output = (
                            self._parse_structured_output(answer)
                        )
                        structured_output_status = (
                            StructuredOutputStatus.PARSED
                        )
                    except ValueError as exc:
                        structured_output_status = (
                            StructuredOutputStatus.PARSE_FAILED
                        )
                        structured_output_error = str(exc)

            raw_usage = data.get("usage") or {}

            usage = GenerationUsage(
                input_tokens=int(
                    raw_usage.get("prompt_tokens", 0)
                ),
                output_tokens=int(
                    raw_usage.get("completion_tokens", 0)
                ),
            )

            pricing = self.pricing_registry.get(
                provider="openai-compatible",
                model=model,
            )

            estimated_cost_usd = self.cost_calculator.calculate(
                usage=usage,
                pricing=pricing,
            )

            observation = GenerationObservation(
                requested_model=model,
                actual_model=actual_model,
                actual_provider=actual_provider,
                http_status=response.status_code,
                finish_reason=finish_reason,
                content=answer,
                refusal=refusal,
                reasoning=reasoning,
                reasoning_details=reasoning_details,
                usage=usage,
                latency_ms=latency_ms,
                response_format_requested=response_format,
                raw_response=data,
                metadata={
                    "structured_output_requested": (
                        request.response_schema is not None
                    ),
                    "structured_output_mode": (
                        self.structured_output_mode
                    ),
                    "structured_output_status": (
                        structured_output_status.value
                    ),
                    "structured_output_error": (
                        structured_output_error
                    ),
                    "content_present": answer is not None,
                    "message_keys": list(message.keys()),
                    "original_content_type": (
                        original_content_type
                    ),
                },
            )

            result_metadata = {
                "usage": raw_usage,
                "structured_output_mode": self.structured_output_mode,
                "requested_model": model,
                "actual_model": actual_model,
                "actual_provider": actual_provider,
                "provider_response": data,
                "structured_output_error": structured_output_error,
                "finish_reason": finish_reason,
                "content_present": answer is not None,
                "message_keys": list(message.keys()),
                "original_content_type": original_content_type,
                "generation_observation": observation,
            }

            result = GenerationResult(
                answer=answer,
                model=model,
                provider="openai-compatible",
                usage=usage,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                structured_output=structured_output,
                structured_output_status=structured_output_status,
                observation=observation,
                metadata=result_metadata,
            )

            if trace_observation is not None:
                trace_observation.update(
                    output={
                        "answer": answer,
                        "structured_output": structured_output,
                    },
                    metadata={
                        "provider": result.provider,
                        "model": result.model,
                        "requested_model": model,
                        "actual_model": actual_model,
                        "actual_provider": actual_provider,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "estimated_cost_usd": (
                            result.estimated_cost_usd
                        ),
                        "latency_ms": latency_ms,
                        "structured_output": (
                            request.response_schema is not None
                        ),
                        "structured_output_mode": (
                            self.structured_output_mode
                        ),
                        "structured_output_status": (
                            structured_output_status.value
                        ),
                        "finish_reason": finish_reason,
                    },
                )

            return result

    def _build_response_format(
        self,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if self.structured_output_mode == "json_object":
            return {
                "type": "json_object",
            }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "evalforge_response",
                "schema": schema,
            },
        }

    @staticmethod
    def _parse_structured_output(
        content: str,
    ) -> dict[str, Any]:
        normalized = content.strip()

        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()

            if lines and lines[0].strip().lower() in {
                "```json",
                "```json5",
                "```",
            }:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            normalized = "\n".join(lines).strip()

        try:
            parsed = loads(normalized)
        except JSONDecodeError as exc:
            raise ValueError(
                "Structured output is not valid JSON"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Structured output must be a JSON object"
            )

        return parsed


