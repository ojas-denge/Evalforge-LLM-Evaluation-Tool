from unittest.mock import patch

from app.generation.base import GenerationRequest
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.retrieval.retriever import RetrievedDocument


def _structured_request() -> GenerationRequest:
    document = RetrievedDocument(
        rank=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Python is a programming language.",
        distance=0.1,
    )

    return GenerationRequest(
        question="What is Python?",
        context=[document],
        model="test-model",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    )


def test_openai_compatible_generator_parses_response():
    document = RetrievedDocument(
        rank=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Python is a programming language.",
        distance=0.1,
    )

    request = GenerationRequest(
        question="What is Python?",
        context=[document],
        model="test-model",
    )

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language.",
                    "refusal": None,
                    "reasoning": "The context directly answers the question.",
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "The context directly answers the question.",
                            "format": "unknown",
                            "index": 0,
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 8,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer == "Python is a programming language."
    assert result.structured_output is None
    assert result.structured_output_status == "not_requested"
    assert result.model == "test-model"
    assert result.provider == "openai-compatible"
    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 33
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0
    assert result.estimated_cost_usd == 0.0

    observation = result.observation

    assert observation is not None
    assert observation.requested_model == "test-model"
    assert observation.actual_model == "test-model-actual"
    assert observation.actual_provider == "TestProvider"
    assert observation.http_status == 200
    assert observation.finish_reason == "stop"
    assert observation.content == "Python is a programming language."
    assert observation.refusal is None
    assert observation.reasoning == (
        "The context directly answers the question."
    )
    assert observation.reasoning_details == [
        {
            "type": "reasoning.text",
            "text": "The context directly answers the question.",
            "format": "unknown",
            "index": 0,
        }
    ]
    assert observation.usage.input_tokens == 25
    assert observation.usage.output_tokens == 8
    assert observation.latency_ms >= 0
    assert observation.response_format_requested is None
    assert observation.raw_response == response_payload


def test_openai_compatible_generator_supports_structured_output():
    request = _structured_request()

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer":"Python is a programming language.",'
                        '"confidence":0.95}'
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer == (
        '{"answer":"Python is a programming language.",'
        '"confidence":0.95}'
    )

    assert result.structured_output == {
        "answer": "Python is a programming language.",
        "confidence": 0.95,
    }

    assert result.structured_output_status == "parsed"

    assert result.observation is not None
    assert result.observation.response_format_requested == {
        "type": "json_schema",
        "json_schema": {
            "name": "evalforge_response",
            "schema": request.response_schema,
        },
    }
    assert result.observation.content == result.answer
    assert result.observation.raw_response == response_payload

    payload = mock_post.call_args.kwargs["json"]

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "evalforge_response",
            "schema": request.response_schema,
        },
    }


def test_openai_compatible_generator_preserves_invalid_structured_json():
    request = _structured_request()

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": '{"answer":"Python is a language."'
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer == '{"answer":"Python is a language."'
    assert result.structured_output is None
    assert result.structured_output_status == "parse_failed"

    assert result.metadata["actual_model"] == "test-model-actual"
    assert result.metadata["actual_provider"] == "TestProvider"
    assert (
        result.metadata["structured_output_error"]
        == "Structured output is not valid JSON"
    )

    assert result.observation is not None
    assert result.observation.actual_model == "test-model-actual"
    assert result.observation.actual_provider == "TestProvider"
    assert result.observation.http_status == 200
    assert result.observation.content == (
        '{"answer":"Python is a language."'
    )
    assert result.observation.finish_reason == "stop"
    assert result.observation.raw_response == response_payload


def test_openai_compatible_generator_does_not_validate_schema_at_provider_boundary():
    request = _structured_request()

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer":"Python is a language.",'
                        '"confidence":"high"}'
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.structured_output == {
        "answer": "Python is a language.",
        "confidence": "high",
    }

    assert result.structured_output_status == "parsed"
    assert result.metadata["actual_model"] == "test-model-actual"
    assert result.metadata["actual_provider"] == "TestProvider"

    assert result.observation is not None
    assert result.observation.actual_model == "test-model-actual"
    assert result.observation.actual_provider == "TestProvider"
    assert result.observation.content == result.answer
    assert result.observation.raw_response == response_payload


def test_openai_compatible_generator_normalizes_fenced_json():
    request = _structured_request()

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": (
                        "```json\n"
                        '{"answer":"Python is a language.",'
                        '"confidence":0.95}'
                        "\n```"
                    )
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.structured_output == {
        "answer": "Python is a language.",
        "confidence": 0.95,
    }

    assert result.structured_output_status == "parsed"

    assert result.observation is not None
    assert result.observation.content == result.answer
    assert result.observation.raw_response == response_payload


def test_openai_compatible_generator_preserves_null_content():
    request = _structured_request()

    response_payload = {
        "model": "test-model-actual",
        "provider": "TestProvider",
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning": "The model produced reasoning.",
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "The model produced reasoning.",
                            "format": "unknown",
                            "index": 0,
                        }
                    ],
                },
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 100,
        },
    }

    with patch(
        "app.generation.openai_compatible.httpx.post"
    ) as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = response_payload

        generator = OpenAICompatibleGenerator(
            api_key="test-key",
            base_url="http://localhost:9000/v1",
            default_model="test-model",
        )

        result = generator.generate(request)

    assert result.answer is None
    assert result.structured_output is None
    assert result.structured_output_status == "missing_content"
    assert result.finish_reason == "length"
    assert result.metadata["actual_model"] == "test-model-actual"
    assert result.metadata["actual_provider"] == "TestProvider"
    assert result.metadata["content_present"] is False
    assert result.metadata["message_keys"] == [
        "content",
        "reasoning",
        "reasoning_details",
    ]

    assert result.observation is not None
    assert result.observation.actual_model == "test-model-actual"
    assert result.observation.actual_provider == "TestProvider"
    assert result.observation.http_status == 200
    assert result.observation.content is None
    assert result.observation.refusal is None
    assert result.observation.reasoning == (
        "The model produced reasoning."
    )
    assert result.observation.reasoning_details == [
        {
            "type": "reasoning.text",
            "text": "The model produced reasoning.",
            "format": "unknown",
            "index": 0,
        }
    ]
    assert result.observation.finish_reason == "length"
    assert result.observation.raw_response == response_payload
