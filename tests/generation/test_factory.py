from app.core.config import Settings
from app.generation.deterministic import DeterministicGenerator
from app.generation.factory import create_generator
from app.generation.openai_compatible import OpenAICompatibleGenerator


def test_factory_creates_deterministic_generator():
    settings = Settings(
        _env_file=None,
        llm_provider="deterministic",
        llm_model="test-model",
    )

    generator = create_generator(settings)

    assert isinstance(generator, DeterministicGenerator)


def test_factory_creates_openai_compatible_generator():
    settings = Settings(
        _env_file=None,
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1",
    )

    generator = create_generator(settings)

    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.structured_output_mode == "json_schema"


def test_factory_passes_structured_output_mode():
    settings = Settings(
        _env_file=None,
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1",
        llm_structured_output_mode="json_object",
    )

    generator = create_generator(settings)

    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.structured_output_mode == "json_object"


def test_factory_requires_base_url_for_openai_compatible():
    settings = Settings(
        _env_file=None,
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="test-key",
    )

    try:
        create_generator(settings)
    except ValueError as exc:
        assert "LLM_BASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_factory_rejects_unsupported_provider():
    settings = Settings(
        _env_file=None,
        llm_provider="unsupported",
        llm_model="test-model",
    )

    try:
        create_generator(settings)
    except ValueError as exc:
        assert "Unsupported LLM provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
