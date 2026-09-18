from app.core.config import Settings
from app.generation.base import Generator
from app.generation.cost import CostCalculator
from app.generation.deterministic import DeterministicGenerator
from app.generation.openai_compatible import OpenAICompatibleGenerator
from app.generation.pricing_registry import PricingRegistry
from app.observability.tracing import Tracer


def create_generator(
    settings: Settings,
    tracer: Tracer | None = None,
    cost_calculator: CostCalculator | None = None,
    pricing_registry: PricingRegistry | None = None,
) -> Generator:
    """Create the configured generation provider."""

    provider = settings.llm_provider.lower().strip()

    tracer = tracer or Tracer()
    cost_calculator = cost_calculator or CostCalculator()
    pricing_registry = pricing_registry or PricingRegistry()

    if provider == "deterministic":
        return DeterministicGenerator(
            tracer=tracer,
        )

    if provider == "openai-compatible":
        if not settings.llm_base_url:
            raise ValueError(
                "LLM_BASE_URL is required for the openai-compatible provider"
            )

        return OpenAICompatibleGenerator(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_model=settings.llm_model,
            tracer=tracer,
            cost_calculator=cost_calculator,
            pricing_registry=pricing_registry,
            structured_output_mode=settings.llm_structured_output_mode,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )
