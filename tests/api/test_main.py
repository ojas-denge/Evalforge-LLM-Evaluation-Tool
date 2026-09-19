import app.main as main_module

from fastapi.testclient import TestClient

from app.generation.base import GenerationResult
from app.generation.usage import GenerationUsage


client = TestClient(main_module.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "evalforge"
    assert data["version"] == "0.1.0"


def test_query_contract():
    response = client.post(
        "/query",
        json={"question": "What is EvalForge?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    assert data["confidence"] is None
    assert data["latency_ms"] >= 0.0

    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] >= 0
    assert isinstance(data["output_tokens"], int)
    assert data["output_tokens"] >= 0
    assert data["estimated_cost_usd"] == 0.0


def test_query_returns_retrieval_citations():
    response = client.post(
        "/query",
        json={"question": "What is EvalForge?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["citations"]) > 0
    assert all(isinstance(citation, str) for citation in data["citations"])


def test_query_propagates_generation_usage_and_cost(monkeypatch):
    class FakeGenerator:
        def generate(self, request):
            return GenerationResult(
                answer="EvalForge is an LLM evaluation platform.",
                model="test-model",
                provider="test-provider",
                usage=GenerationUsage(
                    input_tokens=120,
                    output_tokens=35,
                ),
                estimated_cost_usd=0.0042,
                latency_ms=12.5,
                finish_reason="stop",
            )

    monkeypatch.setattr(
        main_module,
        "generator",
        FakeGenerator(),
    )

    response = client.post(
        "/query",
        json={"question": "What is EvalForge?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "EvalForge is an LLM evaluation platform."
    assert data["input_tokens"] == 120
    assert data["output_tokens"] == 35
    assert data["estimated_cost_usd"] == 0.0042
    assert data["latency_ms"] >= 0.0
    assert data["confidence"] is None


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_query_rejects_missing_question():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422
