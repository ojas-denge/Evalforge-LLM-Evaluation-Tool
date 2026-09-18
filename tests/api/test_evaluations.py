from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.models.evaluation import EvaluationRun, RetrievalConfig


client = TestClient(app)


def make_run() -> EvaluationRun:
    return EvaluationRun(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        dataset_size=75,
        retrieval_config=RetrievalConfig(
            mode="dense",
            top_k=5,
            candidate_k=None,
            reranking_enabled=False,
            reranker_candidate_k=None,
            hybrid_retrieval_enabled=False,
        ),
        results=[],
        mean_hit_at_1=0.86,
        mean_hit_at_3=0.97,
        mean_hit_at_5=0.98,
        mean_recall_at_1=0.79,
        mean_recall_at_3=0.95,
        mean_recall_at_5=0.98,
        mean_mrr=0.92,
        mean_retrieval_latency_ms=22.0,
    )


def test_list_evaluations(monkeypatch):
    run = make_run()

    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.list_runs",
        lambda self: [run],
    )

    response = client.get("/evaluations")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["run_id"] == run.run_id
    assert data[0]["dataset_size"] == 75
    assert data[0]["mean_mrr"] == 0.92


def test_get_evaluation(monkeypatch):
    run = make_run()

    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.get_run",
        lambda self, run_id: run if run_id == run.run_id else None,
    )

    response = client.get(f"/evaluations/{run.run_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["run_id"] == run.run_id
    assert data["dataset_size"] == 75
    assert data["retrieval_config"]["mode"] == "dense"


def test_get_evaluation_not_found(monkeypatch):
    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.get_run",
        lambda self, run_id: None,
    )

    response = client.get(
        "/evaluations/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
