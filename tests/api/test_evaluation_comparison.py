from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
import pytest
from app.models.evaluation import EvaluationRun, RetrievalConfig


client = TestClient(app)


def make_run(run_id: str, mrr: float, latency: float) -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
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
        mean_mrr=mrr,
        mean_retrieval_latency_ms=latency,
    )


def test_compare_evaluations(monkeypatch):
    baseline_id = str(uuid4())
    candidate_id = str(uuid4())

    baseline = make_run(
        baseline_id,
        mrr=0.90,
        latency=20.0,
    )

    candidate = make_run(
        candidate_id,
        mrr=0.95,
        latency=50.0,
    )

    def get_run(self, run_id):
        if run_id == baseline_id:
            return baseline
        if run_id == candidate_id:
            return candidate
        return None

    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.get_run",
        get_run,
    )

    response = client.get(
        f"/evaluations/{baseline_id}/compare/{candidate_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["baseline_run_id"] == baseline_id
    assert data["candidate_run_id"] == candidate_id
    assert data["metric_deltas"]["mean_mrr"] == pytest.approx(0.05)
    assert data["metric_deltas"]["mean_retrieval_latency_ms"] == pytest.approx(30.0)


def test_compare_evaluations_baseline_not_found(monkeypatch):
    candidate_id = str(uuid4())

    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.get_run",
        lambda self, run_id: None,
    )

    response = client.get(
        f"/evaluations/00000000-0000-0000-0000-000000000000"
        f"/compare/{candidate_id}"
    )

    assert response.status_code == 404


def test_compare_evaluations_candidate_not_found(monkeypatch):
    baseline_id = str(uuid4())

    baseline = make_run(
        baseline_id,
        mrr=0.90,
        latency=20.0,
    )

    def get_run(self, run_id):
        if run_id == baseline_id:
            return baseline
        return None

    monkeypatch.setattr(
        "app.db.repository.EvaluationRepository.get_run",
        get_run,
    )

    response = client.get(
        f"/evaluations/{baseline_id}/compare/"
        f"00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
