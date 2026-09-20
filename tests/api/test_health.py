import app.main as main_module

from fastapi.testclient import TestClient


client = TestClient(main_module.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "evalforge",
        "version": "0.1.0",
    }


def test_readiness(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement):
            return None

    monkeypatch.setattr(
        main_module.engine,
        "connect",
        lambda: FakeConnection(),
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "evalforge",
        "version": "0.1.0",
    }


def test_readiness_returns_503_when_database_is_unavailable(monkeypatch):
    def failing_connect():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        main_module.engine,
        "connect",
        failing_connect,
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "status": "not_ready",
            "service": "evalforge",
            "version": "0.1.0",
        }
    }