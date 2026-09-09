from fastapi.testclient import TestClient

from app.main import app
from app.services.hsg_adapter import HSGAdapter


async def mongo_unavailable() -> bool:
    return False


def test_liveness_is_independent_of_algorithm_and_database(monkeypatch) -> None:
    monkeypatch.setattr("app.main.check_connection", mongo_unavailable)
    with TestClient(app) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_algorithm_status_matches_adapter(monkeypatch) -> None:
    monkeypatch.setattr("app.main.check_connection", mongo_unavailable)
    with TestClient(app) as client:
        response = client.get("/api/v1/system/algorithm")
    assert response.status_code == 200
    assert response.json()["status"] == HSGAdapter().status()["status"]
