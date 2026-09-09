from fastapi.testclient import TestClient

from app.main import app, settings
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


def test_configured_frontend_origin_is_allowed(monkeypatch) -> None:
    monkeypatch.setattr("app.main.check_connection", mongo_unavailable)
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/system/algorithm",
            headers={
                "Origin": settings.frontend_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == settings.frontend_origin


def test_production_frontend_origin_is_allowed(monkeypatch) -> None:
    production_origin = "https://hsg-mt.netlify.app"
    monkeypatch.setattr("app.main.check_connection", mongo_unavailable)
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/system/algorithm",
            headers={
                "Origin": production_origin,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == production_origin
