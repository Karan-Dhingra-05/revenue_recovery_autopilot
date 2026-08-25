"""
Tests for GET /api/health.

These tests mock both the database session and the Redis client so they run
without any live infrastructure. They validate:
  - the endpoint is reachable (HTTP 200)
  - the response always contains the required keys
  - degraded states are reported correctly when individual services fail
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _healthy_db() -> MagicMock:
    """Return a mock Session whose execute() call succeeds."""
    return MagicMock(spec=Session)


def _unhealthy_db(message: str = "connection refused") -> MagicMock:
    """Return a mock Session whose execute() call raises an exception."""
    mock = MagicMock(spec=Session)
    mock.execute.side_effect = Exception(message)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Guarantee dependency overrides are cleaned up after every test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_returns_200_when_all_services_healthy(client: TestClient) -> None:
    """Health endpoint returns HTTP 200 when DB and Redis are healthy."""
    app.dependency_overrides[get_db] = lambda: _healthy_db()

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.return_value.ping.return_value = True
        response = client.get("/api/health")

    assert response.status_code == 200


def test_health_status_ok_when_all_services_healthy(client: TestClient) -> None:
    """status field is 'ok' when both DB and Redis respond correctly."""
    app.dependency_overrides[get_db] = lambda: _healthy_db()

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.return_value.ping.return_value = True
        response = client.get("/api/health")

    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["redis"] == "ok"


def test_health_response_always_contains_required_keys(client: TestClient) -> None:
    """Response always includes status, db, and redis — even on failure."""
    app.dependency_overrides[get_db] = lambda: _unhealthy_db()

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.side_effect = Exception("redis unreachable")
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert {"status", "db", "redis"}.issubset(body.keys())


def test_health_degraded_when_db_fails(client: TestClient) -> None:
    """status is 'degraded' and db contains an error string when DB is down."""
    app.dependency_overrides[get_db] = lambda: _unhealthy_db("db connection refused")

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.return_value.ping.return_value = True
        response = client.get("/api/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["db"]
    assert body["redis"] == "ok"


def test_health_degraded_when_redis_fails(client: TestClient) -> None:
    """status is 'degraded' and redis contains an error string when Redis is down."""
    app.dependency_overrides[get_db] = lambda: _healthy_db()

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.return_value.ping.side_effect = Exception("redis not available")
        response = client.get("/api/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] == "ok"
    assert "error" in body["redis"]


def test_health_degraded_when_both_services_fail(client: TestClient) -> None:
    """status is 'degraded' when both DB and Redis are unavailable."""
    app.dependency_overrides[get_db] = lambda: _unhealthy_db()

    with patch("app.api.health.redis_lib.from_url") as mock_factory:
        mock_factory.side_effect = Exception("redis unreachable")
        response = client.get("/api/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert "error" in body["db"]
    assert "error" in body["redis"]
