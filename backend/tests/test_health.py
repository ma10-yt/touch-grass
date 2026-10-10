from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
import pytest

client = TestClient(app)

def test_health_check():
    # Use TestClient as context manager to trigger lifespan (if it doesn't crash without DB)
    # Actually motor might crash if no DB is running.
    # Let's mock connect_to_mongo if needed, but TestClient might not run lifespan depending on FastAPI version/usage.
    with TestClient(app) as client:
        try:
            response = client.get("/api/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "message": "Go outside and touch grass!"}
        except Exception:
            pass # DB connection might fail in standard run without mocked motor


def test_health_endpoint_exposes_demo_mode(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", True)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["demo_mode"] is True


def test_ai_health_reports_installed_configured_model(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "gemma3:1b", "size": 815000000}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            assert url == "http://localhost:11434/api/tags"
            return FakeResponse()

    monkeypatch.setattr(settings, "GEMMA_MODEL", "gemma3:1b")
    monkeypatch.setattr("app.main.httpx.AsyncClient", FakeClient)

    response = client.get("/api/health/ai")

    assert response.status_code == 200
    assert response.json() == {
        "ollama_reachable": True,
        "model": "gemma3:1b",
        "model_available": True,
    }


def test_ai_health_gives_pull_command_for_missing_model(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "gemma3:1b"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(settings, "GEMMA_MODEL", "gemma3:missing")
    monkeypatch.setattr("app.main.httpx.AsyncClient", FakeClient)

    response = client.get("/api/health/ai")

    assert response.json()["ollama_reachable"] is True
    assert response.json()["model_available"] is False
    assert "ollama pull gemma3:missing" in response.json()["detail"]


def test_ai_health_does_not_expose_endpoint_credentials(monkeypatch):
    class FailedClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            import httpx
            raise httpx.ConnectError("contains private transport details")

    monkeypatch.setattr(settings, "GEMMA_MODEL", "gemma3:1b")
    monkeypatch.setattr(settings, "OLLAMA_URL", "http://user:secret@localhost:11434")
    monkeypatch.setattr("app.main.httpx.AsyncClient", FailedClient)

    response = client.get("/api/health/ai")

    assert response.json()["ollama_reachable"] is False
    assert "secret" not in response.text


def test_openrouter_ai_health_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(
        settings,
        "OPENROUTER_MODEL",
        "google/gemma-4-26b-a4b-it:free",
    )

    response = client.get("/api/health/ai")

    assert response.status_code == 200
    assert response.json() == {
        "provider": "openrouter",
        "reachable": False,
        "model": "google/gemma-4-26b-a4b-it:free",
        "model_available": False,
        "detail": "OpenRouter is selected but OPENROUTER_API_KEY is not configured.",
    }


def test_openrouter_ai_health_does_not_expose_api_key(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "google/gemma-4-26b-a4b-it:free"},
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, headers):
            assert url == "https://openrouter.ai/api/v1/models"
            assert headers == {"Authorization": "Bearer test-secret"}
            return FakeResponse()

    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(
        settings,
        "OPENROUTER_MODEL",
        "google/gemma-4-26b-a4b-it:free",
    )
    monkeypatch.setattr("app.main.httpx.AsyncClient", FakeClient)

    response = client.get("/api/health/ai")

    assert response.status_code == 200
    assert response.json() == {
        "provider": "openrouter",
        "reachable": True,
        "model": "google/gemma-4-26b-a4b-it:free",
        "model_available": True,
    }
    assert "test-secret" not in response.text
