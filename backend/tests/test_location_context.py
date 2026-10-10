import json
import httpx
import pytest
from app.services.location_context_service import LocationContextService


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeHTTPClient:
    def __init__(self, *args, **kwargs):
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def get(self, url, params):
        self.requests.append(("get", url, params))
        return FakeResponse(
            {
                "address": {"city": "Toronto", "state": "Ontario"},
                "category": "place",
                "type": "city",
            }
        )

    async def post(self, url, data):
        self.requests.append(("post", url, data))
        return FakeResponse(
            {
                "elements": [
                    {"tags": {"leisure": "park"}},
                    {"tags": {"natural": "wood"}},
                    {"tags": {"highway": "bus_stop"}},
                    {"tags": {"highway": "footway"}},
                    {"tags": {"landuse": "residential"}},
                ]
            }
        )


@pytest.mark.asyncio
async def test_resolve_uses_coarse_coordinates_and_categories(monkeypatch):
    clients = []

    def make_client(*args, **kwargs):
        client = FakeHTTPClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("app.services.location_context_service.httpx.AsyncClient", make_client)

    context = await LocationContextService().resolve(43.653226, -79.383184)

    assert len(clients[0].requests) == 2
    reverse_params = clients[0].requests[0][2]
    assert reverse_params["lat"] == 43.653
    assert reverse_params["lon"] == -79.383
    assert context.locality == "Toronto"
    assert context.region == "Ontario"
    assert context.environment_type == "urban"
    assert context.nearby_categories == [
        "natural_areas",
        "parks_and_recreation",
        "pedestrian_paths",
        "public_transport",
        "residential_areas",
    ]
    assert context.nearby_feature_counts["pedestrian_paths"] == 1
    assert context.nearby_feature_counts["residential_areas"] == 1
    assert context.context_mode == "contextual"
    assert context.context_provider == "openstreetmap"
    assert context.context_status == "available"
    assert "43.653226" not in json.dumps(context.model_dump())


@pytest.mark.asyncio
async def test_missing_location_is_reported_as_unavailable(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("No lookup should run without coordinates")

    monkeypatch.setattr("app.services.location_context_service.httpx.AsyncClient", fail_if_called)

    context = await LocationContextService().resolve(None, None)

    assert context.context_status == "unavailable"
    assert context.environment_type == "unknown"
    assert context.notice


@pytest.mark.asyncio
async def test_lookup_failure_is_explicit_and_degraded(monkeypatch):
    class FailedHTTPClient(FakeHTTPClient):
        async def get(self, url, params):
            raise httpx.ConnectError("request failed")

        async def post(self, url, data):
            raise httpx.ConnectError("request failed")

    monkeypatch.setattr("app.services.location_context_service.httpx.AsyncClient", FailedHTTPClient)

    context = await LocationContextService().resolve(10.0, 20.0)

    assert context.context_status == "unavailable"
    assert context.environment_type == "unknown"
    assert "general outdoor context" in context.notice


@pytest.mark.asyncio
async def test_resolve_reuses_context_cache_for_same_coarse_location(monkeypatch):
    clients = []

    def make_client(*args, **kwargs):
        client = FakeHTTPClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("app.services.location_context_service.httpx.AsyncClient", make_client)
    service = LocationContextService()

    first = await service.resolve(43.65321, -79.38311)
    second = await service.resolve(43.65324, -79.38314)

    assert len(clients) == 1
    assert len(clients[0].requests) == 2
    assert second.nearby_feature_counts == first.nearby_feature_counts


@pytest.mark.asyncio
async def test_context_provider_urls_and_user_agent_are_configurable(monkeypatch):
    clients = []
    instances = []

    def make_client(*args, **kwargs):
        clients.append(kwargs)
        client = FakeHTTPClient(*args, **kwargs)
        instances.append(client)
        return client

    monkeypatch.setattr("app.services.location_context_service.httpx.AsyncClient", make_client)
    monkeypatch.setattr(
        "app.services.location_context_service.settings.GEOCONTEXT_NOMINATIM_URL",
        "https://geo.example/reverse",
    )
    monkeypatch.setattr(
        "app.services.location_context_service.settings.GEOCONTEXT_OVERPASS_URL",
        "https://geo.example/query",
    )
    monkeypatch.setattr(
        "app.services.location_context_service.settings.GEOCONTEXT_USER_AGENT",
        "test-agent/1.0",
    )
    service = LocationContextService()

    await service.resolve(10.0, 20.0)

    assert clients[0]["headers"]["User-Agent"] == "test-agent/1.0"
    assert instances[0].requests[0][1] == "https://geo.example/reverse"
    assert instances[0].requests[1][1] == "https://geo.example/query"
    assert "landuse" in instances[0].requests[1][2]["data"]
