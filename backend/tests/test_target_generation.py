import json
import logging
import pytest
import httpx
from app.ai.gemma_generator import (
    GemmaModelUnavailable,
    GemmaRateLimited,
    GemmaTargetGenerator,
    GemmaTargetList,
    TargetGenerationError,
)
from app.config import settings
from app.models.location_context import LocationContext
from app.models.target import OUTDOOR_CANDIDATE_CLASSES, YOLO_SUPPORTED_CLASSES


def target_set(classes):
    return [
        {
            "class_name": class_name,
            "hint": "Find it in a public outdoor area.",
        }
        for class_name in classes
    ]


class FakeResponse:
    def __init__(self, content=None, *, status_code=200, error=None):
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://ollama.test/api/chat")
        self._payload = (
            {"error": error}
            if error is not None
            else {"message": {"content": content}}
        )

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                    json=self._payload,
                ),
            )


class FakeOpenRouterResponse(FakeResponse):
    def __init__(self, content=None, *, status_code=200, error=None):
        self.status_code = status_code
        self.request = httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self._payload = (
            {"error": {"message": error}}
            if error is not None
            else {"choices": [{"message": {"content": content}}]}
        )


class FakeOllamaClient:
    responses = []
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, json, **kwargs):
        self.requests.append({"url": url, "body": json, "kwargs": kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def configure_fake_ollama(monkeypatch, responses):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    FakeOllamaClient.responses = list(responses)
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient", FakeOllamaClient
    )


def valid_response(classes):
    return FakeResponse(json.dumps({"targets": target_set(classes)}))


def context():
    return LocationContext(
        locality="Northside",
        region="Test Region",
        environment_type="suburban",
        nearby_categories=["parks_and_recreation", "public_transport"],
        nearby_feature_counts={"parks_and_recreation": 2, "public_transport": 1},
        context_mode="contextual",
        context_status="available",
    )


@pytest.mark.asyncio
async def test_valid_five_target_structured_response_uses_chat_schema(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [valid_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )
    generator = GemmaTargetGenerator()
    generator.model = "gemma3:1b"

    targets = await generator.generate_targets("game-a", context())

    request = FakeOllamaClient.requests[0]
    schema = request["body"]["format"]
    assert request["url"] == "http://localhost:11434/api/chat"
    assert request["body"]["model"] == "gemma3:1b"
    assert request["body"]["stream"] is False
    expected_schema = GemmaTargetList.model_json_schema()
    expected_schema["$defs"]["GemmaTarget"]["properties"]["class_name"]["enum"] = (
        schema["$defs"]["GemmaTarget"]["properties"]["class_name"]["enum"]
    )
    expected_schema["properties"]["targets"]["minItems"] = 5
    expected_schema["properties"]["targets"]["maxItems"] = 5
    assert schema == expected_schema
    assert schema["properties"]["targets"]["minItems"] == 5
    assert schema["properties"]["targets"]["maxItems"] == 5
    item_schema = schema["$defs"]["GemmaTarget"]
    assert set(item_schema["required"]) == {"class_name", "hint"}
    assert set(item_schema["properties"]["class_name"]["enum"]).issubset(
        set(OUTDOOR_CANDIDATE_CLASSES)
    )
    assert len(item_schema["properties"]["class_name"]["enum"]) >= 5
    assert len(targets) == 5
    assert all(target.class_name in YOLO_SUPPORTED_CLASSES for target in targets)
    assert all(target.id for target in targets)
    assert set(schema["$defs"]["GemmaTarget"]["properties"]["class_name"]["enum"]) \
        <= set(OUTDOOR_CANDIDATE_CLASSES)


@pytest.mark.asyncio
async def test_gemma_cannot_supply_a_misleading_display_title(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            FakeResponse(
                json.dumps(
                    {
                        "targets": [
                            {
                                "name": "Oak tree",
                                "class_name": "potted plant",
                                "hint": "Look nearby.",
                            },
                            *target_set(["bench", "bird", "car", "backpack"]),
                        ]
                    }
                )
            ),
            valid_response(["bicycle", "bench", "bird", "car", "backpack"]),
        ],
    )

    generator = GemmaTargetGenerator()
    targets = await generator.generate_targets(
        "game-canonical-title", context()
    )

    assert [(target.name, target.class_name) for target in targets] == [
        (class_name.title(), class_name)
        for class_name in ("bicycle", "bench", "bird", "car", "backpack")
    ]
    assert len(FakeOllamaClient.requests) == 2
    schema_fields = set(
        FakeOllamaClient.requests[0]["body"]["format"]["$defs"]["GemmaTarget"][
            "properties"
        ]
    )
    assert schema_fields == {"class_name", "hint"}


@pytest.mark.asyncio
async def test_generation_is_restricted_to_classes_exposed_by_loaded_weights(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [valid_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )
    supported = {"bicycle", "bench", "bird", "car", "backpack"}

    targets = await GemmaTargetGenerator().generate_targets(
        "game-filtered-vocabulary",
        context(),
        supported_classes=supported,
    )

    allowed = set(
        FakeOllamaClient.requests[0]["body"]["format"]["$defs"]["GemmaTarget"][
            "properties"
        ]["class_name"]["enum"]
    )
    assert allowed == supported
    assert {target.class_name for target in targets} <= supported


@pytest.mark.asyncio
async def test_generator_can_select_one_unique_replacement_class(monkeypatch):
    configure_fake_ollama(monkeypatch, [valid_response(["bench"])])

    targets = await GemmaTargetGenerator().generate_targets(
        "game-one-replacement",
        context(),
        count=1,
        excluded_classes={"backpack", "bicycle"},
        supported_classes={"bench", "bird", "dog", "car", "bus"},
    )

    schema = FakeOllamaClient.requests[0]["body"]["format"]
    assert schema["properties"]["targets"]["minItems"] == 1
    assert schema["properties"]["targets"]["maxItems"] == 1
    assert [target.class_name for target in targets] == ["bench"]


@pytest.mark.parametrize(
    "classes",
    [
        ["bicycle", "bench", "bird", "car"],
        ["bicycle", "bench", "bird", "car", "backpack", "bus"],
    ],
    ids=["fewer-than-five", "more-than-five"],
)
def test_schema_validation_rejects_non_five_target_payloads(classes):
    with pytest.raises(TargetGenerationError, match="strict five-target schema"):
        GemmaTargetGenerator._validate_targets({"targets": target_set(classes)}, 5)


def test_duplicate_target_classes_are_rejected():
    targets = target_set(["bicycle", "bicycle", "bird", "car", "backpack"])
    with pytest.raises(TargetGenerationError, match="strict five-target schema"):
        GemmaTargetGenerator._validate_targets({"targets": targets}, 5)


def test_invalid_yolo_class_is_rejected_by_schema():
    targets = target_set(["bicycle", "bench", "bird", "car", "book"])
    with pytest.raises(TargetGenerationError, match="strict five-target schema"):
        GemmaTargetGenerator._validate_targets({"targets": targets}, 5)


@pytest.mark.asyncio
async def test_malformed_json_retries_and_logs_safe_diagnostics(monkeypatch, caplog):
    configure_fake_ollama(
        monkeypatch,
        [
            FakeResponse("{not-json"),
            valid_response(["bus", "dog", "umbrella", "bottle", "kite"]),
        ],
    )

    with caplog.at_level(logging.INFO):
        targets = await GemmaTargetGenerator().generate_targets("game-retry", context())

    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2
    assert FakeOllamaClient.requests[0]["body"]["options"]["temperature"] == 0.2
    assert FakeOllamaClient.requests[1]["body"]["options"]["temperature"] == 0.0
    assert "exactly five unique targets" in FakeOllamaClient.requests[1]["body"]["messages"][0]["content"]
    assert "json_valid=False" in caplog.text
    assert "{not-json" not in caplog.text


@pytest.mark.asyncio
async def test_validation_diagnostics_include_count_without_model_content(monkeypatch, caplog):
    configure_fake_ollama(
        monkeypatch,
        [
            valid_response(["bicycle", "bench", "bird", "car"]),
            valid_response(["bus", "dog", "umbrella", "bottle", "kite"]),
        ],
    )

    with caplog.at_level(logging.INFO):
        await GemmaTargetGenerator().generate_targets("game-count-diagnostic", context())

    assert "status=200" in caplog.text
    assert "field_parsed=true" in caplog.text
    assert "json_valid=True" in caplog.text
    assert "top_level_keys=['targets']" in caplog.text
    assert "target_count=4" in caplog.text
    assert "Find it in a public outdoor area." not in caplog.text


@pytest.mark.asyncio
async def test_ollama_unavailable_retries_once_then_succeeds(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            valid_response(["bicycle", "bench", "bird", "car", "backpack"]),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("game-offline-retry", context())

    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_missing_configured_model_has_pull_instruction(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            FakeResponse(
                status_code=404,
                error="model 'gemma3:missing' not found, try pulling it first",
            )
        ],
    )
    generator = GemmaTargetGenerator()
    generator.model = "gemma3:missing"

    with pytest.raises(GemmaModelUnavailable, match="ollama pull gemma3:missing"):
        await generator.generate_targets("game-model-missing", context())

    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_both_generation_attempts_failing_returns_clear_error(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            valid_response(["bicycle", "bench", "bird", "car"]),
            valid_response(["bicycle", "bench", "bird", "car", "backpack", "bus"]),
        ],
    )

    with pytest.raises(
        TargetGenerationError,
        match="unavailable or returned invalid output after two attempts",
    ):
        await GemmaTargetGenerator().generate_targets("game-both-fail", context())

    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_location_context_and_game_signal_reach_prompt(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [valid_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    await GemmaTargetGenerator().generate_targets("game-context", context())

    prompt = FakeOllamaClient.requests[0]["body"]["messages"][0]["content"]
    assert "Northside" in prompt
    assert "suburban" in prompt
    assert "parks_and_recreation" in prompt
    assert '"parks_and_recreation": 2' in prompt
    assert "public transport nearby" in prompt
    assert "difficulty" in prompt
    assert "game-context" in prompt
    assert "43.653226" not in prompt
    assert "-79.383184" not in prompt
    assert "Return exactly five targets. Do not include any additional text." in prompt


@pytest.mark.asyncio
async def test_unavailable_map_context_is_identified_as_general_fallback(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [valid_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )
    unavailable = LocationContext(
        context_mode="general",
        context_status="unavailable",
        notice="Map context unavailable.",
    )

    await GemmaTargetGenerator().generate_targets("game-general", unavailable)

    prompt = FakeOllamaClient.requests[0]["body"]["messages"][0]["content"]
    assert '"context_mode": "general"' in prompt
    assert "Map context unavailable." in prompt
    assert "map evidence unavailable" in prompt


@pytest.mark.asyncio
async def test_candidate_order_is_ranked_by_environment_evidence(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            valid_response(["bicycle", "bench", "bird", "car", "backpack"]),
            valid_response(["bus", "dog", "umbrella", "bottle", "kite"]),
        ],
    )
    generator = GemmaTargetGenerator()
    park_context = LocationContext(
        environment_type="park_nature",
        nearby_categories=["parks_and_recreation"],
        nearby_feature_counts={"parks_and_recreation": 2},
        context_mode="contextual",
        context_status="available",
    )
    transport_context = LocationContext(
        environment_type="urban",
        nearby_categories=["roads_and_junctions", "public_transport"],
        nearby_feature_counts={"roads_and_junctions": 2, "public_transport": 1},
        context_mode="contextual",
        context_status="available",
    )

    park_ranked = generator.rank_candidates(park_context, "same-game")
    transport_ranked = generator.rank_candidates(transport_context, "same-game")
    assert park_ranked[0].class_name == "bench"
    assert transport_ranked[0].class_name in {"car", "bus", "traffic light"}
    assert park_ranked != transport_ranked

    await generator.generate_targets("game-1", context())
    await generator.generate_targets("game-2", context())
    prompts = [
        request["body"]["messages"][0]["content"]
        for request in FakeOllamaClient.requests
    ]
    assert prompts[0] != prompts[1]


@pytest.mark.asyncio
async def test_model_generation_does_not_silently_fallback_when_ollama_is_down(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    with pytest.raises(TargetGenerationError, match="after two attempts"):
        await GemmaTargetGenerator().generate_targets("game-no-fallback", context())

    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_openrouter_uses_secret_and_strict_json_schema(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    FakeOllamaClient.responses = [
        FakeOpenRouterResponse(
            json.dumps(
                {
                    "targets": target_set(
                        ["bicycle", "bench", "bird", "car", "backpack"]
                    )
                }
            )
        )
    ]
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    generator = GemmaTargetGenerator()
    targets = await generator.generate_targets("openrouter-game", context())

    request = FakeOllamaClient.requests[0]
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["kwargs"]["headers"] == {"Authorization": "Bearer test-secret"}
    assert request["body"]["model"] == "google/gemma-4-26b-a4b-it:free"
    assert request["body"]["stream"] is False
    assert request["body"]["response_format"]["type"] == "json_schema"
    schema = request["body"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["name"] == "gemma_target_list"
    assert schema["schema"]["properties"]["targets"]["minItems"] == 5
    assert schema["schema"]["properties"]["targets"]["maxItems"] == 5
    assert set(
        schema["schema"]["$defs"]["GemmaTarget"]["properties"]["class_name"]["enum"]
    ) <= set(OUTDOOR_CANDIDATE_CLASSES)
    assert len(targets) == 5
    assert "test-secret" not in repr(request["body"])


@pytest.mark.asyncio
async def test_openrouter_requires_key_and_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    generator = GemmaTargetGenerator()

    with pytest.raises(TargetGenerationError, match="OPENROUTER_API_KEY"):
        await generator.generate_targets("openrouter-no-key", context())

    assert FakeOllamaClient.requests == []


@pytest.mark.asyncio
async def test_openrouter_unavailable_does_not_return_demo_targets(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    FakeOllamaClient.responses = [
        httpx.ConnectError("offline"),
        httpx.ConnectError("offline"),
    ]
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    with pytest.raises(TargetGenerationError, match="Openrouter"):
        await GemmaTargetGenerator().generate_targets(
            "openrouter-offline",
            context(),
        )

    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_openrouter_rate_limit_fails_without_immediate_retry(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    FakeOllamaClient.responses = [
        FakeOpenRouterResponse(status_code=429, error="rate limit"),
    ]
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    with pytest.raises(GemmaRateLimited, match="rate limit"):
        await GemmaTargetGenerator().generate_targets(
            "openrouter-rate-limit",
            context(),
        )

    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_demo_mode_remains_an_explicit_sample_mode(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    FakeOllamaClient.requests = []

    targets = await GemmaTargetGenerator().generate_targets("demo-game", context())

    assert len(targets) == 5
    assert FakeOllamaClient.requests == []
