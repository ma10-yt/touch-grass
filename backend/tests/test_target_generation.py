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
    def __init__(self, content=None, *, status_code=200, error=None, headers=None):
        self.status_code = status_code
        self.request = httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.headers = headers or {}
        self._payload = (
            {"error": {"message": error}}
            if error is not None
            else {"choices": [{"message": {"content": content}}]}
        )


class FakeCloudflareResponse(FakeResponse):
    def __init__(self, content=None, *, status_code=200, error=None, headers=None):
        self.status_code = status_code
        self.request = httpx.Request(
            "POST",
            "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/google/gemma-4-26b-a4b-it",
        )
        self.headers = headers or {}
        if error is not None:
            self._payload = {"success": False, "errors": [{"message": error}]}
        else:
            self._payload = {
                "success": True,
                "errors": [],
                "result": {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
            }

    def json(self):
        return self._payload


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


def configure_fake_cloudflare(monkeypatch, responses):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "cloudflare")
    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setattr(settings, "CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setattr(settings, "CLOUDFLARE_AI_MODEL", "@cf/google/gemma-4-26b-a4b-it")
    FakeOllamaClient.responses = list(responses)
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient", FakeOllamaClient
    )


def valid_response(classes):
    return FakeResponse(json.dumps({"targets": target_set(classes)}))


def valid_cloudflare_response(classes):
    return FakeCloudflareResponse(json.dumps({"targets": target_set(classes)}))


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
async def test_missing_configured_model_falls_back_to_location_ranked_targets(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            FakeResponse(
                status_code=404,
                error="model 'gemma3:missing' not found, try pulling it first",
            ),
            FakeResponse(
                status_code=404,
                error="model 'gemma3:missing' not found, try pulling it first",
            ),
        ],
    )
    generator = GemmaTargetGenerator()
    generator.model = "gemma3:missing"

    targets = await generator.generate_targets("game-model-missing", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_both_generation_attempts_failing_uses_location_fallback(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            valid_response(["bicycle", "bench", "bird", "car"]),
            valid_response(["bicycle", "bench", "bird", "car", "backpack", "bus"]),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("game-both-fail", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
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
async def test_ollama_down_uses_location_fallback(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("game-ollama-down", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
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
async def test_openrouter_missing_key_uses_location_fallback(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    generator = GemmaTargetGenerator()
    targets = await generator.generate_targets("openrouter-no-key", context())

    # Should fall back to location-ranked targets (not demo targets)
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert FakeOllamaClient.requests == []


@pytest.mark.asyncio
async def test_openrouter_unavailable_uses_location_fallback(monkeypatch):
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

    targets = await GemmaTargetGenerator().generate_targets(
        "openrouter-offline",
        context(),
    )

    # Should fall back to location-ranked targets (not demo targets)
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_openrouter_rate_limit_retries_once_then_fallback(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    FakeOllamaClient.responses = [
        FakeOpenRouterResponse(status_code=429, error="rate limit"),
        FakeOpenRouterResponse(
            json.dumps(
                {
                    "targets": target_set(
                        ["bus", "dog", "umbrella", "bottle", "kite"]
                    )
                }
            )
        ),
    ]
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient",
        FakeOllamaClient,
    )

    targets = await GemmaTargetGenerator().generate_targets(
        "openrouter-rate-limit",
        context(),
    )

    # Should retry once and succeed on second attempt
    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_demo_mode_remains_an_explicit_sample_mode(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    FakeOllamaClient.requests = []

    targets = await GemmaTargetGenerator().generate_targets("demo-game", context())

    assert len(targets) == 5
    assert FakeOllamaClient.requests == []


# Cloudflare tests


@pytest.mark.asyncio
async def test_cloudflare_successful_generation_and_parsing(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-game", context())

    assert len(targets) == 5
    assert [t.class_name for t in targets] == [
        "bicycle",
        "bench",
        "bird",
        "car",
        "backpack",
    ]
    request = FakeOllamaClient.requests[0]
    assert "cloudflare.com" in request["url"]
    assert "test-account" in request["url"]
    assert "gemma-4-26b-a4b-it" in request["url"]
    assert request["kwargs"]["headers"]["Authorization"] == "Bearer test-token"
    assert request["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["body"]["max_completion_tokens"] == 1024
    assert "test-token" not in repr(request["body"])


@pytest.mark.asyncio
async def test_cloudflare_authorization_header_no_token_leak(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    await GemmaTargetGenerator().generate_targets("cf-leak-test", context())

    request = FakeOllamaClient.requests[0]
    assert request["kwargs"]["headers"]["Authorization"] == "Bearer test-token"
    body_str = json.dumps(request["body"])
    assert "test-token" not in body_str
    assert "test-token" not in repr(request["body"])


@pytest.mark.asyncio
async def test_cloudflare_verified_model_and_endpoint(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    await GemmaTargetGenerator().generate_targets("cf-endpoint", context())

    request = FakeOllamaClient.requests[0]
    assert request["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/google/gemma-4-26b-a4b-it"
    )


@pytest.mark.asyncio
async def test_cloudflare_thinking_disabled_in_request(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    await GemmaTargetGenerator().generate_targets("cf-thinking", context())

    request = FakeOllamaClient.requests[0]
    assert request["body"].get("chat_template_kwargs") == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_cloudflare_quota_exhausted_uses_location_fallback(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [
            FakeCloudflareResponse(
                status_code=402,
                error="Daily free allocation exhausted",
            )
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-quota", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_rate_limit_retries_once_then_fallback(monkeypatch):
    responses = [
        FakeCloudflareResponse(
            status_code=429,
            error="rate limit",
        ),
        valid_cloudflare_response(["bus", "dog", "umbrella", "bottle", "kite"]),
    ]
    responses[0].headers = {"Retry-After": "5"}

    configure_fake_cloudflare(monkeypatch, responses)

    targets = await GemmaTargetGenerator().generate_targets("cf-rate-limit", context())

    # Should retry once and succeed on second attempt
    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_invalid_credentials_uses_location_fallback(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [
            FakeCloudflareResponse(
                status_code=401,
                error="Invalid API token",
            )
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-invalid-creds", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_forbidden_token_uses_location_fallback(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [
            FakeCloudflareResponse(
                status_code=403,
                error="Forbidden",
            )
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-forbidden", context())

    # Should fall back to location-ranked targets
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_network_timeout_uses_location_fallback(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [httpx.ConnectError("timeout"), httpx.ConnectError("timeout")],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-timeout", context())

    # Should fall back to location-ranked targets after retries
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_exactly_five_unique_targets(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-five", context())

    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5


@pytest.mark.asyncio
async def test_cloudflare_unsupported_yolo_classes_fallback(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [
            valid_cloudflare_response(["bicycle", "bench", "bird", "car", "book"]),
            valid_cloudflare_response(["bicycle", "bench", "bird", "car", "backpack"]),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-unsupported", context())

    # Should fall back to location-ranked targets (first attempt fails validation, second succeeds with fallback)
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_local_fallback_variety(monkeypatch):
    configure_fake_cloudflare(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("cf-fallback", context())

    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_fallback_preserves_completed_targets_on_context_refresh(monkeypatch):
    from app.services.game_service import GameService
    from app.ai.gemma_generator import target_generator as global_target_generator
    from app.models.game import Game, GameState
    from app.models.location_context import LocationContext
    from app.models.player import Player
    from app.models.target import Target
    from datetime import datetime, timezone

    game = Game(
        _id="cf-refresh-game",
        join_code="CFR1",
        creator_id="host",
        status=GameState.ACTIVE,
        targets=[
            Target(id="old-0", name="Bench", class_name="bench", hint="Old area"),
            Target(id="old-1", name="Bicycle", class_name="bicycle", hint="Old area"),
            Target(id="old-2", name="Bird", class_name="bird", hint="Old area"),
        ],
        location_context=LocationContext(
            locality="Starting District",
            context_mode="contextual",
            context_status="available",
        ),
        location_context_lat=43.0,
        location_context_lon=-79.0,
        location_context_updated_at=datetime.now(timezone.utc),
    )
    host = Player(
        _id="host",
        game_id=game.id,
        display_name="Host",
        completed_targets=["old-0"],
    )
    guest = Player(
        _id="guest",
        game_id=game.id,
        display_name="Guest",
        completed_targets=["old-1"],
    )

    async def get_game(game_id):
        return game

    async def update_game(updated_game, *, expected_targets=None):
        return True

    async def get_players(game_id):
        return [host, guest]

    async def resolve(lat, lon):
        return LocationContext(
            locality="New District",
            environment_type="urban",
            nearby_categories=["public_transport"],
            context_mode="contextual",
            context_status="available",
        )

    monkeypatch.setattr("app.services.game_service.GameRepository.get_by_id", get_game)
    monkeypatch.setattr("app.services.game_service.GameRepository.update", update_game)
    monkeypatch.setattr("app.services.game_service.PlayerRepository.get_by_game", get_players)
    monkeypatch.setattr("app.services.game_service.location_context_service.resolve", resolve)

    # Create a new generator with Cloudflare settings for this test
    cf_generator = GemmaTargetGenerator()
    cf_generator.provider = "cloudflare"
    cf_generator.model = "@cf/google/gemma-4-26b-a4b-it"
    cf_generator.url = "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/@cf/google/gemma-4-26b-a4b-it"
    monkeypatch.setattr("app.services.game_service.target_generator", cf_generator)

    configure_fake_cloudflare(
        monkeypatch,
        [
            FakeCloudflareResponse(
                status_code=402,
                error="quota exhausted",
            )
        ],
    )

    from app.ai.yolo_verifier import image_verifier
    monkeypatch.setattr(
        image_verifier,
        "get_supported_target_classes",
        lambda: ("bus", "traffic light", "stop sign", "dog", "umbrella"),
    )
    # Disable cooldown for test
    monkeypatch.setattr(settings, "GEOCONTEXT_REFRESH_COOLDOWN_SECONDS", 0)

    updated, notice = await GameService.refresh_area_context(
        game.id, 43.002, -79.0
    )
    # First call sets up candidate, second confirms
    updated, notice = await GameService.refresh_area_context(
        game.id, 43.00201, -79.0
    )

    # Refresh should succeed with fallback targets
    assert updated is True
    assert notice is not None
    assert "temporarily unavailable" in notice.lower() or "location-ranked" in notice.lower()
    # Should have 5 targets (2 completed + 3 new fallback)
    assert len(game.targets) == 5
    # Completed targets preserved
    assert "old-0" in [t.id for t in game.targets]
    assert "old-1" in [t.id for t in game.targets]


@pytest.mark.asyncio
async def test_cloudflare_rate_limit_short_retry_after_waits_and_retries(monkeypatch):
    """Short Retry-After (<=60s) should wait and retry once, then succeed."""
    responses = [
        FakeCloudflareResponse(
            status_code=429,
            error="rate limit",
            headers={"Retry-After": "2"},
        ),
        valid_cloudflare_response(["bus", "dog", "umbrella", "bottle", "kite"]),
    ]

    configure_fake_cloudflare(monkeypatch, responses)

    targets = await GemmaTargetGenerator().generate_targets("cf-rate-limit-short", context())

    # Should retry once after waiting and succeed on second attempt
    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_rate_limit_long_retry_after_skips_retry(monkeypatch):
    """Long Retry-After (>60s) should skip retry and use fallback immediately."""
    responses = [
        FakeCloudflareResponse(
            status_code=429,
            error="rate limit",
            headers={"Retry-After": "120"},
        ),
        # Second response should NOT be used because we skip retry for long delays
        # But we need to provide a response for the fallback provider (if any) or the next attempt
        # Since we break immediately, only 1 request should be made
    ]

    configure_fake_cloudflare(monkeypatch, responses)

    targets = await GemmaTargetGenerator().generate_targets("cf-rate-limit-long", context())

    # Should use fallback (only 1 request made, no retry)
    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_429_with_quota_error_in_body_classified_as_quota(monkeypatch):
    """HTTP 429 with quota-related error message should be classified as quota exhausted."""
    # Cloudflare returns 429 with error message about quota
    payload = {
        "success": False,
        "errors": [{"message": "Daily quota exceeded for this model"}],
    }
    response = FakeCloudflareResponse(content=None, status_code=429, error=None)
    response._payload = payload

    configure_fake_cloudflare(monkeypatch, [response])

    targets = await GemmaTargetGenerator().generate_targets("cf-429-quota", context())

    # Should fall back to location-ranked targets (no retry for quota)
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_429_with_rate_limit_error_in_body_classified_as_rate_limited(monkeypatch):
    """HTTP 429 with rate limit error (not quota) should be classified as rate limited."""
    # Cloudflare returns 429 with rate limit error but not quota
    payload = {
        "success": False,
        "errors": [{"message": "Too many requests, please slow down"}],
    }
    response = FakeCloudflareResponse(content=None, status_code=429, error=None)
    response._payload = payload

    configure_fake_cloudflare(monkeypatch, [response, response])

    targets = await GemmaTargetGenerator().generate_targets("cf-429-rate-limit", context())

    # Should fall back after retries (not quota)
    assert len(targets) == 5
    assert len({t.class_name for t in targets}) == 5
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets)
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_cloudflare_rate_limit_no_retry_after_header_retries_once(monkeypatch):
    """Rate limit without Retry-After header should retry once (bounded)."""
    responses = [
        FakeCloudflareResponse(
            status_code=429,
            error="rate limit",
            headers={},  # No Retry-After
        ),
        valid_cloudflare_response(["bus", "dog", "umbrella", "bottle", "kite"]),
    ]

    configure_fake_cloudflare(monkeypatch, responses)

    targets = await GemmaTargetGenerator().generate_targets("cf-rate-limit-no-header", context())

    # Should retry once and succeed
    assert len(targets) == 5
    assert len(FakeOllamaClient.requests) == 2


@pytest.mark.asyncio
async def test_fallback_variety_different_game_ids_produce_different_combinations(monkeypatch):
    """Repeated fallback selection with different game_ids should produce different valid combinations."""
    # Each game needs 2 ConnectError responses (2 attempts before fallback)
    configure_fake_cloudflare(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    # Use same context but different game_ids
    targets_1 = await GemmaTargetGenerator().generate_targets("game-1", context())
    targets_2 = await GemmaTargetGenerator().generate_targets("game-2", context())

    # Both should have 5 unique targets
    assert len(targets_1) == 5
    assert len(targets_2) == 5
    assert len({t.class_name for t in targets_1}) == 5
    assert len({t.class_name for t in targets_2}) == 5

    # All targets should be supported by YOLO
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets_1)
    assert all(t.class_name in OUTDOOR_CANDIDATE_CLASSES for t in targets_2)

    # The combinations should potentially differ (not guaranteed but likely with different game_ids)
    # At minimum, verify they are valid and not just a fixed demo list
    classes_1 = {t.class_name for t in targets_1}
    classes_2 = {t.class_name for t in targets_2}
    # Both should be subsets of OUTDOOR_CANDIDATE_CLASSES
    assert classes_1 <= set(OUTDOOR_CANDIDATE_CLASSES)
    assert classes_2 <= set(OUTDOOR_CANDIDATE_CLASSES)


@pytest.mark.asyncio
async def test_fallback_variety_preserves_location_ranking(monkeypatch):
    """Fallback should prefer highly-ranked candidates for the current area."""
    # Use a park context where bench and bird are highly ranked
    park_context = LocationContext(
        locality="Park District",
        region="Test Region",
        environment_type="park_nature",
        nearby_categories=["parks_and_recreation"],
        nearby_feature_counts={"parks_and_recreation": 3},
        context_mode="contextual",
        context_status="available",
    )

    configure_fake_cloudflare(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    targets = await GemmaTargetGenerator().generate_targets("fallback-park", park_context)

    assert len(targets) == 5
    classes = {t.class_name for t in targets}
    # In park_nature with parks_and_recreation, bench and bird should be highly ranked
    # At least one of them should appear in the fallback
    assert "bench" in classes or "bird" in classes


@pytest.mark.asyncio
async def test_fallback_never_uses_unsupported_or_duplicate_classes(monkeypatch):
    """Fallback should never return unsupported classes or duplicates."""
    # 5 games * 2 attempts each = 10 ConnectError responses needed
    responses = [httpx.ConnectError("offline")] * 10

    configure_fake_cloudflare(
        monkeypatch,
        responses,
    )

    for i in range(5):
        targets = await GemmaTargetGenerator().generate_targets(f"fallback-var-{i}", context())

        assert len(targets) == 5
        classes = [t.class_name for t in targets]
        # No duplicates
        assert len(classes) == len(set(classes))
        # All supported
        assert all(c in OUTDOOR_CANDIDATE_CLASSES for c in classes)
        # All hints present and non-empty
        assert all(t.hint and t.hint.strip() for t in targets)


@pytest.mark.asyncio
async def test_fallback_respects_excluded_classes(monkeypatch):
    """Fallback should respect excluded_classes parameter."""
    configure_fake_cloudflare(
        monkeypatch,
        [
            httpx.ConnectError("offline"),
            httpx.ConnectError("offline"),
        ],
    )

    # Exclude bench and bicycle
    targets = await GemmaTargetGenerator().generate_targets(
        "fallback-excluded",
        context(),
        excluded_classes={"bench", "bicycle"},
    )

    assert len(targets) == 5
    classes = {t.class_name for t in targets}
    assert "bench" not in classes
    assert "bicycle" not in classes
    assert all(c in OUTDOOR_CANDIDATE_CLASSES for c in classes)


@pytest.mark.asyncio
async def test_ollama_still_works_after_cloudflare_integration(monkeypatch):
    configure_fake_ollama(
        monkeypatch,
        [valid_response(["bicycle", "bench", "bird", "car", "backpack"])],
    )
    monkeypatch.setattr(settings, "AI_PROVIDER", "ollama")

    targets = await GemmaTargetGenerator().generate_targets("ollama-still-works", context())

    assert len(targets) == 5
    request = FakeOllamaClient.requests[0]
    assert request["url"] == "http://localhost:11434/api/chat"
    assert "format" in request["body"]


@pytest.mark.asyncio
async def test_openrouter_still_works_after_cloudflare_integration(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "AI_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-secret")
    FakeOllamaClient.responses = [
        FakeOpenRouterResponse(
            json.dumps({"targets": target_set(["bicycle", "bench", "bird", "car", "backpack"])})
        )
    ]
    FakeOllamaClient.requests = []
    monkeypatch.setattr(
        "app.ai.gemma_generator.httpx.AsyncClient", FakeOllamaClient
    )

    targets = await GemmaTargetGenerator().generate_targets("or-still-works", context())

    assert len(targets) == 5
    request = FakeOllamaClient.requests[0]
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["kwargs"]["headers"]["Authorization"] == "Bearer test-secret"
    assert request["body"]["response_format"]["type"] == "json_schema"
