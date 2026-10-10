import json
import hashlib
import logging
import uuid
import asyncio
from typing import Any, Optional
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from app.config import settings
from app.models.location_context import LocationContext, TargetCandidate
from app.models.target import Target, OUTDOOR_CANDIDATE_CLASSES

logger = logging.getLogger(__name__)

TARGET_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _target_count_text(count: int) -> str:
    return TARGET_COUNT_WORDS.get(count, str(count))


DEMO_TARGETS = (
    ("Bicycle", "bicycle", "Look near public roads or bike racks."),
    ("Bench", "bench", "Look in a public park or plaza."),
    ("Bird", "bird", "Look in a public outdoor area."),
    ("Car", "car", "Look along a public street."),
    ("Backpack", "backpack", "Look for someone carrying one in a public place."),
)

CLOUDFLARE_MAX_COMPLETION_TOKENS = 1024
CLOUDFLARE_TIMEOUT = httpx.Timeout(45.0, connect=10.0)
MAX_RETRY_AFTER_SECONDS = 60

CATEGORY_SCORES = {
    "parks_and_recreation": {
        "bench": 10, "bird": 8, "frisbee": 6, "sports ball": 5, "dog": 4,
        "bicycle": 3, "potted plant": 3, "kite": 3, "bottle": 2, "backpack": 2,
    },
    "natural_areas": {
        "bird": 10, "potted plant": 8, "dog": 4, "bench": 3, "kite": 3,
        "bicycle": 2, "bottle": 2, "backpack": 2,
    },
    "residential_areas": {
        "car": 8, "dog": 7, "bicycle": 6, "bird": 5, "backpack": 3,
        "bench": 2, "bottle": 2,
    },
    "commercial_areas": {
        "backpack": 7, "bottle": 6, "bicycle": 6, "car": 5, "bus": 5,
        "umbrella": 3, "traffic light": 3, "stop sign": 2,
    },
    "roads_and_junctions": {
        "car": 9, "bicycle": 7, "traffic light": 8, "stop sign": 7, "bus": 6,
        "truck": 4, "backpack": 3, "umbrella": 3, "bottle": 2,
    },
    "public_transport": {
        "bus": 10, "bicycle": 7, "traffic light": 6, "backpack": 5, "car": 4,
        "umbrella": 3, "bottle": 2, "stop sign": 2,
    },
    "pedestrian_paths": {
        "bicycle": 7, "dog": 6, "bench": 5, "bird": 4, "backpack": 4,
        "frisbee": 4, "bottle": 3, "kite": 2,
    },
    "public_amenities": {
        "bench": 6, "backpack": 4, "bottle": 4, "dog": 3, "bicycle": 3,
        "bird": 3, "umbrella": 2,
    },
}

ENVIRONMENT_SCORES = {
    "urban": {"car": 2, "bus": 2, "bicycle": 2, "backpack": 1, "traffic light": 2},
    "suburban": {"dog": 2, "bicycle": 2, "car": 2, "bird": 1, "bench": 1},
    "rural": {"bird": 2, "dog": 2, "bicycle": 1, "potted plant": 1, "bench": 1},
    "park_nature": {"bird": 3, "bench": 2, "dog": 2, "frisbee": 2, "potted plant": 2},
}


class TargetGenerationError(Exception):
    pass


class GemmaModelUnavailable(TargetGenerationError):
    pass


class GemmaRateLimited(TargetGenerationError):
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class GemmaQuotaExhausted(TargetGenerationError):
    pass


class GemmaCredentialsInvalid(TargetGenerationError):
    pass


class GemmaTransientError(TargetGenerationError):
    pass


class GemmaTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    class_name: str = Field(
        min_length=1,
        json_schema_extra={"enum": list(OUTDOOR_CANDIDATE_CLASSES)},
    )
    hint: str = Field(min_length=1)

    @field_validator("hint")
    @classmethod
    def validate_nonempty_hint(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Target hint must not be empty")
        return value

    @field_validator("class_name", mode="before")
    @classmethod
    def normalize_class_name(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold()
        if normalized not in OUTDOOR_CANDIDATE_CLASSES:
            raise ValueError("Target class is not in the outdoor YOLO vocabulary")
        return normalized


class GemmaTargetList(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    targets: list[GemmaTarget] = Field(min_length=1, max_length=5)

    @field_validator("targets")
    @classmethod
    def validate_unique_classes(cls, targets: list[GemmaTarget]) -> list[GemmaTarget]:
        classes = [target.class_name for target in targets]
        if len(classes) != len(set(classes)):
            raise ValueError("Target classes must be unique")
        return targets


class GemmaTargetGenerator:
    def __init__(self) -> None:
        self.provider = settings.AI_PROVIDER
        self.model, self.url = self._provider_model_and_url(self.provider)

    @staticmethod
    def _provider_model_and_url(provider: str) -> tuple[str, str]:
        if provider == "cloudflare":
            return (
                settings.CLOUDFLARE_AI_MODEL,
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/"
                f"{settings.CLOUDFLARE_AI_MODEL}",
            )
        if provider == "openrouter":
            return (
                settings.OPENROUTER_MODEL,
                f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            )
        return settings.GEMMA_MODEL, f"{settings.OLLAMA_URL.rstrip('/')}/api/chat"

    async def generate_targets(
        self,
        game_id: str,
        context: LocationContext,
        count: int = 5,
        excluded_classes: Optional[set[str]] = None,
        supported_classes: Optional[set[str]] = None,
    ) -> list[Target]:
        if count < 1 or count > 5:
            raise ValueError("Target generation count must be between one and five")

        if settings.DEMO_MODE:
            excluded = {
                class_name.casefold()
                for class_name in (excluded_classes or set())
            }
            supported = {
                class_name.casefold()
                for class_name in (
                    supported_classes or set(OUTDOOR_CANDIDATE_CLASSES)
                )
            }
            demo_targets = [
                target
                for target in DEMO_TARGETS
                if target[1].casefold() in supported
                and target[1].casefold() not in excluded
            ]
            if len(demo_targets) < count:
                raise TargetGenerationError(
                    "Not enough distinct supported demo targets remain."
                )
            return [
                Target(id=str(uuid.uuid4()), name=name, class_name=class_name, hint=hint)
                for name, class_name, hint in demo_targets[:count]
            ]

        excluded = {class_name.casefold() for class_name in (excluded_classes or set())}
        supported = {
            class_name.casefold()
            for class_name in (supported_classes or set(OUTDOOR_CANDIDATE_CLASSES))
        }
        ranked_candidates = [
            candidate
            for candidate in self.rank_candidates(context, game_id, supported)
            if candidate.class_name.casefold() not in excluded
        ]
        context.target_candidates = ranked_candidates
        candidate_pool = [
            candidate
            for candidate in ranked_candidates
            if context.context_mode != "contextual" or candidate.score > 0
        ]
        if len(candidate_pool) < count:
            candidate_pool = ranked_candidates
        if len(candidate_pool) < count:
            raise TargetGenerationError(
                "Not enough supported YOLO classes remain for the requested targets."
            )
        candidates = [candidate.class_name for candidate in candidate_pool]
        logger.info(
            "Ranked target candidates game=%s mode=%s environment=%s categories=%s candidates=%s",
            game_id,
            context.context_mode,
            context.environment_type,
            context.nearby_categories,
            [candidate.model_dump() for candidate in ranked_candidates],
        )
        schema = GemmaTargetList.model_json_schema()
        schema["$defs"]["GemmaTarget"]["properties"]["class_name"]["enum"] = candidates
        schema["properties"]["targets"]["minItems"] = count
        schema["properties"]["targets"]["maxItems"] = count
        providers = [(self.provider, self.model, self.url)]
        fallback_provider = settings.AI_FALLBACK_PROVIDER
        if fallback_provider != "none" and fallback_provider != self.provider:
            fallback_model, fallback_url = self._provider_model_and_url(fallback_provider)
            providers.append((fallback_provider, fallback_model, fallback_url))

        provider_failures: list[tuple[str, TargetGenerationError]] = []
        for provider, model, url in providers:
            for attempt in range(2):
                prompt = self._build_prompt(
                    game_id,
                    context,
                    candidate_pool,
                    count=count,
                    simplified=attempt == 1,
                )
                try:
                    structured_targets = await self._request_targets(
                        prompt,
                        schema,
                        temperature=0.2 if attempt == 0 else 0.0,
                        game_id=game_id,
                        attempt=attempt + 1,
                        allowed_classes=set(candidates),
                        count=count,
                        provider=provider,
                        model=model,
                        url=url,
                    )
                    if len(structured_targets.targets) != count:
                        raise TargetGenerationError(
                            f"Gemma returned an unexpected number of targets (expected {count})."
                        )
                    targets = [
                        Target(
                            id=str(uuid.uuid4()),
                            name=item.class_name.title(),
                            class_name=item.class_name,
                            hint=item.hint,
                        )
                        for item in structured_targets.targets
                    ]
                    logger.info(
                        "Gemma selected target classes game=%s provider=%s targets=%s",
                        game_id,
                        provider,
                        [target.class_name for target in targets],
                    )
                    return targets
                except GemmaQuotaExhausted as exc:
                    provider_failures.append((provider, exc))
                    logger.warning(
                        "Gemma provider quota exhausted game=%s provider=%s",
                        game_id,
                        provider,
                    )
                    break
                except GemmaCredentialsInvalid as exc:
                    provider_failures.append((provider, exc))
                    logger.warning(
                        "Gemma provider credentials rejected game=%s provider=%s",
                        game_id,
                        provider,
                    )
                    break
                except GemmaRateLimited as exc:
                    provider_failures.append((provider, exc))
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after is not None and retry_after <= MAX_RETRY_AFTER_SECONDS:
                        logger.warning(
                            "Gemma provider rate limited; waiting %ss before retry game=%s provider=%s attempt=%s",
                            retry_after,
                            game_id,
                            provider,
                            attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        # Continue to next attempt
                        if attempt == 1:
                            break
                        continue
                    elif retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                        logger.warning(
                            "Gemma provider capacity limited game=%s provider=%s attempt=%s retry_after=%s; skipping retry",
                            game_id,
                            provider,
                            attempt + 1,
                            retry_after,
                        )
                        # For long delays (>60s), skip retry and move to fallback
                        break
                    else:
                        # No Retry-After header: bounded retry (once)
                        logger.warning(
                            "Gemma provider capacity limited game=%s provider=%s attempt=%s; no Retry-After header, bounded retry",
                            game_id,
                            provider,
                            attempt + 1,
                        )
                        if attempt == 1:
                            break
                        continue
                except TargetGenerationError as exc:
                    provider_failures.append((provider, exc))
                    logger.warning(
                        "Structured Gemma attempt failed game=%s provider=%s "
                        "attempt=%s error_type=%s",
                        game_id,
                        provider,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    if attempt == 1:
                        break

        failure_types = sorted({type(error).__name__ for _, error in provider_failures})
        logger.warning(
            "AI target generation unavailable; using supported location-ranked "
            "targets game=%s failed_providers=%s failure_types=%s",
            game_id,
            [provider for provider, _ in provider_failures],
            failure_types,
        )
        fallback_targets = self._build_ranked_fallback_targets(candidate_pool, count, game_id)
        if len(fallback_targets) != count:
            raise TargetGenerationError(
                "AI providers failed and there are not enough distinct supported "
                "YOLO classes for safe fallback targets."
            )
        notice = (
            "AI target generation is temporarily unavailable. Tasks were selected "
            "from location-ranked classes supported by the active image detector."
        )
        context.notice = f"{context.notice} {notice}".strip()
        logger.info(
            "Location-ranked fallback targets game=%s targets=%s",
            game_id,
            [target.class_name for target in fallback_targets],
        )
        return fallback_targets

    @staticmethod
    def _build_ranked_fallback_targets(
        candidates: list[TargetCandidate],
        count: int,
        game_id: str,
    ) -> list[Target]:
        """
        Build fallback targets with meaningful variation.

        Selects from top-ranked candidates using game_id for deterministic but varied
        selection. Prefers highly-ranked candidates for the current area while ensuring
        variety across different game sessions.
        """
        if len(candidates) < count:
            raise TargetGenerationError(
                "Not enough supported YOLO classes for fallback targets."
            )

        # Group candidates by score to find similarly-suitable candidates
        # We'll select from the top tier with variation based on game_id
        scored_candidates = [(c.score, c) for c in candidates]
        scored_candidates.sort(key=lambda x: (-x[0], hashlib.sha256(f"{game_id}:{x[1].class_name}".encode()).hexdigest()))

        # Take top candidates, but if there are ties in score, use game_id to vary selection
        targets: list[Target] = []
        selected_classes: set[str] = set()

        # First pass: pick highest scoring candidates, using game_id for tie-breaking
        for score, candidate in scored_candidates:
            if len(targets) >= count:
                break
            if candidate.class_name in selected_classes:
                continue
            reason = candidate.reasons[0] if candidate.reasons else ""
            hint = (
                f"Look for this object in a public outdoor area suited to {reason}."
                if reason
                else "Look for this common object in a public outdoor area."
            )
            targets.append(
                Target(
                    id=str(uuid.uuid4()),
                    name=candidate.class_name.title(),
                    class_name=candidate.class_name,
                    hint=hint,
                )
            )
            selected_classes.add(candidate.class_name)

        return targets

    @staticmethod
    def rank_candidates(
        game_context: LocationContext,
        game_id: str,
        supported_classes: Optional[set[str]] = None,
    ) -> list[TargetCandidate]:
        eligible_classes = tuple(
            class_name
            for class_name in OUTDOOR_CANDIDATE_CLASSES
            if supported_classes is None or class_name in supported_classes
        )
        scores = {class_name: 0 for class_name in eligible_classes}
        reasons = {class_name: [] for class_name in eligible_classes}

        if game_context.context_mode == "contextual":
            for category, weights in CATEGORY_SCORES.items():
                count = game_context.nearby_feature_counts.get(category, 0)
                if category not in game_context.nearby_categories and not count:
                    continue
                strength = max(1, min(count, 3))
                for class_name, weight in weights.items():
                    if class_name not in scores:
                        continue
                    scores[class_name] += weight * strength
                    reasons[class_name].append(
                        f"{category.replace('_', ' ')} nearby"
                    )
            for class_name, weight in ENVIRONMENT_SCORES.get(
                game_context.environment_type, {}
            ).items():
                if class_name not in scores:
                    continue
                scores[class_name] += weight
                reasons[class_name].append(
                    f"{game_context.environment_type.replace('_', ' ')} environment"
                )
        else:
            for class_name in eligible_classes:
                reasons[class_name].append("general outdoor context; map evidence unavailable")

        def tie_break(class_name: str) -> str:
            return hashlib.sha256(f"{game_id}:{class_name}".encode()).hexdigest()

        return [
            TargetCandidate(
                class_name=class_name,
                score=scores[class_name],
                reasons=reasons[class_name],
            )
            for class_name in sorted(
                eligible_classes,
                key=lambda name: (-scores[name], tie_break(name)),
            )
        ]

    async def _request_targets(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float,
        game_id: str,
        attempt: int,
        allowed_classes: set[str],
        count: int = 5,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
    ) -> GemmaTargetList:
        provider = provider or self.provider
        model = model or self.model
        url = url or self.url
        if provider == "openrouter" and not settings.OPENROUTER_API_KEY:
            raise GemmaModelUnavailable(
                "OpenRouter is selected but OPENROUTER_API_KEY is not configured."
            )
        if provider == "cloudflare" and (
            not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN
        ):
            raise GemmaModelUnavailable(
                "Cloudflare is selected but CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN must be configured."
            )
        headers: dict[str, str] = {}
        request_body: dict[str, Any] = {}
        if provider == "openrouter":
            headers["Authorization"] = f"Bearer {settings.OPENROUTER_API_KEY}"
            request_body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt(count)},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "gemma_target_list",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        elif provider == "cloudflare":
            headers["Authorization"] = f"Bearer {settings.CLOUDFLARE_API_TOKEN}"
            request_body = {
                "messages": [
                    {"role": "system", "content": self._build_system_prompt(count)},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": CLOUDFLARE_MAX_COMPLETION_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": temperature,
                "format": schema,
                "options": {"temperature": temperature},
            }
        try:
            async with httpx.AsyncClient(timeout=CLOUDFLARE_TIMEOUT) as client:
                response = await client.post(
                    url,
                    json=request_body,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "%s chat request failed game=%s attempt=%s error_type=%s",
                provider,
                game_id,
                attempt,
                type(exc).__name__,
            )
            raise GemmaTransientError(
                f"Could not connect to the configured {provider} service."
            ) from exc

        logger.info(
            "%s chat response game=%s attempt=%s status=%s",
            provider,
            game_id,
            attempt,
            response.status_code,
        )

        if provider == "openrouter" and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            delay = None
            if retry_after:
                try:
                    delay = int(retry_after)
                except ValueError:
                    delay = None
            raise GemmaRateLimited(
                "OpenRouter rate limit or provider capacity was reached. Please try again later.",
                retry_after=delay,
            )
        if provider == "cloudflare":
            if response.status_code == 401:
                raise GemmaCredentialsInvalid(
                    "Cloudflare API token is invalid or lacks Workers AI permissions."
                )
            if response.status_code == 403:
                raise GemmaCredentialsInvalid(
                    "Cloudflare API token is forbidden for this account or model."
                )
            if response.status_code == 429:
                # Check response body first for quota indicators
                try:
                    payload = response.json()
                    if not payload.get("success", False):
                        errors = payload.get("errors", [])
                        error_messages = "; ".join(
                            e.get("message", str(e)) for e in errors
                        ) if errors else "Unknown Cloudflare error"
                        if "quota" in error_messages.casefold() or "allocation" in error_messages.casefold():
                            raise GemmaQuotaExhausted(
                                "Cloudflare Workers AI daily free allocation exhausted. "
                                "Check your account quota at https://dash.cloudflare.com."
                            )
                except (ValueError, AttributeError):
                    pass
                # Not a quota error - treat as rate limit
                retry_after = response.headers.get("Retry-After")
                delay = None
                if retry_after:
                    try:
                        delay = int(retry_after)
                    except ValueError:
                        delay = None
                raise GemmaRateLimited(
                    "Cloudflare rate limit or provider capacity was reached. Please try again later.",
                    retry_after=delay,
                )
            if response.status_code == 402:
                raise GemmaQuotaExhausted(
                    "Cloudflare Workers AI daily free allocation exhausted. "
                    "Check your account quota at https://dash.cloudflare.com."
                )
        if provider == "ollama" and response.status_code == 404:
            try:
                error_body = response.json()
                error_text = str(error_body.get("error", ""))
            except (ValueError, AttributeError):
                error_text = ""
            if "model" in error_text.casefold() and (
                "not found" in error_text.casefold() or "pull" in error_text.casefold()
            ):
                raise GemmaModelUnavailable(
                    f"Gemma model '{self.model}' is not installed. Run: ollama pull {self.model}"
                )
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TargetGenerationError(
                f"{provider.title()} returned HTTP {response.status_code} or invalid response JSON."
            ) from exc

        content: Any = None
        try:
            if provider == "openrouter":
                content = payload["choices"][0]["message"]["content"]
                finish_reason = payload["choices"][0].get("finish_reason")
                if finish_reason == "length":
                    raise TargetGenerationError(
                        f"{provider.title()} response was truncated (finish_reason=length)."
                    )
            elif provider == "cloudflare":
                if not payload.get("success", False):
                    errors = payload.get("errors", [])
                    error_messages = "; ".join(
                        e.get("message", str(e)) for e in errors
                    ) if errors else "Unknown Cloudflare error"
                    if "quota" in error_messages.casefold() or "allocation" in error_messages.casefold():
                        raise GemmaQuotaExhausted(
                            "Cloudflare Workers AI daily free allocation exhausted. "
                            "Check your account quota at https://dash.cloudflare.com."
                        )
                    raise TargetGenerationError(
                        f"Cloudflare API error: {error_messages}"
                    )
                result_choices = payload.get("result", {}).get("choices", [])
                if not result_choices:
                    raise TargetGenerationError(
                        "Cloudflare response did not contain result choices."
                    )
                content = result_choices[0]["message"]["content"]
                finish_reason = result_choices[0].get("finish_reason")
                if finish_reason == "length":
                    raise TargetGenerationError(
                        f"Cloudflare response was truncated (finish_reason=length)."
                    )
            else:
                content = payload["message"]["content"]
        except (KeyError, TypeError):
            logger.warning(
                "%s message content parse success=false game=%s attempt=%s",
                provider,
                game_id,
                attempt,
            )
            raise TargetGenerationError(
                f"{provider.title()} response did not contain message content."
            )

        if not isinstance(content, str):
            logger.warning(
                "%s message content is not a string game=%s attempt=%s",
                provider,
                game_id,
                attempt,
            )
            raise TargetGenerationError(
                f"{provider.title()} message content was not text."
            )

        if not content.strip():
            raise TargetGenerationError(
                f"{provider.title()} returned empty message content."
            )

        diagnostic_document: Any = None
        json_valid = False
        try:
            diagnostic_document = json.loads(content)
            json_valid = True
        except json.JSONDecodeError:
            pass
        top_level_keys = (
            sorted(diagnostic_document.keys())
            if isinstance(diagnostic_document, dict)
            else None
        )
        target_count = (
            len(diagnostic_document.get("targets", []))
            if isinstance(diagnostic_document, dict)
            and isinstance(diagnostic_document.get("targets"), list)
            else None
        )
        logger.info(
            "%s structured content game=%s attempt=%s field_parsed=true json_valid=%s top_level_keys=%s target_count=%s",
            provider,
            game_id,
            attempt,
            json_valid,
            top_level_keys,
            target_count,
        )
        try:
            result = GemmaTargetList.model_validate_json(content, strict=True)
        except ValidationError as exc:
            reasons = [
                {
                    "location": error["loc"],
                    "reason": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors(include_url=False)
            ]
            logger.warning(
                "%s structured output validation failed game=%s attempt=%s reasons=%s",
                provider,
                game_id,
                attempt,
                reasons,
            )
            raise TargetGenerationError(
                f"Gemma output did not satisfy the strict {_target_count_text(count)}-target schema."
            ) from exc
        except ValueError as exc:
            logger.warning(
                "%s returned malformed JSON game=%s attempt=%s",
                provider,
                game_id,
                attempt,
            )
            raise TargetGenerationError("Gemma returned malformed JSON.") from exc
        if any(target.class_name not in allowed_classes for target in result.targets):
            raise TargetGenerationError(
                "Gemma selected a class outside the ranked candidate list."
            )
        return result

    @staticmethod
    def _build_system_prompt(count: int) -> str:
        return f"""You are the Game Master for an outdoor scavenger hunt. Your task is to select exactly {_target_count_text(count)} varied, realistically findable targets for a game session.

Rules:
- Return ONLY a JSON object with a "targets" array containing exactly {_target_count_text(count)} objects.
- Each target object must have exactly two fields: "class_name" (string) and "hint" (string).
- The "class_name" must be one of the allowed classes provided in the user prompt.
- The "hint" must be a helpful, location-aware clue for finding the object in a public outdoor area.
- Do not include any additional text, commentary, or formatting outside the JSON object.
- Do not use synonyms or invented class names; the class_name must exactly match an allowed detector class.
- Targets must be distinct (no duplicate class_names).
- Targets must be safe, legal, and findable from public areas without trespassing or approaching traffic.
- Hints should reference the general environment type or nearby features when available, but never claim exact object presence.
"""

    @staticmethod
    def _build_prompt(
        game_id: str,
        context: LocationContext,
        candidates: list[TargetCandidate],
        *,
        count: int = 5,
        simplified: bool = False,
    ) -> str:
        coarse_context = {
            "locality": context.locality,
            "region": context.region,
            "environment_type": context.environment_type,
            "nearby_public_categories": context.nearby_categories,
            "nearby_feature_counts": context.nearby_feature_counts,
            "context_status": context.context_status,
            "context_mode": context.context_mode,
            "context_notice": context.notice,
            "difficulty": "common objects with easy public-area access",
        }
        retry_instruction = (
            f"Return exactly {_target_count_text(count)} unique targets from the provided class vocabulary."
            if simplified
            else f"Select exactly {_target_count_text(count)} unique target classes from the allowed vocabulary."
        )
        retry_instruction += (
            " Every displayed name must identify exactly its detector class; never "
            "use a synonym or a visually similar unsupported object."
        )
        return f"""
You are the Game Master for an outdoor scavenger hunt. Create {_target_count_text(count)} varied,
realistically findable targets for this game session: {game_id}.
Use the coarse, non-coordinate location context below to adapt targets and hints.
If the context is unavailable or partial, say nothing that implies you know exact
nearby places and choose broadly accessible outdoor targets.
Location context: {json.dumps(coarse_context, ensure_ascii=True)}

Choose {_target_count_text(count)} distinct object classes from this location-ranked supported-class list.
Prefer candidates with stronger map-evidence scores, but do not claim an object is
present solely because a nearby feature is mapped. Candidate reasons describe
environmental suitability, not guaranteed object presence:
{json.dumps([candidate.model_dump() for candidate in candidates], ensure_ascii=True)}
The classes must be outdoors-realistic, reasonably common, visible, safe, and
findable from public areas without entering private property or restricted places.
Do not ask players to enter roads, approach traffic, trespass, or photograph people.
Return only a detector class and a hint for each target. The displayed target name is
generated from the detector class by the server, so do not provide a title or synonym.
Avoid dangerous, rare, or inaccessible objects. Do not invent classes outside the
provided vocabulary. {retry_instruction}
Return exactly {_target_count_text(count)} targets. Do not include any additional text.
"""

    @staticmethod
    def _validate_targets(payload: Any, count: int) -> list[Target]:
        if count < 1 or count > 5:
            raise ValueError("Target generation count must be between one and five")
        try:
            structured_targets = GemmaTargetList.model_validate(payload, strict=True)
        except ValidationError as exc:
            raise TargetGenerationError(
                f"Gemma output did not satisfy the strict {_target_count_text(count)}-target schema."
            ) from exc
        if len(structured_targets.targets) != count:
            raise TargetGenerationError(
                f"Gemma output did not satisfy the strict {_target_count_text(count)}-target schema."
            )
        return [
            Target(
                id=str(uuid.uuid4()),
                name=item.class_name.title(),
                class_name=item.class_name,
                hint=item.hint,
            )
            for item in structured_targets.targets
        ]


target_generator = GemmaTargetGenerator()
