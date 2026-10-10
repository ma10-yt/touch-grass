import json
import hashlib
import logging
import uuid
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
        self.model = settings.GEMMA_MODEL
        self.base_url = settings.OLLAMA_URL.rstrip("/")
        self.url = f"{self.base_url}/api/chat"

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
                    "Gemma selected target classes game=%s targets=%s",
                    game_id,
                    [target.class_name for target in targets],
                )
                return targets
            except GemmaModelUnavailable:
                raise
            except TargetGenerationError as exc:
                logger.warning(
                    "Structured Gemma attempt %s failed for game %s: %s",
                    attempt + 1,
                    game_id,
                    exc,
                )
                if attempt == 1:
                    raise TargetGenerationError(
                        f"Local Gemma target generation is unavailable or returned invalid output after two attempts: {exc}"
                    ) from exc
        raise TargetGenerationError("Local Gemma target generation failed.")

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
    ) -> GemmaTargetList:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                response = await client.post(
                    self.url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": schema,
                        "options": {"temperature": temperature},
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama chat request failed game=%s attempt=%s error_type=%s",
                game_id,
                attempt,
                type(exc).__name__,
            )
            raise TargetGenerationError("Could not connect to the local Ollama service.") from exc

        logger.info(
            "Ollama chat response game=%s attempt=%s status=%s",
            game_id,
            attempt,
            response.status_code,
        )
        if response.status_code == 404:
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
                f"Ollama returned HTTP {response.status_code} or invalid response JSON."
            ) from exc

        content: Any = None
        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError):
            logger.warning(
                "Ollama message content parse success=false game=%s attempt=%s",
                game_id,
                attempt,
            )
            raise TargetGenerationError("Ollama response did not contain message.content.")

        if not isinstance(content, str):
            logger.warning(
                "Ollama message content is not a string game=%s attempt=%s",
                game_id,
                attempt,
            )
            raise TargetGenerationError("Ollama message.content was not text.")

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
            "Ollama structured content game=%s attempt=%s field_parsed=true json_valid=%s top_level_keys=%s target_count=%s",
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
                "Ollama structured output validation failed game=%s attempt=%s reasons=%s",
                game_id,
                attempt,
                reasons,
            )
            raise TargetGenerationError(
                f"Gemma output did not satisfy the strict {_target_count_text(count)}-target schema."
            ) from exc
        except ValueError as exc:
            logger.warning(
                "Ollama returned malformed JSON game=%s attempt=%s",
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
