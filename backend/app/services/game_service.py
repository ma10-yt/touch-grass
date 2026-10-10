import asyncio
import uuid
import logging
from collections import Counter
from typing import Optional, List
from datetime import datetime, timezone
from app.ai.gemma_generator import target_generator
from app.models.location_context import LocationContext, TargetCandidate
from app.models.game import Game, GameState
from app.models.player import Player
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.config import settings
from app.services.location_context_service import location_context_service
from app.services.location_service import LocationService
from app.utils.security import generate_join_code
from app.utils.geo import calculate_distance
from app.ai.gemma_generator import GemmaModelUnavailable, TargetGenerationError
from app.ai.yolo_verifier import VerificationUnavailable, image_verifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TARGET_HISTORY_SIZE = 3
REPEATED_TARGET_HISTORY_THRESHOLD = 2


def _repeated_context_classes_to_avoid(
    recent_target_sets: list[list[str]],
    ranked_candidates: list[TargetCandidate],
    *,
    target_count: int,
) -> set[str]:
    usage_count: Counter[str] = Counter()
    for target_set in recent_target_sets:
        usage_count.update({class_name.casefold() for class_name in target_set})

    eligible_candidates = [
        candidate for candidate in ranked_candidates if candidate.score > 0
    ]
    if len(eligible_candidates) < target_count:
        return set()

    candidate_by_name = {
        candidate.class_name.casefold(): candidate
        for candidate in eligible_candidates
    }
    repeated = {
        class_name
        for class_name, count in usage_count.items()
        if count >= REPEATED_TARGET_HISTORY_THRESHOLD
        and class_name in candidate_by_name
    }
    if len(candidate_by_name) - len(repeated) < target_count:
        restore_order = sorted(
            repeated,
            key=lambda class_name: (
                usage_count[class_name],
                -candidate_by_name[class_name].score,
                class_name,
            ),
        )
        for class_name in restore_order:
            repeated.remove(class_name)
            if len(candidate_by_name) - len(repeated) >= target_count:
                break
    return repeated


class GameStartError(Exception):
    pass

class GameService:
    _context_refresh_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    async def create_game(creator_name: str, lat: Optional[float] = None, lon: Optional[float] = None) -> Game:
        game_id = str(uuid.uuid4())
        creator_id = str(uuid.uuid4())
        join_code = generate_join_code()

        game = Game(
            _id=game_id,
            join_code=join_code,
            creator_id=creator_id,
            status=GameState.WAITING,
            center_lat=lat,
            center_lon=lon
        )
        await GameRepository.create(game)

        creator = Player(
            _id=creator_id,
            game_id=game_id,
            display_name=creator_name,
            latest_lat=lat,
            latest_lon=lon,
            last_seen_at=datetime.now(timezone.utc),
        )
        await PlayerRepository.create(creator)
        logger.info(
            "Created new waiting game game=%s target_count=0 demo_mode=%s",
            game_id,
            settings.DEMO_MODE,
        )

        return game

    @staticmethod
    async def join_game(join_code: str, display_name: str) -> Optional[Player]:
        game = await GameRepository.get_by_join_code(join_code)
        if not game or game.status != GameState.WAITING:
            return None

        player_id = str(uuid.uuid4())
        player = Player(
            _id=player_id,
            game_id=game.id,
            display_name=display_name,
            last_seen_at=datetime.now(timezone.utc),
        )
        await PlayerRepository.create(player)
        return player

    @staticmethod
    async def start_game(game_id: str, creator_id: str) -> LocationContext:
        game = await GameRepository.get_by_id(game_id)
        if not game:
            raise GameStartError("Game not found.")
        if game.creator_id != creator_id:
            raise GameStartError("Only the game host can start the game.")
        if game.status != GameState.WAITING:
            raise GameStartError("Game must be waiting to start.")

        players = await PlayerRepository.get_by_game(game_id)

        host = next((player for player in players if player.id == game.creator_id), None)
        if game.location_context is None:
            context_lat = context_lon = None
            if host:
                location_valid, _, _ = await LocationService.validate_player_location(host)
                if location_valid:
                    context_lat = host.latest_lat
                    context_lon = host.latest_lon
            game.location_context = await location_context_service.resolve(
                context_lat, context_lon
            )
        if (
            host
            and host.latest_lat is not None
            and host.latest_lon is not None
            and host.last_location_update is not None
            and host.location_accuracy_m is not None
            and host.location_accuracy_m <= settings.GPS_MAX_ACCURACY_METERS
        ):
            game.location_context_lat = host.latest_lat
            game.location_context_lon = host.latest_lon
        else:
            game.location_context_lat = None
            game.location_context_lon = None
        game.location_context_updated_at = datetime.now(timezone.utc)

        context = game.location_context
        logger.info(
            "Game %s location context locality=%s environment=%s nearby_categories=%s status=%s",
            game.id,
            context.locality or "unknown",
            context.environment_type,
            context.nearby_categories,
            context.context_status,
        )
        source = "demo targets" if settings.DEMO_MODE else "Gemma"
        model_name = None if settings.DEMO_MODE else settings.GEMMA_MODEL
        logger.info(
            "Starting fresh target generation game=%s existing_target_count=%s "
            "source=%s model=%s demo_mode=%s yolo_model=%s context_mode=%s "
            "context_status=%s context_provider=%s",
            game.id,
            len(game.targets),
            source,
            model_name,
            settings.DEMO_MODE,
            settings.YOLO_MODEL,
            context.context_mode,
            context.context_status,
            context.context_provider,
        )
        try:
            supported_classes = set(image_verifier.get_supported_target_classes())
            repeated_classes: set[str] = set()
            if not settings.DEMO_MODE and context.context_mode == "contextual":
                recent_target_sets = await GameRepository.get_recent_target_sets_for_context(
                    game.id,
                    context,
                    limit=TARGET_HISTORY_SIZE,
                )
                ranked_candidates = target_generator.rank_candidates(
                    context,
                    game.id,
                    supported_classes,
                )
                repeated_classes = _repeated_context_classes_to_avoid(
                    recent_target_sets,
                    ranked_candidates,
                    target_count=5,
                )
                logger.info(
                    "Context target history game=%s matching_games=%s "
                    "repeated_classes_avoided=%s",
                    game.id,
                    len(recent_target_sets),
                    sorted(repeated_classes),
                )
            targets = await target_generator.generate_targets(
                game_id=game.id,
                context=context,
                count=5,
                excluded_classes=repeated_classes,
                supported_classes=supported_classes,
            )
        except (GemmaModelUnavailable, TargetGenerationError, VerificationUnavailable):
            logger.exception(
                "Fresh target generation failed game=%s source=%s model=%s "
                "demo_mode=%s context_mode=%s context_status=%s",
                game.id,
                source,
                model_name,
                settings.DEMO_MODE,
                context.context_mode,
                context.context_status,
            )
            raise
        if (
            len(targets) != 5
            or len({target.class_name for target in targets}) != 5
            or any(target.class_name not in supported_classes for target in targets)
        ):
            raise TargetGenerationError(
                "The target generator did not return five unique supported targets."
            )
        previous_targets = list(game.targets)
        game.targets = targets
        logger.info(
            "Generated targets game=%s mode=%s environment=%s categories=%s "
            "candidate_classes=%s candidate_ranking=%s selected_targets=%s",
            game.id,
            context.context_mode,
            context.environment_type,
            context.nearby_categories,
            [candidate.class_name for candidate in context.target_candidates],
            [candidate.model_dump() for candidate in context.target_candidates],
            [
                (
                    target.name,
                    target.class_name,
                    image_verifier.class_id_for_name(target.class_name),
                )
                for target in targets
            ],
        )
        game.status = GameState.ACTIVE
        game.start_time = datetime.now(timezone.utc)
        if not await GameRepository.activate_with_targets(game, previous_targets):
            raise GameStartError(
                "The game changed while targets were being prepared. Refresh and try again."
            )

        for player in players:
            player.movement_meters = 0.0
            if (
                player.latest_lat is not None
                and player.latest_lon is not None
                and player.last_location_update is not None
                and player.location_accuracy_m is not None
                and player.location_accuracy_m <= settings.GPS_MOVEMENT_MAX_ACCURACY_METERS
            ):
                player.movement_last_lat = player.latest_lat
                player.movement_last_lon = player.latest_lon
                player.movement_last_sample_at = player.last_location_update
                player.movement_last_accuracy_m = player.location_accuracy_m
            else:
                player.movement_last_lat = None
                player.movement_last_lon = None
                player.movement_last_sample_at = None
                player.movement_last_accuracy_m = None
            player.movement_candidate_lat = None
            player.movement_candidate_lon = None
            player.movement_candidate_count = 0
            await PlayerRepository.update(player)
        return context

    @staticmethod
    async def refresh_area_context(
        game_id: str,
        lat: float,
        lon: float,
    ) -> tuple[bool, str | None]:
        lock = GameService._context_refresh_locks.setdefault(game_id, asyncio.Lock())
        async with lock:
            game = await GameRepository.get_by_id(game_id)
            if not game or game.status != GameState.ACTIVE:
                return False, None

            original_targets = list(game.targets)
            now = datetime.now(timezone.utc)
            candidate_is_stable = (
                game.context_candidate_lat is not None
                and game.context_candidate_lon is not None
                and calculate_distance(
                    game.context_candidate_lat,
                    game.context_candidate_lon,
                    lat,
                    lon,
                ) <= settings.GEOCONTEXT_REFRESH_STABILITY_METERS
            )
            if candidate_is_stable:
                candidate_count = game.context_candidate_count + 1
            else:
                game.context_candidate_lat = lat
                game.context_candidate_lon = lon
                candidate_count = 1
            game.context_candidate_count = candidate_count

            if candidate_count < settings.GEOCONTEXT_REFRESH_CONFIRMATIONS:
                await GameRepository.update(game)
                return False, None

            distance_from_context = (
                calculate_distance(game.location_context_lat, game.location_context_lon, lat, lon)
                if game.location_context_lat is not None
                and game.location_context_lon is not None
                else settings.GEOCONTEXT_REFRESH_DISTANCE_METERS
            )
            elapsed_since_refresh = (
                (now - LocationService._as_utc(game.location_context_updated_at)).total_seconds()
                if game.location_context_updated_at is not None
                else float("inf")
            )
            if distance_from_context < settings.GEOCONTEXT_REFRESH_DISTANCE_METERS:
                await GameRepository.update(game)
                return False, None
            if elapsed_since_refresh < settings.GEOCONTEXT_REFRESH_COOLDOWN_SECONDS:
                await GameRepository.update(game)
                return False, None

            context = await location_context_service.resolve(lat, lon)
            if not context.locality and context.context_status != "unavailable":
                context.notice = (
                    (context.notice + " ") if context.notice else ""
                ) + "Place identification is approximate; tasks use available nearby map evidence."

            players = await PlayerRepository.get_by_game(game_id)
            completed_ids = {
                target_id
                for player in players
                for target_id in player.completed_targets
            }
            completed_targets = [
                target for target in game.targets if target.id in completed_ids
            ]
            required_new_targets = max(0, 5 - len(completed_targets))
            if required_new_targets:
                excluded_classes = {
                    target.class_name.casefold() for target in completed_targets
                }
                try:
                    supported_classes = set(
                        image_verifier.get_supported_target_classes()
                    )
                    generated = await target_generator.generate_targets(
                        game_id=game.id,
                        context=context,
                        count=5,
                        excluded_classes=excluded_classes,
                        supported_classes=supported_classes,
                    )
                except (
                    GemmaModelUnavailable,
                    TargetGenerationError,
                    VerificationUnavailable,
                ) as exc:
                    logger.warning(
                        "Area task refresh failed game=%s error_type=%s",
                        game.id,
                        type(exc).__name__,
                    )
                    return False, (
                        "New area was reached, but the available AI models could not "
                        "refresh tasks. "
                        "The current shared tasks remain active."
                    )
                new_targets = [
                    target
                    for target in generated
                    if target.class_name.casefold() not in excluded_classes
                ][:required_new_targets]
                if len(new_targets) < required_new_targets:
                    return False, (
                        "Nearby context was found, but Gemma could not provide enough "
                        "new supported targets. Existing tasks remain active."
                    )
                game.targets = completed_targets + new_targets
            else:
                game.targets = completed_targets

            game.location_context = context
            game.location_context_lat = lat
            game.location_context_lon = lon
            game.location_context_updated_at = now
            game.context_candidate_lat = None
            game.context_candidate_lon = None
            game.context_candidate_count = 0
            await GameRepository.update(
                game,
                expected_targets=original_targets,
            )
            logger.info(
                "Refreshed shared game area context game=%s locality=%s "
                "target_count=%s completed_preserved=%s",
                game.id,
                context.locality or "approximate area",
                len(game.targets),
                len(completed_targets),
            )
            return True, context.notice

    @staticmethod
    async def get_leaderboard(game_id: str) -> List[Player]:
        return await PlayerRepository.get_by_game(game_id)
