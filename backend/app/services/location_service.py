from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional, Tuple

from app.config import settings
from app.db.repositories.repositories import PlayerRepository
from app.models.player import Player
from app.utils.geo import calculate_distance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationUpdateResult:
    accepted: bool
    message: str
    movement_meters: float
    movement_added_meters: float = 0.0
    game_finished: bool = False
    movement_accepted: bool = False


class LocationService:
    @staticmethod
    def _as_utc(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    async def update_location(
        game_id: str,
        player_id: str,
        lat: float,
        lon: float,
        accuracy_m: float,
        timestamp: datetime,
    ) -> LocationUpdateResult:
        player = await PlayerRepository.get_by_id(player_id)
        if not player or player.game_id != game_id:
            logger.warning(
                "Location fix rejected: player not found in game game_id=%s player_id=%s",
                game_id,
                player_id,
            )
            return LocationUpdateResult(False, "Player not found in this game.", 0.0)

        now = datetime.now(timezone.utc)
        sample_time = LocationService._as_utc(timestamp)
        age_seconds = (now - sample_time).total_seconds()
        if age_seconds < -5:
            logger.warning(
                "Location fix rejected reason=future_timestamp game_id=%s player_id=%s",
                game_id,
                player_id,
            )
            return LocationUpdateResult(
                False, "Location timestamp is in the future.", player.movement_meters
            )
        if age_seconds > settings.LOCATION_MAX_AGE_SECONDS:
            logger.warning(
                "Location fix rejected reason=stale_timestamp age_seconds=%.1f "
                "game_id=%s player_id=%s",
                age_seconds,
                game_id,
                player_id,
            )
            return LocationUpdateResult(
                False, "Location sample is stale; request a fresh GPS fix.",
                player.movement_meters,
            )
        if (
            player.last_location_update is not None
            and sample_time <= LocationService._as_utc(player.last_location_update)
        ):
            logger.warning(
                "Location fix rejected reason=out_of_order_timestamp game_id=%s player_id=%s",
                game_id,
                player_id,
            )
            return LocationUpdateResult(
                False, "Location sample is older than the last update.",
                player.movement_meters,
            )

        if accuracy_m > settings.GPS_MAX_ACCURACY_METERS:
            logger.warning(
                "Location fix rejected reason=poor_accuracy accuracy_m=%.1f "
                "maximum_m=%.1f game_id=%s player_id=%s",
                accuracy_m,
                settings.GPS_MAX_ACCURACY_METERS,
                game_id,
                player_id,
            )
            return LocationUpdateResult(
                False,
                f"GPS accuracy is {accuracy_m:.0f} m; wait for a fix within "
                f"{settings.GPS_MAX_ACCURACY_METERS:.0f} m.",
                player.movement_meters,
            )

        previous_location = (
            player.latest_lat,
            player.latest_lon,
            player.last_location_update,
            player.location_accuracy_m,
            player.movement_meters,
            player.movement_last_lat,
            player.movement_last_lon,
            player.movement_last_sample_at,
            player.movement_last_accuracy_m,
            player.movement_candidate_lat,
            player.movement_candidate_lon,
            player.movement_candidate_count,
        )

        movement_added = 0.0
        movement_accepted = accuracy_m <= settings.GPS_MOVEMENT_MAX_ACCURACY_METERS
        movement_message = "Approximate location accepted; movement awaits stable readings."
        if movement_accepted:
            if (
                player.movement_last_lat is None
                or player.movement_last_lon is None
                or player.movement_last_sample_at is None
            ):
                player.movement_last_lat = lat
                player.movement_last_lon = lon
                player.movement_last_sample_at = sample_time
                player.movement_last_accuracy_m = accuracy_m
                player.movement_candidate_lat = None
                player.movement_candidate_lon = None
                player.movement_candidate_count = 0
            else:
                elapsed = (
                    sample_time - LocationService._as_utc(player.movement_last_sample_at)
                ).total_seconds()
                segment_distance = calculate_distance(
                    player.movement_last_lat,
                    player.movement_last_lon,
                    lat,
                    lon,
                )
                speed = segment_distance / elapsed if elapsed > 0 else float("inf")
                if elapsed <= 0 or speed > settings.GPS_MAX_SPEED_METERS_PER_SECOND:
                    logger.warning(
                        "Movement segment ignored reason=implausible_speed speed_mps=%.2f "
                        "maximum_mps=%.2f game_id=%s player_id=%s",
                        speed,
                        settings.GPS_MAX_SPEED_METERS_PER_SECOND,
                        game_id,
                        player_id,
                    )
                    player.movement_candidate_lat = None
                    player.movement_candidate_lon = None
                    player.movement_candidate_count = 0
                    movement_accepted = False
                    movement_message = (
                        "Approximate location accepted for area context; "
                        "movement ignored because the jump was implausible."
                    )
                elif segment_distance < (
                    settings.GPS_MIN_SEGMENT_METERS
                    + settings.GPS_MOVEMENT_NOISE_FLOOR_METERS
                ):
                    player.movement_candidate_lat = None
                    player.movement_candidate_lon = None
                    player.movement_candidate_count = 0
                    movement_message = (
                        "Approximate location accepted; no meaningful movement detected."
                    )
                else:
                    if (
                        player.movement_candidate_lat is not None
                        and player.movement_candidate_lon is not None
                        and calculate_distance(
                            player.movement_candidate_lat,
                            player.movement_candidate_lon,
                            lat,
                            lon,
                        ) <= settings.GPS_MOVEMENT_STABILITY_METERS
                    ):
                        player.movement_candidate_count += 1
                    else:
                        player.movement_candidate_lat = lat
                        player.movement_candidate_lon = lon
                        player.movement_candidate_count = 1

                    if player.movement_candidate_count >= settings.GPS_MOVEMENT_CONFIRMATIONS:
                        movement_added = max(
                            0.0,
                            segment_distance - settings.GPS_MOVEMENT_NOISE_FLOOR_METERS,
                        )
                        player.movement_meters += movement_added
                        player.movement_last_lat = lat
                        player.movement_last_lon = lon
                        player.movement_last_sample_at = sample_time
                        player.movement_last_accuracy_m = accuracy_m
                        player.movement_candidate_lat = None
                        player.movement_candidate_lon = None
                        player.movement_candidate_count = 0
                        movement_message = (
                            "Approximate location accepted; stable movement recorded."
                        )
                    else:
                        movement_message = (
                            "Approximate location accepted; confirming movement with another fix."
                        )
        else:
            movement_message = (
                "Approximate location accepted for area context; accuracy is too low "
                "for movement tracking."
            )

        player.latest_lat = lat
        player.latest_lon = lon
        player.last_location_update = sample_time
        player.location_accuracy_m = accuracy_m

        try:
            saved = await PlayerRepository.update(player)
        except Exception:
            (
                player.latest_lat,
                player.latest_lon,
                player.last_location_update,
                player.location_accuracy_m,
                player.movement_meters,
                player.movement_last_lat,
                player.movement_last_lon,
                player.movement_last_sample_at,
                player.movement_last_accuracy_m,
                player.movement_candidate_lat,
                player.movement_candidate_lon,
                player.movement_candidate_count,
            ) = previous_location
            logger.exception(
                "Failed to save location fix game_id=%s player_id=%s",
                game_id,
                player_id,
            )
            raise
        if not saved:
            (
                player.latest_lat,
                player.latest_lon,
                player.last_location_update,
                player.location_accuracy_m,
                player.movement_meters,
                player.movement_last_lat,
                player.movement_last_lon,
                player.movement_last_sample_at,
                player.movement_last_accuracy_m,
                player.movement_candidate_lat,
                player.movement_candidate_lon,
                player.movement_candidate_count,
            ) = previous_location
            logger.error(
                "Location fix not saved: player document was not matched "
                "game_id=%s player_id=%s",
                game_id,
                player_id,
            )
            return LocationUpdateResult(
                False,
                "Location could not be saved. Retry when connected.",
                player.movement_meters,
            )

        logger.info(
            "Location estimate accepted game_id=%s player_id=%s accuracy_m=%.1f "
            "quality=%s movement_added_m=%.1f movement_total_m=%.1f",
            game_id,
            player_id,
            accuracy_m,
            "higher-confidence" if accuracy_m <= settings.GPS_HIGH_CONFIDENCE_METERS else "approximate",
            movement_added,
            player.movement_meters,
        )
        return LocationUpdateResult(
            True,
            movement_message,
            player.movement_meters,
            movement_added,
            False,
            movement_accepted,
        )

    @staticmethod
    async def validate_player_location(
        player: Player
    ) -> Tuple[bool, str, Optional[float]]:
        if (
            player.latest_lat is None
            or player.latest_lon is None
            or player.last_location_update is None
        ):
            logger.warning(
                "Photo verification location unavailable: no accepted fix is stored "
                "player_id=%s game_id=%s",
                player.id,
                player.game_id,
            )
            return False, "Location not available", None

        now = datetime.now(timezone.utc)
        last_update = LocationService._as_utc(player.last_location_update)
        age_seconds = (now - last_update).total_seconds()
        if age_seconds > settings.LOCATION_MAX_AGE_SECONDS or age_seconds < -5:
            logger.warning(
                "Photo verification location rejected reason=stale age_seconds=%.1f "
                "player_id=%s game_id=%s",
                age_seconds,
                player.id,
                player.game_id,
            )
            return False, "Location is stale; request a fresh GPS fix.", None
        if (
            player.location_accuracy_m is None
            or player.location_accuracy_m > settings.GPS_MAX_ACCURACY_METERS
        ):
            logger.warning(
                "Photo verification location rejected reason=poor_accuracy "
                "accuracy_m=%s maximum_m=%.1f player_id=%s game_id=%s",
                player.location_accuracy_m,
                settings.GPS_MAX_ACCURACY_METERS,
                player.id,
                player.game_id,
            )
            return (
                False,
                "GPS accuracy is insufficient. Keep walking until a reliable fix is available.",
                None,
            )
        return True, "Location valid", None
