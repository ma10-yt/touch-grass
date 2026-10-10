import logging
import uuid
from datetime import datetime, timezone
from app.ai.yolo_verifier import image_verifier
from app.config import settings
from app.db.repositories.repositories import (
    GameRepository,
    PlayerRepository,
    SubmissionRepository,
)
from app.models.game import Game, GameState
from app.models.player import Player
from app.models.submission import Submission, SubmissionResult
from app.services.location_service import LocationService
from app.utils.exif import extract_exif_data

logger = logging.getLogger(__name__)


class VerificationService:
    @staticmethod
    def _result(
        *,
        success: bool,
        target: str,
        verified: bool,
        confidence: float | None,
        message: str,
        game: Game | None = None,
        player: Player | None = None,
    ) -> SubmissionResult:
        target_ids = {item.id for item in game.targets} if game else set()
        completed_count = (
            len(set(player.completed_targets) & target_ids) if player else 0
        )
        total_count = len(target_ids)
        progress = round(completed_count / total_count * 100) if total_count else 0
        if settings.DEMO_MODE and verified:
            message = f"DEMO MODE: {message}"
        return SubmissionResult(
            success=success,
            target=target,
            verified=verified,
            confidence=confidence,
            message=message,
            completed_targets=completed_count,
            total_targets=total_count,
            progress_percent=progress,
            demo_mode=settings.DEMO_MODE,
        )

    @staticmethod
    async def process_submission(
        game_id: str,
        player_id: str,
        target_id: str,
        file_path: str,
    ) -> SubmissionResult:
        game = await GameRepository.get_by_id(game_id)
        player = await PlayerRepository.get_by_id(player_id)

        if not game or game.status != GameState.ACTIVE:
            return VerificationService._result(
                success=False, target="", verified=False, confidence=None,
                message="Game not active.", game=game, player=player,
            )
        if not player or player.game_id != game_id:
            return VerificationService._result(
                success=False, target="", verified=False, confidence=None,
                message="Player not found in game.", game=game,
            )

        target = next((item for item in game.targets if item.id == target_id), None)
        if not target:
            return VerificationService._result(
                success=False, target="", verified=False, confidence=None,
                message="Target not found in game.", game=game, player=player,
            )
        logger.info(
            "Image verification started game=%s target_id=%s target_title=%s "
            "required_class=%s required_class_id=%s confidence_threshold=%.2f",
            game_id,
            target.id,
            target.name,
            target.class_name,
            image_verifier.class_id_for_name(target.class_name),
            settings.YOLO_CONFIDENCE_THRESHOLD,
        )
        if target_id in player.completed_targets:
            return VerificationService._result(
                success=False, target=target.class_name, verified=False, confidence=None,
                message="Target already completed.", game=game, player=player,
            )

        loc_valid, loc_msg, _ = await LocationService.validate_player_location(player)
        if not loc_valid:
            return VerificationService._result(
                success=False, target=target.class_name, verified=False, confidence=None,
                message=loc_msg, game=game, player=player,
            )

        extract_exif_data(file_path)
        detection = await image_verifier.detect_image(file_path, target.class_name)
        current_game = await GameRepository.get_by_id(game_id)
        current_target = (
            next(
                (item for item in current_game.targets if item.id == target_id),
                None,
            )
            if current_game
            else None
        )
        if current_target is None or current_target.class_name != target.class_name:
            return VerificationService._result(
                success=False,
                target=target.class_name,
                verified=False,
                confidence=detection.confidence,
                message=(
                    "This target changed while the photo was being verified. "
                    "Refresh the game and submit its current target."
                ),
                game=current_game,
                player=player,
            )
        class_matches = (
            detection.class_name is not None
            and detection.class_name.casefold() == target.class_name.casefold()
        )
        verified = class_matches and detection.confidence >= settings.YOLO_CONFIDENCE_THRESHOLD

        if verified:
            message = f"{target.name} verified!"
        elif class_matches:
            message = (
                f"YOLO detected {target.name} at {detection.confidence:.0%} confidence, "
                f"below the {settings.YOLO_CONFIDENCE_THRESHOLD:.0%} verification threshold."
            )
        elif detection.class_name is None:
            message = "No supported objects were detected. Try a clearer, closer photo."
        else:
            message = (
                f"YOLO detected {detection.class_name}, not {target.name}. "
                "Center the target and try again."
            )

        logger.info(
            "Image verification decision target_class=%s detected_class=%s "
            "detected_class_id=%s confidence=%.4f threshold=%.2f "
            "class_matches=%s verified=%s reason=%s",
            target.class_name,
            detection.class_name,
            detection.class_id,
            detection.confidence,
            settings.YOLO_CONFIDENCE_THRESHOLD,
            class_matches,
            verified,
            (
                "verified"
                if verified
                else "below_threshold"
                if class_matches
                else "no_detections"
                if detection.class_name is None
                else "different_class"
            ),
        )

        await SubmissionRepository.create(
            Submission(
                _id=str(uuid.uuid4()),
                game_id=game_id,
                player_id=player_id,
                target_id=target_id,
                verified=verified,
                confidence=detection.confidence,
                message=message,
                distance_meters=None,
            )
        )

        if not verified:
            return VerificationService._result(
                success=False, target=target.class_name, verified=False,
                confidence=detection.confidence, message=message, game=game, player=player,
            )

        updated_player = await PlayerRepository.complete_target_once(player_id, target_id)
        if not updated_player:
            current_player = await PlayerRepository.get_by_id(player_id) or player
            return VerificationService._result(
                success=False, target=target.class_name, verified=False,
                confidence=detection.confidence, message="Target already completed.",
                game=game, player=current_player,
            )

        player = updated_player
        completed_all = (
            len(set(player.completed_targets) & {item.id for item in game.targets})
            == len(game.targets)
        )
        if completed_all:
            game.status = GameState.FINISHED
            game.end_time = datetime.now(timezone.utc)
            await GameRepository.update(game)
            message = f"{target.name} verified! You found all targets and won the game!"

        return VerificationService._result(
            success=True, target=target.class_name, verified=True,
            confidence=detection.confidence, message=message, game=game, player=player,
        )
