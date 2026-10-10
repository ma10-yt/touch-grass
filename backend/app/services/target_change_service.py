import logging
import uuid
from datetime import datetime, timedelta, timezone

from app.ai.gemma_generator import GemmaModelUnavailable, TargetGenerationError, target_generator
from app.ai.yolo_verifier import VerificationUnavailable, image_verifier
from app.config import settings
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.models.game import Game, GameState, TargetChangeVote
from app.models.player import Player

logger = logging.getLogger(__name__)


class TargetChangeError(Exception):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


class TargetChangeService:
    @staticmethod
    def _active_players(players: list[Player], now: datetime) -> list[Player]:
        cutoff = now - timedelta(seconds=settings.TARGET_VOTE_INACTIVE_SECONDS)
        return [
            player
            for player in players
            if player.last_seen_at is not None
            and TargetChangeService._as_utc(player.last_seen_at) >= cutoff
        ]

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    async def request(game_id: str, player_id: str, target_id: str) -> Game:
        if not await PlayerRepository.touch_last_seen(game_id, player_id):
            raise TargetChangeError("Player not found in this game.", 404)
        game = await GameRepository.get_by_id(game_id)
        if not game or game.status != GameState.ACTIVE:
            raise TargetChangeError("Target changes require an active game.", 409)

        target = next((item for item in game.targets if item.id == target_id), None)
        if target is None:
            raise TargetChangeError("That target is no longer active.", 404)

        players = await PlayerRepository.get_by_game(game_id)
        if any(target_id in player.completed_targets for player in players):
            raise TargetChangeError(
                "This target has already been completed by a teammate and cannot be replaced."
            )

        current_vote = game.target_change_vote
        if current_vote and current_vote.status == "pending":
            if current_vote.target_id == target_id:
                return await TargetChangeService.cast_vote(
                    game_id,
                    current_vote.proposal_id,
                    player_id,
                    True,
                )
            raise TargetChangeError(
                "Another target-change vote is already pending for this game."
            )

        now = datetime.now(timezone.utc)
        roster = TargetChangeService._active_players(players, now)
        roster_ids = sorted({player.id for player in roster} | {player_id})
        vote = TargetChangeVote(
            proposal_id=str(uuid.uuid4()),
            target_id=target_id,
            target_name=target.name,
            requested_by=player_id,
            eligible_player_ids=roster_ids,
            approved_player_ids=[player_id],
            status="pending",
        )
        if not await GameRepository.create_target_vote(game_id, vote):
            latest = await GameRepository.get_by_id(game_id)
            if latest and latest.target_change_vote:
                return latest
            raise TargetChangeError("The target vote changed concurrently. Please refresh.")

        game.target_change_vote = vote
        if set(roster_ids).issubset(vote.approved_player_ids):
            await TargetChangeService._generate_and_finalize(game, vote, players)
        return await GameRepository.get_by_id(game_id) or game

    @staticmethod
    async def cast_vote(
        game_id: str,
        proposal_id: str,
        player_id: str,
        approve: bool,
    ) -> Game:
        if not await PlayerRepository.touch_last_seen(game_id, player_id):
            raise TargetChangeError("Player not found in this game.", 404)

        for _ in range(5):
            game = await GameRepository.get_by_id(game_id)
            if not game or game.status != GameState.ACTIVE:
                raise TargetChangeError("Target changes require an active game.", 409)
            vote = game.target_change_vote
            if not vote or vote.proposal_id != proposal_id:
                raise TargetChangeError("This target-change vote is no longer active.")
            if vote.status != "pending":
                return game

            players = await PlayerRepository.get_by_game(game_id)
            roster_ids = sorted({
                player.id
                for player in TargetChangeService._active_players(
                    players, datetime.now(timezone.utc)
                )
            } | {player_id})
            if player_id not in roster_ids:
                raise TargetChangeError("Only an active teammate can vote.", 403)

            previous_revision = vote.revision
            vote.eligible_player_ids = roster_ids
            vote.approved_player_ids = [
                voter_id
                for voter_id in vote.approved_player_ids
                if voter_id in roster_ids
            ]
            if approve:
                if player_id not in vote.approved_player_ids:
                    vote.approved_player_ids.append(player_id)
            else:
                vote.approved_player_ids = [
                    voter_id
                    for voter_id in vote.approved_player_ids
                    if voter_id != player_id
                ]
                vote.status = "rejected"
                vote.message = "A teammate rejected the request. The original target remains."
            vote.revision += 1

            if not await GameRepository.update_pending_target_vote(
                game_id, vote, previous_revision
            ):
                continue
            game.target_change_vote = vote

            if vote.status == "pending" and set(roster_ids).issubset(
                vote.approved_player_ids
            ):
                await TargetChangeService._generate_and_finalize(game, vote, players)
            return await GameRepository.get_by_id(game_id) or game

        raise TargetChangeError("Votes changed concurrently. Refresh and try again.")

    @staticmethod
    async def _generate_and_finalize(
        game: Game,
        vote: TargetChangeVote,
        players: list[Player],
    ) -> None:
        target = next((item for item in game.targets if item.id == vote.target_id), None)
        if target is None:
            return
        if any(vote.target_id in player.completed_targets for player in players):
            vote.status = "unavailable"
            vote.message = (
                "This target was completed while the vote was open; it remains unchanged."
            )
            vote.revision += 1
            await GameRepository.update_pending_target_vote(
                game.id, vote, vote.revision - 1
            )
            return
        try:
            if game.location_context is None:
                raise TargetGenerationError(
                    "Current-area map context is unavailable, so a replacement cannot be selected safely."
                )
            supported = set(image_verifier.get_supported_target_classes())
            if target.class_name not in supported:
                raise VerificationUnavailable(
                    "The current detector does not support this game's target vocabulary."
                )
            completed_classes = {
                item.class_name.casefold()
                for item in game.targets
                if item.id != vote.target_id
            }
            completed_ids = {
                target_id for player in players for target_id in player.completed_targets
            }
            completed_classes.update(
                item.class_name.casefold()
                for item in game.targets
                if item.id in completed_ids
            )
            # Also exclude the class of the target being replaced
            replaced_target = next(
                (item for item in game.targets if item.id == vote.target_id), None
            )
            if replaced_target:
                completed_classes.add(replaced_target.class_name.casefold())
            alternatives = supported - completed_classes
            generated = await target_generator.generate_targets(
                game_id=game.id,
                context=game.location_context,
                count=1,
                excluded_classes=completed_classes,
                supported_classes=alternatives,
            )
            if len(generated) != 1:
                raise TargetGenerationError(
                    "Gemma did not return exactly one supported replacement target."
                )
            replacement = generated[0]
            if (
                replacement.class_name not in supported
                or replacement.class_name.casefold() in completed_classes
                or (
                    not settings.DEMO_MODE
                    and image_verifier.class_id_for_name(replacement.class_name) is None
                )
            ):
                raise TargetGenerationError(
                    "No unique replacement from the active detector's supported classes was available."
                )
            vote.replacement_target = replacement
            vote.status = "approved"
            vote.message = (
                f"Unanimous approval replaced {target.name} with {replacement.name}. "
                "The replacement still requires a verified photo."
            )
            vote.revision += 1
            finalized = await GameRepository.finalize_target_vote(
                game.id, vote, vote.revision - 1
            )
            if finalized:
                logger.info(
                    "Target replacement finalized game=%s target_id=%s replacement_class=%s",
                    game.id,
                    target.id,
                    replacement.class_name,
                )
        except (GemmaModelUnavailable, TargetGenerationError, VerificationUnavailable) as exc:
            vote.status = "unavailable"
            vote.message = (
                f"Unanimous approval was recorded, but no valid replacement could be "
                f"generated: {exc} The original target remains."
            )
            vote.replacement_target = None
            vote.revision += 1
            await GameRepository.update_pending_target_vote(
                game.id, vote, vote.revision - 1
            )
            logger.warning(
                "Target replacement unavailable game=%s target_id=%s reason=%s",
                game.id,
                vote.target_id,
                exc,
            )
