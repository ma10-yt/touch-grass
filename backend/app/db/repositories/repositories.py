from datetime import datetime, timezone
from typing import Optional, List
import logging
from app.db.client import get_db
from app.models.game import Game
from app.models.game import TargetChangeVote
from app.models.location_context import LocationContext
from app.models.player import Player
from app.models.submission import Submission
from app.models.target import Target

logger = logging.getLogger(__name__)

class GameRepository:
    @staticmethod
    async def create(game: Game) -> Game:
        db = get_db()
        game_dict = game.model_dump(by_alias=True)
        await db.games.insert_one(game_dict)
        return game

    @staticmethod
    async def get_by_id(game_id: str) -> Optional[Game]:
        db = get_db()
        doc = await db.games.find_one({"_id": game_id})
        if doc:
            return Game(**doc)
        return None

    @staticmethod
    async def get_by_join_code(join_code: str) -> Optional[Game]:
        db = get_db()
        doc = await db.games.find_one({"join_code": join_code})
        if doc:
            return Game(**doc)
        return None

    @staticmethod
    async def get_recent_target_sets_for_context(
        game_id: str,
        context: LocationContext,
        *,
        limit: int = 3,
    ) -> list[list[str]]:
        db = get_db()
        cursor = (
            db.games.find(
                {
                    "_id": {"$ne": game_id},
                    "status": {"$in": ["ACTIVE", "FINISHED"]},
                    "location_context.context_mode": "contextual",
                    "location_context.context_provider": context.context_provider,
                    "location_context.locality": context.locality,
                    "location_context.region": context.region,
                    "location_context.environment_type": context.environment_type,
                    "location_context.nearby_categories": context.nearby_categories,
                },
                {
                    "targets.class_name": 1,
                    "location_context.nearby_feature_counts": 1,
                },
            )
            .sort("created_at", -1)
            .limit(limit * 10)
        )
        target_sets: list[list[str]] = []
        async for document in cursor:
            stored_context = document.get("location_context") or {}
            if stored_context.get("nearby_feature_counts", {}) != context.nearby_feature_counts:
                continue
            target_classes = [
                target.get("class_name")
                for target in document.get("targets", [])
                if isinstance(target, dict) and isinstance(target.get("class_name"), str)
            ]
            if len(target_classes) != 5:
                continue
            target_sets.append(target_classes)
            if len(target_sets) == limit:
                break
        return target_sets

    @staticmethod
    async def update(
        game: Game,
        *,
        expected_targets: Optional[List[Target]] = None,
    ) -> bool:
        db = get_db()
        game_dict = game.model_dump(by_alias=True)
        game_dict.pop("target_change_vote", None)
        expected_target_docs = [
            target.model_dump(mode="python")
            for target in (
                expected_targets if expected_targets is not None else game.targets
            )
        ]
        query = {"_id": game.id, "targets": expected_target_docs}
        vote = game.target_change_vote
        if vote is None:
            query["$or"] = [
                {"target_change_vote": None},
                {"target_change_vote": {"$exists": False}},
            ]
        else:
            query["target_change_vote"] = vote.model_dump(mode="python")
        result = await db.games.update_one(
            query,
            {
                "$set": {
                    key: value
                    for key, value in game_dict.items()
                    if key != "_id"
                }
            },
        )
        return result.modified_count > 0

    @staticmethod
    async def activate_with_targets(game: Game, expected_targets: List[Target]) -> bool:
        db = get_db()
        game_dict = game.model_dump(by_alias=True)
        game_dict.pop("target_change_vote", None)
        expected_target_docs = [
            target.model_dump(mode="python") for target in expected_targets
        ]
        result = await db.games.update_one(
            {
                "_id": game.id,
                "creator_id": game.creator_id,
                "status": "WAITING",
                "targets": expected_target_docs,
                "$or": [
                    {"target_change_vote": None},
                    {"target_change_vote": {"$exists": False}},
                ],
            },
            {
                "$set": {
                    key: value for key, value in game_dict.items() if key != "_id"
                }
            },
        )
        return result.modified_count == 1

    @staticmethod
    async def create_target_vote(game_id: str, vote: TargetChangeVote) -> bool:
        db = get_db()
        result = await db.games.update_one(
            {
                "_id": game_id,
                "status": "ACTIVE",
                "target_change_vote.status": {"$ne": "pending"},
                "targets.id": vote.target_id,
            },
            {"$set": {"target_change_vote": vote.model_dump(mode="python")}},
        )
        return result.modified_count == 1

    @staticmethod
    async def update_pending_target_vote(
        game_id: str,
        vote: TargetChangeVote,
        expected_revision: int,
    ) -> bool:
        db = get_db()
        result = await db.games.update_one(
            {
                "_id": game_id,
                "target_change_vote.proposal_id": vote.proposal_id,
                "target_change_vote.status": "pending",
                "target_change_vote.revision": expected_revision,
            },
            {"$set": {"target_change_vote": vote.model_dump(mode="python")}},
        )
        return result.modified_count == 1

    @staticmethod
    async def finalize_target_vote(
        game_id: str,
        vote: TargetChangeVote,
        expected_revision: int,
    ) -> bool:
        db = get_db()
        replacement_target = vote.replacement_target
        if replacement_target is None:
            return False
        result = await db.games.update_one(
            {
                "_id": game_id,
                "status": "ACTIVE",
                "target_change_vote.proposal_id": vote.proposal_id,
                "target_change_vote.status": "pending",
                "target_change_vote.revision": expected_revision,
                "targets.id": vote.target_id,
            },
            {
                "$set": {
                    "targets.$": replacement_target.model_dump(mode="python"),
                    "target_change_vote": vote.model_dump(mode="python"),
                }
            },
        )
        return result.modified_count == 1

class PlayerRepository:
    @staticmethod
    async def create(player: Player) -> Player:
        db = get_db()
        player_dict = player.model_dump(by_alias=True)
        await db.players.insert_one(player_dict)
        return player

    @staticmethod
    async def get_by_id(player_id: str) -> Optional[Player]:
        db = get_db()
        doc = await db.players.find_one({"_id": player_id})
        if doc:
            return Player(**doc)
        return None

    @staticmethod
    async def get_by_game(game_id: str) -> List[Player]:
        db = get_db()
        cursor = db.players.find({"game_id": game_id}).sort("score", -1)
        return [Player(**doc) async for doc in cursor]

    @staticmethod
    async def touch_last_seen(game_id: str, player_id: str) -> bool:
        db = get_db()
        result = await db.players.update_one(
            {"_id": player_id, "game_id": game_id},
            {"$set": {"last_seen_at": datetime.now(timezone.utc)}},
        )
        return result.matched_count == 1

    @staticmethod
    async def update(player: Player) -> bool:
        db = get_db()
        player_dict = player.model_dump(by_alias=True)
        result = await db.players.replace_one({"_id": player.id}, player_dict)
        if result.matched_count == 0:
            logger.warning(
                "Player location update matched no stored player document player_id=%s",
                player.id,
            )
        return result.matched_count > 0

    @staticmethod
    async def complete_target_once(player_id: str, target_id: str) -> Optional[Player]:
        db = get_db()
        result = await db.players.update_one(
            {"_id": player_id, "completed_targets": {"$ne": target_id}},
            {"$addToSet": {"completed_targets": target_id}, "$inc": {"score": 1}},
        )
        if result.modified_count != 1:
            return None
        return await PlayerRepository.get_by_id(player_id)

class SubmissionRepository:
    @staticmethod
    async def create(submission: Submission) -> Submission:
        db = get_db()
        sub_dict = submission.model_dump(by_alias=True)
        await db.submissions.insert_one(sub_dict)
        return submission
