import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.ai.yolo_verifier import Detection
from app.config import settings
from app.db.repositories.repositories import (
    GameRepository,
    PlayerRepository,
    SubmissionRepository,
)
from app.models.game import Game, GameState, TargetChangeVote
from app.models.location_context import LocationContext
from app.models.player import Player
from app.models.target import Target
from app.services.location_service import LocationService
from app.services.target_change_service import TargetChangeError, TargetChangeService
from app.services.verification_service import VerificationService


def make_game_and_players(player_count=2):
    game = Game(
        _id="vote-game",
        join_code="VOTE1",
        creator_id="player-0",
        status=GameState.ACTIVE,
        targets=[
            Target(
                id=f"target-{index}",
                name=name,
                class_name=class_name,
                hint="Find it nearby.",
            )
            for index, (name, class_name) in enumerate(
                (
                    ("Backpack", "backpack"),
                    ("Bench", "bench"),
                    ("Bird", "bird"),
                    ("Bicycle", "bicycle"),
                    ("Car", "car"),
                )
            )
        ],
        location_context=LocationContext(
            locality="Current District",
            context_mode="contextual",
            context_status="available",
        ),
    )
    now = datetime.now(timezone.utc)
    players = [
        Player(
            _id=f"player-{index}",
            game_id=game.id,
            display_name=f"Player {index}",
            last_seen_at=now,
        )
        for index in range(player_count)
    ]
    return game, players


def install_memory_repositories(monkeypatch, game, players):
    generated = []

    async def get_game(game_id):
        return Game.model_validate(game.model_dump(by_alias=True)) if game_id == game.id else None

    async def get_players(game_id):
        return players if game_id == game.id else []

    async def touch(game_id, player_id):
        return game_id == game.id and any(item.id == player_id for item in players)

    async def create_vote(game_id, vote):
        if game_id != game.id or (
            game.target_change_vote
            and game.target_change_vote.status == "pending"
        ):
            return False
        game.target_change_vote = TargetChangeVote.model_validate(
            vote.model_dump()
        )
        return True

    async def update_vote(game_id, vote, expected_revision):
        await asyncio.sleep(0)
        current = game.target_change_vote
        if (
            game_id != game.id
            or current is None
            or current.status != "pending"
            or current.proposal_id != vote.proposal_id
            or current.revision != expected_revision
        ):
            return False
        game.target_change_vote = TargetChangeVote.model_validate(
            vote.model_dump()
        )
        return True

    async def finalize_vote(game_id, vote, expected_revision):
        await asyncio.sleep(0)
        current = game.target_change_vote
        if (
            game_id != game.id
            or current is None
            or current.status != "pending"
            or current.proposal_id != vote.proposal_id
            or current.revision != expected_revision
        ):
            return False
        target_index = next(
            (
                index for index, target in enumerate(game.targets)
                if target.id == vote.target_id
            ),
            None,
        )
        if target_index is None or vote.replacement_target is None:
            return False
        game.targets[target_index] = vote.replacement_target
        game.target_change_vote = TargetChangeVote.model_validate(
            vote.model_dump()
        )
        return True

    async def generate_targets(**kwargs):
        generated.append(kwargs)
        class_name = next(iter(kwargs["supported_classes"]))
        return [
            Target(
                id="replacement-id",
                name=class_name.title(),
                class_name=class_name,
                hint="Find the replacement in the current area.",
            )
        ]

    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(GameRepository, "create_target_vote", create_vote)
    monkeypatch.setattr(GameRepository, "update_pending_target_vote", update_vote)
    monkeypatch.setattr(GameRepository, "finalize_target_vote", finalize_vote)
    monkeypatch.setattr(PlayerRepository, "get_by_game", get_players)
    monkeypatch.setattr(PlayerRepository, "touch_last_seen", touch)
    monkeypatch.setattr(
        "app.services.target_change_service.image_verifier.get_supported_target_classes",
        lambda: ("backpack", "bench", "bird", "bicycle", "car", "dog", "bus"),
    )
    monkeypatch.setattr(
        "app.services.target_change_service.image_verifier.class_id_for_name",
        lambda class_name: {
            "backpack": 24,
            "bench": 13,
            "bird": 14,
            "bicycle": 1,
            "car": 2,
            "dog": 16,
            "bus": 5,
        }.get(class_name),
    )
    monkeypatch.setattr(
        "app.services.target_change_service.target_generator.generate_targets",
        generate_targets,
    )
    return generated


@pytest.mark.asyncio
async def test_unanimous_approval_replaces_only_selected_unfinished_target(monkeypatch):
    game, players = make_game_and_players()
    players[1].completed_targets.append("target-1")
    generated = install_memory_repositories(monkeypatch, game, players)

    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    assert pending.target_change_vote.status == "pending"
    assert pending.target_change_vote.approved_player_ids == [players[0].id]

    result = await TargetChangeService.cast_vote(
        game.id,
        pending.target_change_vote.proposal_id,
        players[1].id,
        True,
    )

    assert result.target_change_vote.status == "approved"
    assert result.targets[0].id == "replacement-id"
    assert players[1].completed_targets == ["target-1"]
    assert [target.id for target in result.targets[1:]] == [
        "target-1",
        "target-2",
        "target-3",
        "target-4",
    ]
    assert generated[0]["context"].locality == "Current District"
    assert generated[0]["count"] == 1
    assert "backpack" in generated[0]["excluded_classes"]


@pytest.mark.asyncio
async def test_rejection_cancels_and_keeps_original_target(monkeypatch):
    game, players = make_game_and_players()
    generated = install_memory_repositories(monkeypatch, game, players)
    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")

    result = await TargetChangeService.cast_vote(
        game.id,
        pending.target_change_vote.proposal_id,
        players[0].id,
        False,
    )

    assert result.target_change_vote.status == "rejected"
    assert result.target_change_vote.message.startswith("A teammate rejected")
    assert result.targets[0].id == "target-0"
    assert generated == []


@pytest.mark.asyncio
async def test_duplicate_approvals_are_idempotent_and_proposal_persists(monkeypatch):
    game, players = make_game_and_players()
    generated = install_memory_repositories(monkeypatch, game, players)
    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    proposal_id = pending.target_change_vote.proposal_id
    refreshed_pending = Game.model_validate(game.model_dump(by_alias=True))

    first = await TargetChangeService.cast_vote(
        game.id, proposal_id, players[1].id, True
    )
    duplicate = await TargetChangeService.cast_vote(
        game.id, proposal_id, players[1].id, True
    )
    restored = Game.model_validate(game.model_dump(by_alias=True))

    assert first.target_change_vote.status == "approved"
    assert refreshed_pending.target_change_vote.status == "pending"
    assert duplicate.targets[0].id == "replacement-id"
    assert restored.target_change_vote.proposal_id == proposal_id
    assert len(restored.target_change_vote.approved_player_ids) == 2
    assert len(generated) == 1


@pytest.mark.asyncio
async def test_stale_players_do_not_block_solo_or_active_team_vote(monkeypatch):
    game, players = make_game_and_players()
    players[1].last_seen_at = datetime.now(timezone.utc) - timedelta(minutes=3)
    install_memory_repositories(monkeypatch, game, players)

    result = await TargetChangeService.request(game.id, players[0].id, "target-0")

    assert result.target_change_vote.status == "approved"
    assert result.target_change_vote.eligible_player_ids == [players[0].id]
    assert result.targets[0].id == "replacement-id"


@pytest.mark.asyncio
async def test_solo_requester_approval_is_sufficient(monkeypatch):
    game, players = make_game_and_players(player_count=1)
    install_memory_repositories(monkeypatch, game, players)

    result = await TargetChangeService.request(game.id, players[0].id, "target-0")

    assert result.target_change_vote.status == "approved"
    assert result.targets[0].id == "replacement-id"
    assert players[0].score == 0
    assert players[0].completed_targets == []


@pytest.mark.asyncio
async def test_repeated_request_reuses_pending_proposal_without_duplicate_approval(monkeypatch):
    game, players = make_game_and_players()
    install_memory_repositories(monkeypatch, game, players)
    first = await TargetChangeService.request(game.id, players[0].id, "target-0")

    repeated = await TargetChangeService.request(game.id, players[0].id, "target-0")

    assert repeated.target_change_vote.proposal_id == first.target_change_vote.proposal_id
    assert repeated.target_change_vote.approved_player_ids == [players[0].id]


@pytest.mark.asyncio
async def test_already_completed_target_cannot_be_replaced(monkeypatch):
    game, players = make_game_and_players()
    players[1].completed_targets.append("target-0")
    install_memory_repositories(monkeypatch, game, players)

    with pytest.raises(TargetChangeError, match="already been completed"):
        await TargetChangeService.request(game.id, players[0].id, "target-0")

    assert game.targets[0].id == "target-0"
    assert game.target_change_vote is None


@pytest.mark.asyncio
async def test_unsupported_detector_replacement_leaves_original_target_intact(monkeypatch):
    game, players = make_game_and_players(player_count=1)
    generated = install_memory_repositories(monkeypatch, game, players)
    monkeypatch.setattr(
        "app.services.target_change_service.target_generator.generate_targets",
        lambda **_kwargs: asyncio.sleep(
            0,
            result=[
                Target(
                    id="unsupported-replacement",
                    name="Book",
                    class_name="book",
                    hint="A book.",
                )
            ],
        ),
    )

    result = await TargetChangeService.request(game.id, players[0].id, "target-0")

    assert result.target_change_vote.status == "unavailable"
    assert "supported classes" in result.target_change_vote.message
    assert result.targets[0].id == "target-0"
    assert generated == []


@pytest.mark.asyncio
async def test_concurrent_votes_finalize_replacement_once(monkeypatch):
    game, players = make_game_and_players(player_count=3)
    generated = install_memory_repositories(monkeypatch, game, players)
    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    proposal_id = pending.target_change_vote.proposal_id

    await asyncio.gather(
        TargetChangeService.cast_vote(game.id, proposal_id, players[1].id, True),
        TargetChangeService.cast_vote(game.id, proposal_id, players[2].id, True),
    )

    assert game.target_change_vote.status == "approved"
    assert len(generated) == 1
    assert sum(target.id == "replacement-id" for target in game.targets) == 1


@pytest.mark.asyncio
async def test_replacement_excludes_class_of_replaced_target_server_side(monkeypatch):
    """Regression: replacement class is excluded server-side via supported_classes."""
    game, players = make_game_and_players()
    players[1].completed_targets.append("target-1")

    # Capture the arguments passed to generate_targets
    captured_kwargs = {}

    async def capture_generate_targets(**kwargs):
        captured_kwargs.update(kwargs)
        # Return a valid alternative (dog) - backpack is excluded from supported_classes
        return [
            Target(
                id="replacement-valid",
                name="Dog",
                class_name="dog",
                hint="Valid replacement",
            )
        ]

    generated = install_memory_repositories(monkeypatch, game, players)
    monkeypatch.setattr(
        "app.services.target_change_service.target_generator.generate_targets",
        capture_generate_targets,
    )

    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    result = await TargetChangeService.cast_vote(
        game.id,
        pending.target_change_vote.proposal_id,
        players[1].id,
        True,
    )

    # Should succeed with valid replacement
    assert result.target_change_vote.status == "approved"
    assert result.targets[0].class_name == "dog"
    # Verify server-side exclusion: backpack excluded from supported_classes
    assert "backpack" not in captured_kwargs.get("supported_classes", set())
    # backpack should also be in excluded_classes
    assert "backpack" in captured_kwargs.get("excluded_classes", set())


@pytest.mark.asyncio
async def test_replacement_fails_when_no_valid_alternative_exists(monkeypatch):
    """Regression: clear error when no valid alternative class exists."""
    game, players = make_game_and_players(player_count=1)

    # Only "backpack" is supported, but it's the class being replaced
    # So alternatives = supported - completed_classes = empty set
    monkeypatch.setattr(
        "app.services.target_change_service.image_verifier.get_supported_target_classes",
        lambda: ("backpack",),
    )
    monkeypatch.setattr(
        "app.services.target_change_service.image_verifier.class_id_for_name",
        lambda class_name: 24 if class_name == "backpack" else None,
    )

    # Set up minimal mocks without install_memory_repositories (which captures wrong supported_classes)
    async def get_game(game_id):
        return Game.model_validate(game.model_dump(by_alias=True)) if game_id == game.id else None

    async def get_players(game_id):
        return players if game_id == game.id else []

    async def touch(game_id, player_id):
        return game_id == game.id and any(item.id == player_id for item in players)

    async def create_vote(game_id, vote):
        if game_id != game.id or (
            game.target_change_vote
            and game.target_change_vote.status == "pending"
        ):
            return False
        game.target_change_vote = TargetChangeVote.model_validate(vote.model_dump())
        return True

    async def update_vote(game_id, vote, expected_revision):
        await asyncio.sleep(0)
        current = game.target_change_vote
        if (
            game_id != game.id
            or current is None
            or current.status != "pending"
            or current.proposal_id != vote.proposal_id
            or current.revision != expected_revision
        ):
            return False
        game.target_change_vote = TargetChangeVote.model_validate(vote.model_dump())
        return True

    async def finalize_vote(game_id, vote, expected_revision):
        await asyncio.sleep(0)
        current = game.target_change_vote
        if (
            game_id != game.id
            or current is None
            or current.status != "pending"
            or current.proposal_id != vote.proposal_id
            or current.revision != expected_revision
        ):
            return False
        target_index = next(
            (
                index for index, target in enumerate(game.targets)
                if target.id == vote.target_id
            ),
            None,
        )
        if target_index is None or vote.replacement_target is None:
            return False
        game.targets[target_index] = vote.replacement_target
        game.target_change_vote = TargetChangeVote.model_validate(vote.model_dump())
        return True

    generated = []

    async def generate_targets(**kwargs):
        generated.append(kwargs)
        # Generator would raise TargetGenerationError when no candidates available
        from app.ai.gemma_generator import TargetGenerationError
        raise TargetGenerationError("No supported YOLO classes available")

    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(GameRepository, "create_target_vote", create_vote)
    monkeypatch.setattr(GameRepository, "update_pending_target_vote", update_vote)
    monkeypatch.setattr(GameRepository, "finalize_target_vote", finalize_vote)
    monkeypatch.setattr(PlayerRepository, "get_by_game", get_players)
    monkeypatch.setattr(PlayerRepository, "touch_last_seen", touch)
    monkeypatch.setattr(
        "app.services.target_change_service.target_generator.generate_targets",
        generate_targets,
    )

    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    result = await TargetChangeService.cast_vote(
        game.id,
        pending.target_change_vote.proposal_id,
        players[0].id,
        True,
    )

    # Should fail with unavailable status since no valid alternative
    assert result.target_change_vote.status == "unavailable"
    assert "supported" in result.target_change_vote.message.lower() or "classes" in result.target_change_vote.message.lower()
    assert result.targets[0].id == "target-0"  # Original target unchanged
    assert len(generated) == 1  # Generator was called but raised error


@pytest.mark.asyncio
async def test_replacement_still_requires_normal_image_verification(monkeypatch, tmp_path):
    game, players = make_game_and_players(player_count=1)
    install_memory_repositories(monkeypatch, game, players)
    players[0].last_location_update = datetime.now(timezone.utc)
    pending = await TargetChangeService.request(game.id, players[0].id, "target-0")
    replacement = pending.targets[0]
    monkeypatch.setattr(
        GameRepository,
        "get_by_id",
        lambda game_id: asyncio.sleep(0, result=game),
    )
    monkeypatch.setattr(
        PlayerRepository,
        "get_by_id",
        lambda player_id: asyncio.sleep(0, result=players[0]),
    )

    async def valid_location(_player):
        return True, "Location valid", 2.0

    async def detect(_image_path, class_name):
        return Detection(class_name=class_name, confidence=0.91)

    async def complete(_player_id, target_id):
        players[0].completed_targets.append(target_id)
        players[0].score += 1
        return players[0]

    async def save(submission):
        return submission

    async def update_game(_game):
        return True

    monkeypatch.setattr(LocationService, "validate_player_location", valid_location)
    monkeypatch.setattr(PlayerRepository, "complete_target_once", complete)
    monkeypatch.setattr(SubmissionRepository, "create", save)
    monkeypatch.setattr(GameRepository, "update", update_game)
    monkeypatch.setattr(
        "app.services.verification_service.image_verifier.detect_image", detect
    )
    image = tmp_path / "verified.jpg"
    image.write_bytes(b"image bytes")

    result = await VerificationService.process_submission(
        game.id, players[0].id, replacement.id, str(image)
    )

    assert result.verified is True
    assert replacement.id in players[0].completed_targets
