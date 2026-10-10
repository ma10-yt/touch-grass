from datetime import datetime, timedelta, timezone

import pytest

from app.models.game import Game, GameState
from app.models.location_context import LocationContext
from app.models.player import Player
from app.models.target import Target
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.services.game_service import GameService
from app.services.location_context_service import location_context_service
from app.ai.gemma_generator import target_generator
from app.ai.yolo_verifier import image_verifier


def make_active_game():
    targets = [
        Target(id=f"old-{index}", name=name, class_name=class_name, hint="Old area")
        for index, (name, class_name) in enumerate(
            (
                ("Bench", "bench"),
                ("Bicycle", "bicycle"),
                ("Bird", "bird"),
                ("Car", "car"),
                ("Backpack", "backpack"),
            )
        )
    ]
    return Game(
        _id="area-game",
        join_code="AREA1",
        creator_id="host",
        status=GameState.ACTIVE,
        targets=targets,
        location_context=LocationContext(
            locality="Starting District",
            context_mode="contextual",
            context_status="available",
        ),
        location_context_lat=43.0,
        location_context_lon=-79.0,
        location_context_updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )


@pytest.mark.asyncio
async def test_stable_new_area_refreshes_map_context_and_shared_targets(monkeypatch):
    game = make_active_game()
    initial_targets = list(game.targets)
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
    resolved = []
    generated = []

    async def get_game(game_id):
        assert game_id == game.id
        return game

    async def update_game(updated_game, *, expected_targets=None):
        assert updated_game is game
        if expected_targets is not None:
            assert expected_targets == initial_targets
        return True

    async def get_players(game_id):
        assert game_id == game.id
        return [host, guest]

    async def resolve(lat, lon):
        resolved.append((lat, lon))
        return LocationContext(
            locality="New District",
            environment_type="urban",
            nearby_categories=["public_transport"],
            context_mode="contextual",
            context_status="available",
        )

    async def generate_targets(
        game_id, context, count, excluded_classes, supported_classes=None
    ):
        generated.append((game_id, context.locality, count, excluded_classes))
        assert set(supported_classes) >= {
            "bus",
            "traffic light",
            "stop sign",
            "dog",
            "umbrella",
        }
        return [
            Target(
                id=f"new-{index}",
                name=f"New {class_name}",
                class_name=class_name,
                hint="Nearby in the new area.",
            )
            for index, class_name in enumerate(
                ("bus", "traffic light", "stop sign", "dog", "umbrella")
            )
        ]

    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(GameRepository, "update", update_game)
    monkeypatch.setattr(PlayerRepository, "get_by_game", get_players)
    monkeypatch.setattr(location_context_service, "resolve", resolve)
    monkeypatch.setattr(target_generator, "generate_targets", generate_targets)
    monkeypatch.setattr(
        image_verifier,
        "get_supported_target_classes",
        lambda: ("bus", "traffic light", "stop sign", "dog", "umbrella"),
    )

    new_lat = 43.002
    first_updated, first_notice = await GameService.refresh_area_context(
        game.id, new_lat, -79.0
    )
    second_updated, notice = await GameService.refresh_area_context(
        game.id, new_lat + 0.00001, -79.0
    )

    assert first_updated is False
    assert first_notice is None
    assert second_updated is True
    assert notice is None
    assert len(resolved) == 1
    assert resolved[0][0] == new_lat + 0.00001
    assert generated[0][0:3] == (game.id, "New District", 5)
    assert {"bench", "bicycle"} <= generated[0][3]
    assert [target.id for target in game.targets if target.id.startswith("old-")] == [
        "old-0",
        "old-1",
    ]
    assert len(game.targets) == 5
    assert game.location_context.locality == "New District"
    assert game.location_context_lat == new_lat + 0.00001
    assert host.completed_targets[0] in {target.id for target in game.targets}
    assert guest.completed_targets[0] in {target.id for target in game.targets}


@pytest.mark.asyncio
async def test_area_context_does_not_refresh_from_nearby_jitter(monkeypatch):
    game = make_active_game()
    game.context_candidate_lat = None
    game.context_candidate_lon = None
    game.context_candidate_count = 0

    async def get_game(_):
        return game

    async def update_game(_):
        return True

    async def should_not_resolve(_, __):
        raise AssertionError("Nearby jitter must not trigger OSM refresh")

    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(GameRepository, "update", update_game)
    monkeypatch.setattr(location_context_service, "resolve", should_not_resolve)

    first = await GameService.refresh_area_context(game.id, 43.0002, -79.0)
    second = await GameService.refresh_area_context(game.id, 43.00025, -79.0)

    assert first == (False, None)
    assert second == (False, None)
    assert game.location_context.locality == "Starting District"
