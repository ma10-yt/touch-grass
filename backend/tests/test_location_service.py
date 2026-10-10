from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.models.game import Game, GameState
from app.models.player import Player
from app.models.target import Target
from app.services.location_service import LocationService


def make_player(lat=43.0, lon=-79.0, movement=0.0):
    now = datetime.now(timezone.utc)
    return Player(
        _id="player-id",
        game_id="game-id",
        display_name="Walker",
        latest_lat=lat,
        latest_lon=lon,
        last_location_update=now - timedelta(seconds=20),
        location_accuracy_m=3,
        movement_meters=movement,
        movement_last_lat=lat,
        movement_last_lon=lon,
        movement_last_sample_at=now - timedelta(seconds=20),
        movement_last_accuracy_m=3,
    )


def install_repositories(monkeypatch, player, game=None):
    async def get_player(player_id):
        return player if player_id == player.id else None

    async def update_player(updated_player):
        assert updated_player is player
        return True

    async def get_game(game_id):
        return game if game and game_id == game.id else None

    async def update_game(updated_game):
        assert updated_game is game
        return True

    monkeypatch.setattr(PlayerRepository, "get_by_id", get_player)
    monkeypatch.setattr(PlayerRepository, "update", update_player)
    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(GameRepository, "update", update_game)


@pytest.mark.asyncio
async def test_player_can_validate_far_beyond_the_former_starting_radius():
    game = Game(
        _id="game-id",
        join_code="JOIN1",
        creator_id="player-id",
        center_lat=43.0,
        center_lon=-79.0,
    )
    player = make_player(lat=43.01, lon=-79.0)

    valid, message, _ = await LocationService.validate_player_location(player)

    assert valid is True
    assert message == "Location valid"


@pytest.mark.asyncio
async def test_route_movement_accumulates_out_and_back(monkeypatch):
    player = make_player()
    now = datetime.now(timezone.utc)
    player.last_location_update = now - timedelta(seconds=55)
    player.movement_last_sample_at = now - timedelta(seconds=55)
    install_repositories(monkeypatch, player)
    start_lat = player.latest_lat
    results = []

    for step, meters in enumerate((20, 40, 60, 80, 100, 80, 60, 40, 20, 0), start=1):
        results.append(
            await LocationService.update_location(
                "game-id",
                player.id,
                start_lat + meters / 111_000,
                -79.0,
                3,
                now - timedelta(seconds=55 - step * 5),
            )
        )

    assert all(result.accepted for result in results)
    assert player.movement_meters >= 80
    assert results[-1].movement_meters > results[3].movement_meters
    assert player.movement_last_lat is not None


@pytest.mark.asyncio
async def test_small_gps_jitter_does_not_accumulate_movement(monkeypatch):
    player = make_player()
    install_repositories(monkeypatch, player)
    now = datetime.now(timezone.utc)
    start_lat, start_lon = player.latest_lat, player.latest_lon

    for index in range(1, 5):
        result = await LocationService.update_location(
            "game-id",
            player.id,
            start_lat + index * 0.000002,
            start_lon,
            8,
            now - timedelta(seconds=20 - index * 5),
        )
        assert result.accepted is True

    assert player.movement_meters == 0
    assert player.movement_last_lat == start_lat


@pytest.mark.asyncio
async def test_213_meter_accuracy_is_accepted_as_approximate_location(monkeypatch):
    player = make_player()
    install_repositories(monkeypatch, player)

    result = await LocationService.update_location(
        "game-id",
        player.id,
        43.0,
        -79.0,
        213,
        datetime.now(timezone.utc),
    )

    assert result.accepted is True
    assert player.location_accuracy_m == 213
    assert player.last_location_update is not None
    assert result.movement_added_meters == 0
    valid, message, _ = await LocationService.validate_player_location(player)
    assert valid is True
    assert message == "Location valid"


@pytest.mark.asyncio
async def test_consistent_200_meter_trip_counts_with_213_meter_accuracy(monkeypatch):
    player = make_player()
    now = datetime.now(timezone.utc)
    player.last_location_update = now - timedelta(seconds=55)
    player.movement_last_sample_at = now - timedelta(seconds=55)
    install_repositories(monkeypatch, player)
    start_lat = player.latest_lat

    results = []
    for step in range(1, 11):
        results.append(
            await LocationService.update_location(
                "game-id",
                player.id,
                start_lat + step * 20 / 111_000,
                -79.0,
                213,
                now - timedelta(seconds=55 - step * 5),
            )
        )

    assert all(result.accepted for result in results)
    assert player.movement_meters >= 50
    assert player.movement_meters < 200


@pytest.mark.asyncio
async def test_wide_stationary_jitter_does_not_accumulate_movement(monkeypatch):
    player = make_player()
    now = datetime.now(timezone.utc)
    player.last_location_update = now - timedelta(seconds=55)
    player.movement_last_sample_at = now - timedelta(seconds=55)
    install_repositories(monkeypatch, player)
    start_lat = player.latest_lat

    for step, offset in enumerate((0, 8, -6, 10, -9, 5, -3, 7, 0, 4), start=1):
        result = await LocationService.update_location(
            "game-id",
            player.id,
            start_lat + offset / 111_000,
            -79.0,
            213,
            now - timedelta(seconds=55 - step * 5),
        )
        assert result.accepted is True

    assert player.movement_meters == 0


@pytest.mark.asyncio
async def test_stale_and_poor_accuracy_samples_are_not_counted(monkeypatch):
    player = make_player()
    install_repositories(monkeypatch, player)
    old_location = (
        player.latest_lat,
        player.latest_lon,
        player.last_location_update,
        player.location_accuracy_m,
    )
    stale = await LocationService.update_location(
        "game-id",
        player.id,
        43.001,
        -79.0,
        2,
        datetime.now(timezone.utc) - timedelta(seconds=settings.LOCATION_MAX_AGE_SECONDS + 10),
    )

    assert stale.accepted is False
    assert "stale" in stale.message
    assert player.movement_meters == 0

    poor_fix = await LocationService.update_location(
        "game-id",
        player.id,
        43.001,
        -79.0,
        settings.GPS_MAX_ACCURACY_METERS + 5,
        datetime.now(timezone.utc),
    )
    game = Game(_id="game-id", join_code="JOIN1", creator_id=player.id)
    valid, message, _ = await LocationService.validate_player_location(player)

    assert poor_fix.accepted is False
    assert "accuracy" in poor_fix.message
    assert player.movement_meters == 0
    assert (
        player.latest_lat,
        player.latest_lon,
        player.last_location_update,
        player.location_accuracy_m,
    ) == old_location
    assert valid is True
    assert message == "Location valid"


@pytest.mark.asyncio
async def test_unmatched_player_document_is_not_reported_as_accepted(monkeypatch):
    player = make_player()
    assert player.latest_lat is not None
    assert player.latest_lon is not None
    old_location = (
        player.latest_lat,
        player.latest_lon,
        player.last_location_update,
        player.location_accuracy_m,
        player.movement_meters,
        player.movement_last_lat,
        player.movement_last_lon,
        player.movement_last_sample_at,
        player.movement_last_accuracy_m,
    )
    install_repositories(monkeypatch, player)

    async def update_missing_player(_):
        return False

    monkeypatch.setattr(PlayerRepository, "update", update_missing_player)
    result = await LocationService.update_location(
        "game-id",
        player.id,
        player.latest_lat + 0.0002,
        player.latest_lon,
        1,
        datetime.now(timezone.utc),
    )

    assert result.accepted is False
    assert "could not be saved" in result.message
    assert result.movement_added_meters == 0
    assert (
        player.latest_lat,
        player.latest_lon,
        player.last_location_update,
        player.location_accuracy_m,
        player.movement_meters,
        player.movement_last_lat,
        player.movement_last_lon,
        player.movement_last_sample_at,
        player.movement_last_accuracy_m,
    ) == old_location


@pytest.mark.asyncio
async def test_implausible_gps_jump_is_ignored(monkeypatch):
    player = make_player()
    install_repositories(monkeypatch, player)
    old_lat, old_lon = player.latest_lat, player.latest_lon

    result = await LocationService.update_location(
        "game-id",
        player.id,
        old_lat + 0.01,
        old_lon,
        2,
        datetime.now(timezone.utc) - timedelta(seconds=5),
    )

    assert result.accepted is True
    assert result.movement_accepted is False
    assert "implausible" in result.message
    assert player.movement_meters == 0
    assert player.latest_lat != old_lat
    assert player.movement_last_lat == old_lat


@pytest.mark.asyncio
async def test_location_updates_never_finish_a_game(monkeypatch):
    player = make_player(movement=1000)
    targets = [
        Target(id=f"target-{index}", name=f"Target {index}", class_name=cls, hint="Nearby")
        for index, cls in enumerate(
            ("bicycle", "bench", "bird", "car", "backpack"), start=1
        )
    ]
    player.completed_targets = [target.id for target in targets]
    game = Game(
        _id="game-id",
        join_code="JOIN1",
        creator_id=player.id,
        status=GameState.ACTIVE,
        targets=targets,
    )
    install_repositories(monkeypatch, player, game)

    first = await LocationService.update_location(
        "game-id",
        player.id,
        player.latest_lat + 0.0004,
        player.latest_lon,
        1,
        datetime.now(timezone.utc) - timedelta(seconds=15),
    )
    result = await LocationService.update_location(
        "game-id",
        player.id,
        player.latest_lat + 0.00002,
        player.latest_lon,
        1,
        datetime.now(timezone.utc) - timedelta(seconds=10),
    )

    assert first.accepted is True
    assert result.accepted is True
    assert result.game_finished is False
    assert game.status == GameState.ACTIVE
