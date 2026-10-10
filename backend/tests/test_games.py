import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace
from app.main import app
from app.models.game import Game, GameState, TargetChangeVote
from app.models.location_context import LocationContext, TargetCandidate
from app.models.player import Player
from app.models.target import Target
from app.models.target import OUTDOOR_CANDIDATE_CLASSES
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.config import settings
from app.services.game_service import (
    GameService,
    GameStartError,
    _repeated_context_classes_to_avoid,
)
from app.services.target_change_service import TargetChangeService
from app.services.location_context_service import location_context_service
from app.services.location_service import LocationService, LocationUpdateResult
from app.ai.gemma_generator import target_generator
from app.ai.yolo_verifier import image_verifier
from datetime import datetime, timezone

client = TestClient(app)


@pytest.mark.asyncio
async def test_activate_with_targets_atomically_persists_five_targets(monkeypatch):
    game = make_game()
    initial_targets = list(game.targets)
    game.targets = [
        Target(
            id=f"target-{index}",
            name=name,
            class_name=class_name,
            hint="Nearby.",
        )
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
    game.status = GameState.ACTIVE
    game.start_time = datetime.now(timezone.utc)
    operations = []

    class GamesCollection:
        async def update_one(self, query, update):
            operations.append((query, update))
            return SimpleNamespace(modified_count=1)

    monkeypatch.setattr(
        "app.db.repositories.repositories.get_db",
        lambda: SimpleNamespace(games=GamesCollection()),
    )

    saved = await GameRepository.activate_with_targets(game, initial_targets)

    assert saved is True
    assert len(operations) == 1
    query, update = operations[0]
    assert query["status"] == "WAITING"
    assert query["creator_id"] == game.creator_id
    assert query["targets"] == []
    assert update["$set"]["status"] == GameState.ACTIVE
    assert len(update["$set"]["targets"]) == 5


def test_create_game(monkeypatch):
    payload = {
        "creator_name": "TestUser",
        "lat": 10.0,
        "lon": 20.0
    }

    async def create_game(creator_name, lat, lon):
        assert creator_name == payload["creator_name"]
        return Game(
            _id="game-id",
            join_code="JOIN1",
            creator_id="creator-id",
            status=GameState.WAITING,
            center_lat=lat,
            center_lon=lon,
        )

    monkeypatch.setattr(GameService, "create_game", create_game)
    response = client.post("/api/games", json=payload)
    
    assert response.status_code == 200, response.text
    
    data = response.json()
    assert data["id"] == "game-id"
    assert "_id" not in data
    assert data["creator_id"] == "creator-id"
    assert data["status"] == "WAITING"
    assert data["join_code"] == "JOIN1"


def prepare_start_game(monkeypatch, game, players):
    async def get_by_id(game_id):
        assert game_id == game.id
        return game

    async def get_by_game(game_id):
        assert game_id == game.id
        return players

    async def update(updated_game):
        assert updated_game.id == game.id
        return True

    async def activate(updated_game, expected_targets):
        assert updated_game.id == game.id
        assert expected_targets == []
        return True

    async def update_player(updated_player):
        return True

    async def resolve_context(lat, lon):
        return LocationContext(
            locality="Test City",
            environment_type="urban",
            nearby_categories=["parks_and_recreation"],
            context_mode="contextual",
            context_status="available",
        )

    async def generate_targets(
        game_id,
        context,
        count,
        supported_classes=None,
        excluded_classes=None,
    ):
        assert game_id == game.id
        assert context.locality == "Test City"
        assert count == 5
        assert set(supported_classes) >= {
            "bench",
            "bicycle",
            "bird",
            "car",
            "backpack",
        }
        return [
            Target(id=f"target-{index}", name=name, class_name=class_name, hint="Nearby.")
            for index, (name, class_name) in enumerate(
                (
                    ("Bench", "bench"),
                    ("Bicycle", "bicycle"),
                    ("Bird", "bird"),
                    ("Car", "car"),
                    ("Backpack", "backpack"),
                ),
                start=1,
            )
        ]

    monkeypatch.setattr(GameRepository, "get_by_id", get_by_id)
    monkeypatch.setattr(PlayerRepository, "get_by_game", get_by_game)
    monkeypatch.setattr(GameRepository, "update", update)
    monkeypatch.setattr(GameRepository, "activate_with_targets", activate)
    async def no_recent_targets(game_id, context, *, limit=3):
        return []

    monkeypatch.setattr(
        GameRepository,
        "get_recent_target_sets_for_context",
        no_recent_targets,
    )
    monkeypatch.setattr(PlayerRepository, "update", update_player)
    monkeypatch.setattr(location_context_service, "resolve", resolve_context)
    monkeypatch.setattr(target_generator, "generate_targets", generate_targets)
    monkeypatch.setattr(
        image_verifier,
        "get_supported_target_classes",
        lambda: ("bench", "bicycle", "bird", "car", "backpack"),
    )
    monkeypatch.setattr(image_verifier, "class_id_for_name", lambda _: None)


def make_game(status=GameState.WAITING):
    return Game(
        _id="game-id",
        join_code="JOIN1",
        creator_id="host-id",
        status=status,
        center_lat=10.0,
        center_lon=20.0,
    )


def make_players():
    return [
        Player(_id="host-id", game_id="game-id", display_name="Host"),
        Player(_id="guest-id", game_id="game-id", display_name="Guest"),
    ]


def test_start_game_with_two_players(monkeypatch, caplog):
    game = make_game()
    prepare_start_game(monkeypatch, game, make_players())

    with caplog.at_level("INFO"):
        response = client.post("/api/games/game-id/start?creator_id=host-id")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "message": "Game started",
        "context_status": "available",
        "context_mode": "contextual",
        "environment_type": "urban",
        "nearby_categories": ["parks_and_recreation"],
        "context_provider": None,
        "context_notice": None,
    }
    assert game.status == GameState.ACTIVE
    assert game.targets[0].class_name == "bench"
    assert game.start_time is not None
    assert "locality=Test City" in caplog.text
    assert "environment=urban" in caplog.text
    assert "'bench'" in caplog.text
    assert "source=Gemma" in caplog.text
    assert "context_mode=contextual" in caplog.text
    assert "candidate_classes=" in caplog.text
    assert "10.0" not in caplog.text
    assert "20.0" not in caplog.text


@pytest.mark.asyncio
async def test_start_game_reuses_cached_location_context(monkeypatch):
    game = make_game()
    game.location_context = LocationContext(
        locality="Test City",
        environment_type="urban",
        context_status="available",
    )
    prepare_start_game(monkeypatch, game, make_players())

    async def should_not_resolve_again(lat, lon):
        raise AssertionError("A cached game context must be reused")

    monkeypatch.setattr(location_context_service, "resolve", should_not_resolve_again)

    context = await GameService.start_game(game.id, game.creator_id)

    assert context.locality == "Test City"
    assert game.status == GameState.ACTIVE


@pytest.mark.asyncio
async def test_start_uses_latest_host_location_for_context(monkeypatch):
    game = make_game()
    players = make_players()
    players[0].latest_lat = 43.654
    players[0].latest_lon = -79.382
    players[0].location_accuracy_m = 4
    players[0].last_location_update = datetime.now(timezone.utc)
    prepare_start_game(monkeypatch, game, players)
    resolved_coordinates = []

    async def resolve_context(lat, lon):
        resolved_coordinates.append((lat, lon))
        return LocationContext(
            locality="Test City",
            environment_type="urban",
            context_status="available",
        )

    monkeypatch.setattr(location_context_service, "resolve", resolve_context)

    await GameService.start_game(game.id, game.creator_id)

    assert resolved_coordinates == [(43.654, -79.382)]


def test_start_game_rejects_non_host(monkeypatch):
    game = make_game()
    prepare_start_game(monkeypatch, game, make_players())

    response = client.post("/api/games/game-id/start?creator_id=guest-id")

    assert response.status_code == 400
    assert response.json()["detail"] == "Only the game host can start the game."
    assert game.status == GameState.WAITING


def test_host_can_start_game_alone_and_real_target_generator_is_used(monkeypatch):
    game = make_game()
    host = Player(_id="host-id", game_id="game-id", display_name="Host")
    prepare_start_game(monkeypatch, game, [host])
    generated = []

    async def generate_targets(
        game_id,
        context,
        count,
        supported_classes=None,
        excluded_classes=None,
    ):
        generated.append((game_id, count, set(supported_classes)))
        return [
            Target(id=f"target-{index}", name=name, class_name=class_name, hint="Nearby.")
            for index, (name, class_name) in enumerate(
                (
                    ("Bench", "bench"),
                    ("Bicycle", "bicycle"),
                    ("Bird", "bird"),
                    ("Car", "car"),
                    ("Backpack", "backpack"),
                ),
                start=1,
            )
        ]

    monkeypatch.setattr(target_generator, "generate_targets", generate_targets)
    response = client.post("/api/games/game-id/start?creator_id=host-id")

    assert response.status_code == 200, response.text
    assert game.status == GameState.ACTIVE
    assert len(game.targets) == 5
    assert generated == [
        ("game-id", 5, {"bench", "bicycle", "bird", "car", "backpack"})
    ]


@pytest.mark.asyncio
async def test_start_excludes_repeated_classes_from_same_location_context(monkeypatch):
    game = make_game()
    host = Player(_id="host-id", game_id="game-id", display_name="Host")
    prepare_start_game(monkeypatch, game, [host])
    recent_sets = [
        ["car", "bicycle", "bird", "dog", "traffic light"],
        ["car", "bicycle", "bird", "dog", "bus"],
        ["car", "bicycle", "bird", "dog", "backpack"],
    ]
    received = {}

    async def get_recent_sets(game_id, context, *, limit=3):
        assert game_id == game.id
        assert context.context_mode == "contextual"
        assert limit == 3
        return recent_sets

    async def generate_targets(
        game_id,
        context,
        count,
        supported_classes=None,
        excluded_classes=None,
    ):
        received["excluded_classes"] = excluded_classes
        return [
            Target(
                id=f"target-{index}",
                name=class_name.title(),
                class_name=class_name,
                hint="Suitable for this area.",
            )
            for index, class_name in enumerate(
                ("bench", "frisbee", "sports ball", "potted plant", "kite")
            )
        ]

    monkeypatch.setattr(
        GameRepository,
        "get_recent_target_sets_for_context",
        get_recent_sets,
    )
    monkeypatch.setattr(
        image_verifier,
        "get_supported_target_classes",
        lambda: OUTDOOR_CANDIDATE_CLASSES,
    )
    monkeypatch.setattr(target_generator, "generate_targets", generate_targets)

    await GameService.start_game(game.id, game.creator_id)

    assert received["excluded_classes"] == {"car", "bicycle", "bird", "dog"}
    assert {target.class_name for target in game.targets}.isdisjoint(
        received["excluded_classes"]
    )


def test_repeated_classes_from_matching_context_are_avoided_when_alternatives_exist():
    ranked = [
        TargetCandidate(class_name="car", score=53),
        TargetCandidate(class_name="bicycle", score=47),
        TargetCandidate(class_name="bird", score=45),
        TargetCandidate(class_name="dog", score=33),
        TargetCandidate(class_name="traffic light", score=26),
        TargetCandidate(class_name="backpack", score=25),
        TargetCandidate(class_name="potted plant", score=24),
        TargetCandidate(class_name="stop sign", score=21),
        TargetCandidate(class_name="bus", score=20),
        TargetCandidate(class_name="bottle", score=18),
        TargetCandidate(class_name="bench", score=15),
        TargetCandidate(class_name="truck", score=12),
        TargetCandidate(class_name="umbrella", score=9),
        TargetCandidate(class_name="kite", score=9),
    ]
    history = [
        ["Car", "Bicycle", "Bird", "Dog", "Traffic Light"],
        ["car", "bicycle", "bird", "dog", "bus"],
        ["car", "bicycle", "bird", "dog", "backpack"],
    ]

    avoided = _repeated_context_classes_to_avoid(
        history,
        ranked,
        target_count=5,
    )

    assert avoided == {"car", "bicycle", "bird", "dog"}
    assert len({candidate.class_name for candidate in ranked} - avoided) >= 5


def test_context_repetition_filter_relaxes_exclusions_if_too_few_candidates_remain():
    ranked = [
        TargetCandidate(class_name=f"class-{index}", score=10 - index)
        for index in range(7)
    ]
    history = [
        [candidate.class_name for candidate in ranked[:6]],
        [candidate.class_name for candidate in ranked[:6]],
    ]

    avoided = _repeated_context_classes_to_avoid(
        history,
        ranked,
        target_count=5,
    )

    assert len(set(candidate.class_name for candidate in ranked) - avoided) == 5
    assert "class-0" not in avoided


def test_context_repetition_filter_keeps_classes_when_positive_context_is_too_small():
    ranked = [
        TargetCandidate(class_name=f"class-{index}", score=10 if index < 4 else 0)
        for index in range(7)
    ]
    history = [
        [candidate.class_name for candidate in ranked[:4]],
        [candidate.class_name for candidate in ranked[:4]],
    ]

    avoided = _repeated_context_classes_to_avoid(
        history,
        ranked,
        target_count=5,
    )

    assert avoided == set()


@pytest.mark.asyncio
async def test_recent_target_sets_are_limited_to_the_same_coarse_context(monkeypatch):
    context = LocationContext(
        locality="Test City",
        region="Test Region",
        environment_type="urban",
        nearby_categories=["roads_and_junctions"],
        nearby_feature_counts={"roads_and_junctions": 3},
        context_mode="contextual",
        context_provider="openstreetmap",
        context_status="available",
    )
    matching = {
        "_id": "prior-game",
        "targets": [{"class_name": "car"}] * 5,
        "location_context": {"nearby_feature_counts": {"roads_and_junctions": 3}},
    }
    other_area = {
        "_id": "other-area",
        "targets": [{"class_name": "bird"}] * 5,
        "location_context": {"nearby_feature_counts": {"roads_and_junctions": 4}},
    }
    operations = []

    class Cursor:
        def __init__(self, documents):
            self.documents = documents

        def sort(self, field, direction):
            assert (field, direction) == ("created_at", -1)
            return self

        def limit(self, count):
            assert count == 30
            return self

        def __aiter__(self):
            self.iterator = iter(self.documents)
            return self

        async def __anext__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                raise StopAsyncIteration

    class GamesCollection:
        def find(self, query, projection):
            operations.append((query, projection))
            return Cursor([other_area, matching])

    monkeypatch.setattr(
        "app.db.repositories.repositories.get_db",
        lambda: SimpleNamespace(games=GamesCollection()),
    )

    result = await GameRepository.get_recent_target_sets_for_context(
        "new-game",
        context,
        limit=3,
    )

    assert result == [["car"] * 5]
    query, projection = operations[0]
    assert query["_id"] == {"$ne": "new-game"}
    assert query["status"] == {"$in": ["ACTIVE", "FINISHED"]}
    assert query["location_context.nearby_categories"] == ["roads_and_junctions"]
    assert projection == {
        "targets.class_name": 1,
        "location_context.nearby_feature_counts": 1,
    }


def test_solo_leaderboard_contains_movement_but_not_coordinates(monkeypatch):
    host = Player(
        _id="host-id",
        game_id="game-id",
        display_name="Solo",
        latest_lat=10.0,
        latest_lon=20.0,
        movement_meters=12.5,
    )

    async def leaderboard(game_id):
        assert game_id == "game-id"
        return [host]

    monkeypatch.setattr(GameService, "get_leaderboard", leaderboard)
    response = client.get("/api/games/game-id/leaderboard")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["movement_meters"] == 12.5
    assert "movement_required_meters" not in response.json()[0]
    assert "latest_lat" not in response.json()[0]
    assert "latest_lon" not in response.json()[0]


def test_location_api_passes_timestamp_and_accuracy_to_server_tracker(monkeypatch):
    received = []

    async def update_location(game_id, player_id, lat, lon, accuracy_m, timestamp):
        received.append((game_id, player_id, lat, lon, accuracy_m, timestamp))
        return LocationUpdateResult(True, "Location updated.", 18.5, 8.5)

    monkeypatch.setattr(LocationService, "update_location", update_location)
    async def no_area_refresh(game_id, lat, lon):
        return False, None

    async def no_game(game_id):
        return None

    monkeypatch.setattr(GameService, "refresh_area_context", no_area_refresh)
    monkeypatch.setattr(GameRepository, "get_by_id", no_game)
    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/api/games/game-id/players/player-id/location",
        json={
            "lat": 43.0,
            "lon": -79.0,
            "accuracy": 4.0,
            "timestamp": timestamp,
        },
    )

    assert response.status_code == 200, response.text
    assert received[0][:5] == ("game-id", "player-id", 43.0, -79.0, 4.0)
    assert received[0][5].isoformat() == timestamp
    assert response.json()["movement_meters"] == 18.5


def test_location_api_accepts_full_geolocation_names_and_accuracy_m(monkeypatch):
    received = []

    async def update_location(game_id, player_id, lat, lon, accuracy_m, timestamp):
        received.append((lat, lon, accuracy_m, timestamp))
        return LocationUpdateResult(True, "Location updated.", 0.0)

    monkeypatch.setattr(LocationService, "update_location", update_location)
    async def no_area_refresh(game_id, lat, lon):
        return False, None

    async def no_game(game_id):
        return None

    monkeypatch.setattr(GameService, "refresh_area_context", no_area_refresh)
    monkeypatch.setattr(GameRepository, "get_by_id", no_game)
    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/api/games/game-id/players/player-id/location",
        json={
            "latitude": 43.0,
            "longitude": -79.0,
            "accuracy_m": 4.0,
            "timestamp": timestamp,
        },
    )

    assert response.status_code == 200, response.text
    assert received[0][:3] == (43.0, -79.0, 4.0)
    assert received[0][3].isoformat() == timestamp


def test_location_api_labels_213_meter_fix_as_approximate(monkeypatch):
    async def update_location(game_id, player_id, lat, lon, accuracy_m, timestamp):
        return LocationUpdateResult(
            True,
            "Approximate location accepted; movement awaits stable readings.",
            0.0,
        )

    async def no_area_refresh(game_id, lat, lon):
        return False, None

    async def no_game(game_id):
        return Game(
            _id=game_id,
            join_code="JOIN1",
            creator_id="host",
            location_context=LocationContext(
                locality="Test District",
                context_mode="contextual",
                context_status="available",
            ),
        )

    monkeypatch.setattr(LocationService, "update_location", update_location)
    monkeypatch.setattr(GameService, "refresh_area_context", no_area_refresh)
    monkeypatch.setattr(GameRepository, "get_by_id", no_game)
    response = client.post(
        "/api/games/game-id/players/player-id/location",
        json={
            "lat": 43.0,
            "lon": -79.0,
            "accuracy_m": 213,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert response.json()["location_accuracy_m"] == 213
    assert response.json()["location_quality"] == "approximate"
    assert response.json()["location_context"]["locality"] == "Test District"


def test_location_api_logs_validation_details_without_coordinate_values(caplog):
    response = client.post(
        "/api/games/game-id/players/player-id/location",
        json={
            "lat": 91.0,
            "lon": -79.0,
            "accuracy_m": 4.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "lat"]
    assert response.json()["detail"][0]["type"] == "less_than_equal"
    assert "Request validation failed" in caplog.text
    assert "lat" in caplog.text
    assert "91.0" not in caplog.text


def test_location_api_returns_422_for_non_positive_accuracy():
    response = client.post(
        "/api/games/game-id/players/player-id/location",
        json={
            "lat": 43.0,
            "lon": -79.0,
            "accuracy_m": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "accuracy_m"]
    assert response.json()["detail"][0]["type"] == "greater_than"


def test_player_heartbeat_marks_a_current_voter_active(monkeypatch):
    received = []

    async def touch(game_id, player_id):
        received.append((game_id, player_id))
        return True

    monkeypatch.setattr(PlayerRepository, "touch_last_seen", touch)
    response = client.post(
        "/api/games/game-id/players/player-id/heartbeat"
    )

    assert response.status_code == 200
    assert response.json() == {"active": True}
    assert received == [("game-id", "player-id")]


def test_target_change_api_returns_persisted_vote_state(monkeypatch):
    game = make_game(GameState.ACTIVE)
    game.targets = [
        Target(
            id="target-id",
            name="Backpack",
            class_name="backpack",
            hint="Find one nearby.",
        )
    ]
    game.target_change_vote = TargetChangeVote(
        proposal_id="proposal-id",
        target_id="target-id",
        target_name="Backpack",
        requested_by="player-id",
        eligible_player_ids=["player-id", "teammate-id"],
        approved_player_ids=["player-id"],
        status="pending",
    )

    async def request_vote(game_id, player_id, target_id):
        assert (game_id, player_id, target_id) == (
            "game-id",
            "player-id",
            "target-id",
        )
        return game

    monkeypatch.setattr(TargetChangeService, "request", request_vote)
    response = client.post(
        "/api/games/game-id/targets/target-id/replacement-vote",
        json={"player_id": "player-id"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["target_change_vote"]["proposal_id"] == "proposal-id"
    assert response.json()["target_change_vote"]["eligible_player_ids"] == [
        "player-id",
        "teammate-id",
    ]
    assert response.json()["target_change_vote"]["approved_player_ids"] == [
        "player-id"
    ]


@pytest.mark.asyncio
async def test_start_requires_atomic_target_persistence(monkeypatch):
    game = make_game()
    players = make_players()
    prepare_start_game(monkeypatch, game, players)

    async def failed_activation(_game, _expected_targets):
        return False

    monkeypatch.setattr(GameRepository, "activate_with_targets", failed_activation)

    with pytest.raises(GameStartError, match="game changed while targets were being prepared"):
        await GameService.start_game(game.id, game.creator_id)

    assert game.status == GameState.ACTIVE
    assert len(game.targets) == 5


def test_start_game_rejects_already_active_game(monkeypatch):
    game = make_game(status=GameState.ACTIVE)
    prepare_start_game(monkeypatch, game, make_players())

    response = client.post("/api/games/game-id/start?creator_id=host-id")

    assert response.status_code == 400
    assert response.json()["detail"] == "Game must be waiting to start."
    assert game.status == GameState.ACTIVE
