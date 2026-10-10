from datetime import datetime
import sys
import pytest
from types import ModuleType, SimpleNamespace
from app.ai.yolo_verifier import Detection, YOLOImageVerifier
from app.config import settings
from app.db.repositories.repositories import (
    GameRepository,
    PlayerRepository,
    SubmissionRepository,
)
from app.models.game import Game, GameState
from app.models.player import Player
from app.models.target import Target
from app.models.target import YOLO_SUPPORTED_CLASSES
from app.services.location_service import LocationService
from app.services.verification_service import VerificationService
from app.ai.yolo_verifier import VerificationUnavailable

TARGET_CLASSES = ("bicycle", "bench", "bird", "car", "backpack")


def make_game_and_player():
    targets = [
        Target(id=f"target-{index}", name=class_name.title(), class_name=class_name, hint="Nearby.")
        for index, class_name in enumerate(TARGET_CLASSES, start=1)
    ]
    game = Game(
        _id="game-id",
        join_code="JOIN1",
        creator_id="host-id",
        status=GameState.ACTIVE,
        targets=targets,
    )
    player = Player(
        _id="player-id",
        game_id="game-id",
        display_name="Player",
        last_location_update=datetime.utcnow(),
    )
    return game, player


def prepare_service(monkeypatch, game, player, detections, *, mock_location=True):
    submissions = []

    async def get_game(game_id):
        return game if game_id == game.id else None

    async def get_player(player_id):
        return player if player_id == player.id else None

    async def is_valid_location(player_arg):
        return True, "Location valid", 12.0

    async def detect_image(image_path, target_class):
        return detections[target_class]

    async def save_submission(submission):
        submissions.append(submission)
        return submission

    async def complete_target_once(player_id, target_id):
        if target_id in player.completed_targets:
            return None
        player.completed_targets.append(target_id)
        player.score += 1
        return player

    async def update_game(updated_game):
        return updated_game.id == game.id

    monkeypatch.setattr(GameRepository, "get_by_id", get_game)
    monkeypatch.setattr(PlayerRepository, "get_by_id", get_player)
    monkeypatch.setattr(PlayerRepository, "complete_target_once", complete_target_once)
    monkeypatch.setattr(GameRepository, "update", update_game)
    monkeypatch.setattr(SubmissionRepository, "create", save_submission)
    if mock_location:
        monkeypatch.setattr(LocationService, "validate_player_location", is_valid_location)
    monkeypatch.setattr("app.services.verification_service.image_verifier.detect_image", detect_image)
    return submissions


def detections_for(class_name, confidence):
    return {
        target_class: Detection(
            class_name=class_name if target_class == "bicycle" else None,
            confidence=confidence if target_class == "bicycle" else 0.0,
        )
        for target_class in TARGET_CLASSES
    }


@pytest.mark.asyncio
async def test_correct_target_class_above_threshold_is_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for("bicycle", 0.91))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.success is True
    assert result.verified is True
    assert result.confidence == 0.91
    assert result.completed_targets == 1
    assert result.total_targets == 5
    assert result.progress_percent == 20
    assert player.score == 1


@pytest.mark.asyncio
async def test_unrelated_detected_class_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for("sheep", 0.99))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.success is False
    assert result.verified is False
    assert result.message == "YOLO detected sheep, not Bicycle. Center the target and try again."
    assert result.completed_targets == 0
    assert result.progress_percent == 0
    assert player.score == 0


@pytest.mark.asyncio
async def test_no_detections_have_a_distinct_rejection_message(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for(None, 0.0))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.verified is False
    assert result.confidence == 0.0
    assert result.message == "No supported objects were detected. Try a clearer, closer photo."
    assert player.score == 0


@pytest.mark.asyncio
async def test_detection_at_exact_configured_threshold_is_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "YOLO_CONFIDENCE_THRESHOLD", 0.60)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for("bicycle", 0.60))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.verified is True
    assert result.confidence == 0.60


@pytest.mark.asyncio
async def test_matching_detection_below_threshold_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for("bicycle", 0.59))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.success is False
    assert result.verified is False
    assert result.confidence == 0.59
    assert "60%" in result.message
    assert result.progress_percent == 0
    assert player.score == 0


@pytest.mark.asyncio
async def test_duplicate_submission_does_not_increase_score(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    submissions = prepare_service(monkeypatch, game, player, detections_for("bicycle", 0.91))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")
    target_id = game.targets[0].id

    first = await VerificationService.process_submission(game.id, player.id, target_id, str(image))
    second = await VerificationService.process_submission(game.id, player.id, target_id, str(image))

    assert first.success is True
    assert second.success is False
    assert second.message == "Target already completed."
    assert player.completed_targets == [target_id]
    assert player.score == 1
    assert len(submissions) == 1


@pytest.mark.asyncio
async def test_five_verified_targets_produce_exactly_100_percent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    detections = {
        target_class: Detection(class_name=target_class, confidence=0.90)
        for target_class in TARGET_CLASSES
    }
    prepare_service(monkeypatch, game, player, detections)
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    results = [
        await VerificationService.process_submission(
            game.id, player.id, target.id, str(image)
        )
        for target in game.targets
    ]

    assert [result.progress_percent for result in results] == [20, 40, 60, 80, 100]
    assert results[-1].completed_targets == 5
    assert results[-1].total_targets == 5
    assert game.status == GameState.FINISHED
    assert player.score == 5


@pytest.mark.asyncio
async def test_all_verified_targets_win_without_movement(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(
        monkeypatch,
        game,
        player,
        {
            target.class_name: Detection(target.class_name, 0.9)
            for target in game.targets
        },
    )
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    results = [
        await VerificationService.process_submission(
            game.id, player.id, target.id, str(image)
        )
        for target in game.targets
    ]

    assert results[-1].success is True
    assert "won the game" in results[-1].message
    assert game.status == GameState.FINISHED
    assert player.score == 5


@pytest.mark.asyncio
async def test_valid_image_submission_far_from_start_is_not_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    game.center_lat = 43.0
    game.center_lon = -79.0
    player.latest_lat = 43.01
    player.latest_lon = -79.0
    player.location_accuracy_m = 5
    player.last_location_update = datetime.utcnow()
    prepare_service(
        monkeypatch,
        game,
        player,
        detections_for("bicycle", 0.91),
        mock_location=False,
    )
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.verified is True
    assert result.success is True
    assert player.score == 1


@pytest.mark.asyncio
async def test_real_mode_uses_detector_instead_of_demo_success(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    called = []

    class FakeYOLOModel:
        def __init__(self, model_name):
            assert model_name == settings.YOLO_MODEL
            self.names = dict(enumerate(YOLO_SUPPORTED_CLASSES))

        def __call__(self, image_path, verbose=False):
            called.append(image_path)
            box = type("Box", (), {"cls": [1], "conf": [0.42]})()
            result = type("Result", (), {"boxes": [box], "names": self.names})()
            return [result]

    fake_ultralytics = ModuleType("ultralytics")
    setattr(fake_ultralytics, "YOLO", FakeYOLOModel)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    verifier = YOLOImageVerifier()

    detection = await verifier.detect_image(str(tmp_path / "fresh.jpg"), "bicycle")

    assert called == [str(tmp_path / "fresh.jpg")]
    assert detection == Detection(
        class_name="bicycle",
        confidence=0.42,
        top_detections=((1, "bicycle", 0.42),),
        class_id=1,
    )


def test_coco_class_ids_are_taken_from_the_loaded_name_table(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    class FakeYOLOModel:
        names = dict(enumerate(YOLO_SUPPORTED_CLASSES))

        def __init__(self, _model_name):
            pass

    fake_ultralytics = ModuleType("ultralytics")
    setattr(fake_ultralytics, "YOLO", FakeYOLOModel)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    verifier = YOLOImageVerifier()

    assert verifier.get_supported_target_classes() == tuple(
        class_name for class_name in YOLO_SUPPORTED_CLASSES
        if class_name in (
            "bicycle", "car", "bus", "truck", "traffic light", "stop sign",
            "bench", "bird", "cat", "dog", "backpack", "umbrella", "frisbee",
            "sports ball", "kite", "skateboard", "bottle", "potted plant",
        )
    )
    assert verifier.class_id_for_name("bicycle") == 1
    assert verifier.class_id_for_name("sheep") == 18
    assert verifier.class_id_for_name("banana") == 46
    assert verifier.class_id_for_name("sandwich") == 48
    assert verifier.class_id_for_name("potted plant") == 58
    for unsupported_name in ("tree", "rock", "leaf", "flower"):
        assert verifier.class_id_for_name(unsupported_name) is None


def test_target_model_rejects_classes_absent_from_coco_vocabulary():
    for unsupported_name in ("tree", "rock", "leaf", "flower"):
        with pytest.raises(ValueError, match="not supported"):
            Target(
                id=unsupported_name,
                name=unsupported_name.title(),
                class_name=unsupported_name,
                hint="Nearby.",
            )


def test_existing_target_titles_are_canonicalized_from_the_stored_class():
    target = Target(
        id="legacy-title",
        name="Oak tree",
        class_name="potted plant",
        hint="Nearby.",
    )

    assert target.name == "Potted Plant"
    assert target.class_name == "potted plant"


def test_loaded_weights_limit_available_targets_to_actual_class_names(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    class RestrictedYOLOModel:
        names = {0: "bird", 1: "bicycle", 2: "tree"}

        def __init__(self, _model_name):
            pass

    fake_ultralytics = ModuleType("ultralytics")
    setattr(fake_ultralytics, "YOLO", RestrictedYOLOModel)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    verifier = YOLOImageVerifier()

    assert verifier.get_supported_target_classes() == ("bicycle", "bird")
    assert verifier.class_id_for_name("tree") is None
    assert verifier.class_id_for_name("bicycle") == 1


@pytest.mark.asyncio
async def test_missing_yolo_weights_are_reported_without_exposing_exception(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    class MissingWeights:
        def __init__(self, _model_name):
            raise FileNotFoundError("private local weights path")

    fake_ultralytics = ModuleType("ultralytics")
    setattr(fake_ultralytics, "YOLO", MissingWeights)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    verifier = YOLOImageVerifier()

    with pytest.raises(VerificationUnavailable) as exc_info:
        await verifier.detect_image("capture.jpg", "bicycle")

    assert str(exc_info.value) == (
        "The configured YOLO model weights are unavailable or unsupported."
    )
    assert "private local weights path" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_yolo_inference_failure_is_distinct_from_missing_weights(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    class FailingInferenceModel:
        names = dict(enumerate(YOLO_SUPPORTED_CLASSES))

        def __init__(self, _model_name):
            pass

        def __call__(self, _image_path, verbose=False):
            raise RuntimeError("private inference detail")

    fake_ultralytics = ModuleType("ultralytics")
    setattr(fake_ultralytics, "YOLO", FailingInferenceModel)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    verifier = YOLOImageVerifier()

    with pytest.raises(VerificationUnavailable) as exc_info:
        await verifier.detect_image("capture.jpg", "bicycle")

    assert str(exc_info.value) == "YOLO inference failed. Please try again."
    assert "private inference detail" not in str(exc_info.value)


def test_target_mapping_uses_canonical_coco_names_and_rejects_aliases():
    assert Target(
        id="phone",
        name="Phone",
        class_name=" CELL PHONE ",
        hint="Find a phone.",
    ).class_name == "cell phone"
    with pytest.raises(ValueError, match="not supported"):
        Target(
            id="phone-alias",
            name="Phone",
            class_name="phone",
            hint="Find a phone.",
        )


@pytest.mark.asyncio
async def test_demo_mode_is_explicitly_simulated(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    verifier = YOLOImageVerifier()

    detection = await verifier.detect_image(str(tmp_path / "demo.jpg"), "bicycle")

    assert detection == Detection(class_name="bicycle", confidence=0.95)


@pytest.mark.asyncio
async def test_zero_verified_targets_report_zero_percent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    game, player = make_game_and_player()
    prepare_service(monkeypatch, game, player, detections_for("tree", 0.99))
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"test image content")

    result = await VerificationService.process_submission(
        game.id, player.id, game.targets[0].id, str(image)
    )

    assert result.completed_targets == 0
    assert result.total_targets == 5
    assert result.progress_percent == 0


@pytest.mark.asyncio
async def test_repository_claims_completion_with_atomic_mongo_update(monkeypatch):
    captured = {}
    stored_player = Player(
        _id="player-id",
        game_id="game-id",
        display_name="Player",
        completed_targets=["target-id"],
        score=1,
    )

    class FakePlayersCollection:
        async def update_one(self, query, update):
            captured["query"] = query
            captured["update"] = update
            return SimpleNamespace(modified_count=1)

    async def get_by_id(player_id):
        assert player_id == stored_player.id
        return stored_player

    monkeypatch.setattr(
        "app.db.repositories.repositories.get_db",
        lambda: SimpleNamespace(players=FakePlayersCollection()),
    )
    monkeypatch.setattr(PlayerRepository, "get_by_id", get_by_id)

    result = await PlayerRepository.complete_target_once("player-id", "target-id")

    assert result is stored_player
    assert captured["query"] == {
        "_id": "player-id",
        "completed_targets": {"$ne": "target-id"},
    }
    assert captured["update"] == {
        "$addToSet": {"completed_targets": "target-id"},
        "$inc": {"score": 1},
    }
