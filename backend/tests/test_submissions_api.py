import logging
from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from app.config import settings
from app.main import app
from app.models.submission import SubmissionResult
from app.services.verification_service import VerificationService

client = TestClient(app)


def image_bytes(image_format="PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="green").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "expected_suffix", "content_type"),
    [
        ("JPEG", ".jpg", "image/jpeg"),
        ("PNG", ".png", "image/png"),
        ("WEBP", ".webp", "image/webp"),
    ],
)
def test_submission_validates_image_and_deletes_random_temp_file(
    monkeypatch, caplog, image_format, expected_suffix, content_type
):
    caplog.set_level(logging.INFO)
    observed_paths = []

    async def process_submission(game_id, player_id, target_id, file_path):
        path = Path(file_path)
        assert path.exists()
        assert "user-supplied-name" not in path.name
        assert path.suffix == expected_suffix
        with Image.open(path) as uploaded_image:
            assert uploaded_image.format == image_format
        observed_paths.append(path)
        return SubmissionResult(
            success=False,
            target="bicycle",
            verified=False,
            message="Bicycle was not detected in the image.",
            total_targets=5,
        )

    monkeypatch.setattr(VerificationService, "process_submission", process_submission)
    response = client.post(
        "/api/games/game-id/submit",
        data={"player_id": "player-id", "target_id": "target-id"},
        files={
            "image": (
                f"user-supplied-name{expected_suffix}",
                image_bytes(image_format),
                content_type,
            )
        },
    )

    assert response.status_code == 200, response.text
    assert (
        f"format={image_format} width=8 height=8"
        in caplog.text
    )
    assert len(observed_paths) == 1
    assert not observed_paths[0].exists()


def test_submission_rejects_non_image_content(monkeypatch):
    async def should_not_verify(*args, **kwargs):
        raise AssertionError("Invalid image content must not reach verification")

    monkeypatch.setattr(VerificationService, "process_submission", should_not_verify)
    response = client.post(
        "/api/games/game-id/submit",
        data={"player_id": "player-id", "target_id": "target-id"},
        files={"image": ("fake.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image"


def test_submission_enforces_streamed_upload_size(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    async def should_not_verify(*args, **kwargs):
        raise AssertionError("Oversized image must not reach verification")

    monkeypatch.setattr(VerificationService, "process_submission", should_not_verify)
    response = client.post(
        "/api/games/game-id/submit",
        data={"player_id": "player-id", "target_id": "target-id"},
        files={"image": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"
