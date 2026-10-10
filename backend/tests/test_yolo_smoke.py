import asyncio
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image
from app.config import settings
from app.ai.yolo_verifier import YOLOImageVerifier
from app.models.target import OUTDOOR_CANDIDATE_CLASSES, YOLO_SUPPORTED_CLASSES


@pytest.mark.skipif(
    os.getenv("RUN_YOLO_SMOKE_TEST") != "1",
    reason="Set RUN_YOLO_SMOKE_TEST=1 to load YOLO weights and run real inference.",
)
def test_configured_yolo_model_runs_inference_without_changing_game_scores(tmp_path):
    from ultralytics.utils import ASSETS

    image_path = Path(ASSETS) / "bus.jpg"
    assert image_path.is_file()
    uploaded_image_path = tmp_path / "capture.jpg"
    shutil.copyfile(image_path, uploaded_image_path)
    with Image.open(uploaded_image_path) as uploaded_image:
        uploaded_image.verify()
    import cv2

    assert cv2.imread(str(uploaded_image_path)) is not None

    verifier = YOLOImageVerifier()
    matching_detection = asyncio.run(
        verifier.detect_image(str(uploaded_image_path), "bus")
    )
    wrong_class_detection = asyncio.run(
        verifier.detect_image(str(uploaded_image_path), "bicycle")
    )

    assert set(YOLO_SUPPORTED_CLASSES).issubset(set(verifier.model_class_names.values()))
    assert set(OUTDOOR_CANDIDATE_CLASSES).issubset(
        set(verifier.model_class_names.values())
    )
    assert verifier.class_id_for_name("bicycle") == 1
    assert verifier.class_id_for_name("sheep") == 18
    assert verifier.class_id_for_name("banana") == 46
    assert verifier.class_id_for_name("sandwich") == 48
    assert verifier.class_id_for_name("potted plant") == 58
    assert "tree" not in verifier.model_class_names.values()
    assert "rock" not in verifier.model_class_names.values()
    assert matching_detection.class_name == "bus"
    assert matching_detection.confidence >= settings.YOLO_CONFIDENCE_THRESHOLD
    assert wrong_class_detection.class_name == "bus"
    assert wrong_class_detection.confidence == matching_detection.confidence
