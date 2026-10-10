import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

from app.ai.yolo_verifier import YOLOImageVerifier
from app.config import settings


@pytest.mark.skipif(
    os.getenv("RUN_YOLO_ONNX_SMOKE_TEST") != "1",
    reason="Set RUN_YOLO_ONNX_SMOKE_TEST=1 to run actual ONNX inference.",
)
def test_exported_yolov8n_runs_real_onnx_inference(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    model_path = Path(
        os.getenv("YOLO_ONNX_MODEL_PATH", str(project_root / "yolov8n.onnx"))
    )
    ultralytics_spec = importlib.util.find_spec("ultralytics")
    if ultralytics_spec is None or ultralytics_spec.origin is None:
        pytest.fail("Ultralytics is required to locate its bundled real test image.")
    image_path = Path(ultralytics_spec.origin).parent / "assets" / "bus.jpg"

    assert model_path.is_file(), f"ONNX model does not exist: {model_path}"
    assert image_path.is_file(), f"Bundled test image does not exist: {image_path}"
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "YOLO_MODEL", str(model_path))

    verifier = YOLOImageVerifier()
    supported = verifier.get_supported_target_classes()
    detection = asyncio.run(verifier.detect_image(str(image_path), "bus"))

    assert "bus" in supported
    assert verifier.class_id_for_name("bus") == 5
    assert detection.class_name == "bus"
    assert detection.class_id == 5
    assert detection.confidence >= settings.YOLO_CONFIDENCE_THRESHOLD
