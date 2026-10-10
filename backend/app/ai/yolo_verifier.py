import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models.target import (
    OUTDOOR_CANDIDATE_CLASSES,
    YOLO_CLASS_NAME_BY_CASEFOLD,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    class_name: Optional[str]
    confidence: float
    top_detections: tuple[tuple[int, str, float], ...] = ()
    class_id: Optional[int] = None


class VerificationUnavailable(Exception):
    pass


class YOLOImageVerifier:
    def __init__(self):
        self.model_name = settings.YOLO_MODEL
        self.model: Any = None
        self.model_class_names: dict[int, str] = {}
        self.is_onnx_model = Path(self.model_name).suffix.casefold() == ".onnx"

    def _load_model(self):
        if self.model is None and not settings.DEMO_MODE:
            if self.is_onnx_model:
                import onnxruntime as ort

                model = ort.InferenceSession(
                    self.model_name,
                    providers=["CPUExecutionProvider"],
                )
                raw_names = model.get_modelmeta().custom_metadata_map.get("names")
                if raw_names is None:
                    raise ValueError("ONNX model metadata does not contain class names")
                names = ast.literal_eval(raw_names)
            else:
                from ultralytics import YOLO

                model = YOLO(self.model_name)
                names = model.names
            name_items = names.items() if isinstance(names, dict) else enumerate(names)
            canonical_names = {
                int(class_id): YOLO_CLASS_NAME_BY_CASEFOLD.get(str(name).casefold())
                for class_id, name in name_items
            }
            self.model = model
            self.model_class_names = {
                class_id: class_name
                for class_id, class_name in canonical_names.items()
                if class_name is not None
            }
            supported_targets = set(self.model_class_names.values()) & set(
                OUTDOOR_CANDIDATE_CLASSES
            )
            logger.info(
                "YOLO weights loaded successfully model=%s backend=%s class_count=%s "
                "supported_outdoor_target_count=%s",
                self.model_name,
                "onnxruntime-cpu" if self.is_onnx_model else "ultralytics",
                len(self.model_class_names),
                len(supported_targets),
            )

    def get_supported_target_classes(self) -> tuple[str, ...]:
        if settings.DEMO_MODE:
            return OUTDOOR_CANDIDATE_CLASSES
        try:
            self._load_model()
        except Exception as exc:
            logger.exception("YOLO model weights failed to load model=%s", self.model_name)
            raise VerificationUnavailable(
                "The configured YOLO model weights are unavailable or unsupported."
            ) from exc

        supported = set(self.model_class_names.values())
        return tuple(
            class_name
            for class_name in OUTDOOR_CANDIDATE_CLASSES
            if class_name in supported
        )

    def class_id_for_name(self, class_name: str) -> Optional[int]:
        normalized = YOLO_CLASS_NAME_BY_CASEFOLD.get(class_name.casefold())
        if normalized is None:
            return None
        return next(
            (
                class_id
                for class_id, loaded_name in self.model_class_names.items()
                if loaded_name == normalized
            ),
            None,
        )

    async def detect_image(self, image_path: str, target_class: str) -> Detection:
        if settings.DEMO_MODE:
            return Detection(class_name=target_class, confidence=0.95)

        canonical_target = YOLO_CLASS_NAME_BY_CASEFOLD.get(target_class.casefold())
        if canonical_target is None:
            logger.error(
                "YOLO verification rejected unsupported target class=%s",
                target_class,
            )
            raise VerificationUnavailable(
                "This target class is not supported by the configured YOLO model."
            )

        try:
            self._load_model()
        except Exception as exc:
            logger.exception(
                "YOLO model weights failed to load model=%s",
                self.model_name,
            )
            raise VerificationUnavailable(
                "The configured YOLO model weights are unavailable or unsupported."
            ) from exc

        required_class_id = self.class_id_for_name(canonical_target)
        if required_class_id is None:
            logger.error(
                "YOLO target is absent from loaded weights target_class=%s",
                canonical_target,
            )
            raise VerificationUnavailable(
                "This target class is not supported by the configured YOLO model."
            )

        try:
            model = self.model
            if model is None:
                raise RuntimeError("YOLO model did not initialize")
            if self.is_onnx_model:
                detections = self._detect_onnx(model, image_path)
                top_detections = tuple(detections[:5])
                target_detections = [
                    detection
                    for detection in detections
                    if detection[1] == canonical_target
                ]
                best = max(
                    target_detections or detections,
                    key=lambda item: item[2],
                    default=None,
                )
                logger.info(
                    "YOLO ONNX inference complete required_class_id=%s required_class=%s "
                    "threshold=%.2f detection_count=%s top_detections=%s",
                    required_class_id,
                    canonical_target,
                    settings.YOLO_CONFIDENCE_THRESHOLD,
                    len(detections),
                    [
                        (class_id, name, round(confidence, 4))
                        for class_id, name, confidence in top_detections
                    ],
                )
                if best is None:
                    return Detection(
                        class_name=None,
                        confidence=0.0,
                        top_detections=top_detections,
                    )
                return Detection(
                    class_name=best[1],
                    confidence=best[2],
                    top_detections=top_detections,
                    class_id=best[0],
                )
            results = model(image_path, verbose=False)
            detections: list[tuple[int, str, float]] = []

            for result in results:
                for box in result.boxes or []:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model_class_names.get(class_id)
                    if class_name is not None:
                        detections.append((class_id, class_name, confidence))

            detections.sort(key=lambda item: item[2], reverse=True)
            top_detections = tuple(detections[:5])
            target_detections = [
                detection
                for detection in detections
                if detection[1] == canonical_target
            ]
            best = max(target_detections or detections, key=lambda item: item[2], default=None)
            logger.info(
                "YOLO inference complete required_class_id=%s required_class=%s threshold=%.2f "
                "detection_count=%s top_detections=%s",
                required_class_id,
                canonical_target,
                settings.YOLO_CONFIDENCE_THRESHOLD,
                len(detections),
                [
                    (class_id, name, round(confidence, 4))
                    for class_id, name, confidence in top_detections
                ],
            )
            if best is None:
                return Detection(
                    class_name=None,
                    confidence=0.0,
                    top_detections=top_detections,
                )
            return Detection(
                class_name=best[1],
                confidence=best[2],
                top_detections=top_detections,
                class_id=best[0],
            )
        except Exception as exc:
            logger.exception(
                "YOLO inference failed target_class=%s",
                canonical_target,
            )
            raise VerificationUnavailable(
                "YOLO inference failed. Please try again."
            ) from exc

    def _detect_onnx(self, session, image_path: str) -> list[tuple[int, str, float]]:
        import numpy as np
        from PIL import Image

        model_input = session.get_inputs()[0]
        input_shape = model_input.shape
        if (
            len(input_shape) != 4
            or input_shape[0] != 1
            or input_shape[1] != 3
            or not isinstance(input_shape[2], int)
            or not isinstance(input_shape[3], int)
        ):
            raise ValueError("YOLO ONNX model must have a static NCHW image input")

        input_height, input_width = input_shape[2], input_shape[3]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_width, original_height = image.size
        scale = min(input_width / original_width, input_height / original_height)
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)
        resized = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        pad_left = (input_width - resized_width) // 2
        pad_top = (input_height - resized_height) // 2
        letterboxed = Image.new("RGB", (input_width, input_height), (114, 114, 114))
        letterboxed.paste(resized, (pad_left, pad_top))

        tensor = np.asarray(letterboxed, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None] / 255.0)
        output = session.run(None, {model_input.name: tensor})[0]
        predictions = np.asarray(output)[0]
        if predictions.ndim != 2:
            raise ValueError("YOLO ONNX output has an unexpected shape")
        if predictions.shape[0] == 4 + len(self.model_class_names):
            predictions = predictions.T
        if predictions.shape[1] != 4 + len(self.model_class_names):
            raise ValueError("YOLO ONNX output class dimensions do not match metadata")

        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)
        valid = confidences >= 0.25
        boxes = predictions[valid, :4]
        class_ids = class_ids[valid].astype(np.int32, copy=False)
        confidences = confidences[valid]
        if not len(boxes):
            return []

        half_sizes = boxes[:, 2:4] / 2
        corners = np.concatenate((boxes[:, :2] - half_sizes, boxes[:, :2] + half_sizes), axis=1)
        order = np.argsort(confidences)[::-1]
        keep: list[int] = []
        while order.size and len(keep) < 300:
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if not remaining.size:
                break

            intersection_top_left = np.maximum(corners[current, :2], corners[remaining, :2])
            intersection_bottom_right = np.minimum(corners[current, 2:], corners[remaining, 2:])
            intersection_size = np.maximum(
                intersection_bottom_right - intersection_top_left,
                0,
            )
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            current_size = np.maximum(
                corners[current, 2:] - corners[current, :2],
                0,
            )
            remaining_size = np.maximum(
                corners[remaining, 2:] - corners[remaining, :2],
                0,
            )
            current_area = current_size[0] * current_size[1]
            remaining_area = remaining_size[:, 0] * remaining_size[:, 1]
            union = current_area + remaining_area - intersection
            overlap = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            same_class = class_ids[remaining] == class_ids[current]
            order = remaining[~(same_class & (overlap > 0.7))]

        detections = [
            (
                int(class_ids[index]),
                self.model_class_names[int(class_ids[index])],
                float(confidences[index]),
            )
            for index in keep
        ]
        detections.sort(key=lambda detection: detection[2], reverse=True)
        return detections


image_verifier = YOLOImageVerifier()
