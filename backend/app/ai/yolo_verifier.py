import logging
from dataclasses import dataclass
from typing import Optional
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
        self.model = None
        self.model_class_names: dict[int, str] = {}

    def _load_model(self):
        if self.model is None and not settings.DEMO_MODE:
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
                "YOLO weights loaded successfully model=%s class_count=%s "
                "supported_outdoor_target_count=%s",
                self.model_name,
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


image_verifier = YOLOImageVerifier()
