"""Compare the configured YOLO model and yolo11s on a labeled local image set."""

import argparse
import gc
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EVALUATION_CONFIDENCE = 0.001
IMAGE_SIZE = 640
REPORT_THRESHOLDS = tuple(sorted({0.45, float(settings.YOLO_CONFIDENCE_THRESHOLD)}))


class PeakMemorySampler:
    def __init__(self, process, interval_seconds: float = 0.02):
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self.peak_bytes = self.process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(
                    self.peak_bytes,
                    self.process.memory_info().rss,
                )
            except Exception:
                return
            self._stop.wait(self.interval_seconds)


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _confidence_summary(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    detections = [value for value in values if value is not None]
    p90 = _percentile(detections, 0.90)
    return {
        "detected": len(detections),
        "mean": round(statistics.mean(detections), 4) if detections else None,
        "median": round(statistics.median(detections), 4) if detections else None,
        "p90": round(p90, 4) if p90 is not None else None,
    }


def _images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def _negative_groups(
    images: list[Path],
    scores: dict[Path, Optional[float]],
) -> dict[str, list[Optional[float]]]:
    groups: dict[str, list[Optional[float]]] = {}
    for image in images:
        group = (
            image.parent.name
            if image.parent.name != "hard_negatives"
            else "uncategorized"
        )
        groups.setdefault(group, []).append(scores[image])
    return groups


def evaluate_model(model_name: str, positive_images: list[Path], negative_images: list[Path]):
    try:
        import psutil
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Install the backend requirements before running detector evaluation."
        ) from exc

    process = psutil.Process()
    memory_before = process.memory_info().rss
    images = [*positive_images, *negative_images]
    scores: dict[Path, Optional[float]] = {}
    inference_seconds = []
    memory = PeakMemorySampler(process)
    memory.__enter__()
    try:
        load_started = time.perf_counter()
        model = YOLO(model_name)
        load_seconds = time.perf_counter() - load_started
        names = model.names
        name_items = names.items() if isinstance(names, dict) else enumerate(names)
        backpack_ids = {
            int(class_id)
            for class_id, class_name in name_items
            if str(class_name).strip().casefold() == "backpack"
        }
        if not backpack_ids:
            raise RuntimeError(
                f"Model {model_name!r} has no canonical 'backpack' class."
            )
        memory_after_load = process.memory_info().rss
        for index, image_path in enumerate(images):
            started = time.perf_counter()
            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=EVALUATION_CONFIDENCE,
                device="cpu",
                verbose=False,
            )[0]
            inference_seconds.append(time.perf_counter() - started)
            matching = [
                float(box.conf[0])
                for box in (result.boxes or [])
                if int(box.cls[0]) in backpack_ids
            ]
            scores[image_path] = max(matching, default=None)

            # The first positive image is a warm-up and is excluded consistently.
            if index == 0:
                inference_seconds.pop()
    finally:
        memory.__exit__(None, None, None)

    positive_scores = [scores[path] for path in positive_images]
    negative_scores = [scores[path] for path in negative_images]
    thresholds = {}
    for threshold in REPORT_THRESHOLDS:
        thresholds[f"{threshold:.2f}"] = {
            "backpack_recall": round(
                sum(score is not None and score >= threshold for score in positive_scores)
                / len(positive_scores),
                4,
            ),
            "hard_negative_false_positive_rate": round(
                sum(score is not None and score >= threshold for score in negative_scores)
                / len(negative_scores),
                4,
            ),
            "hard_negative_groups": {
                group: {
                    "images": len(group_scores),
                    "false_positive_rate": round(
                        sum(
                            score is not None and score >= threshold
                            for score in group_scores
                        )
                        / len(group_scores),
                        4,
                    ),
                }
                for group, group_scores in _negative_groups(
                    negative_images, scores
                ).items()
            },
        }

    result = {
        "model": model_name,
        "backpack_class_ids": sorted(backpack_ids),
        "image_size": IMAGE_SIZE,
        "prediction_confidence_floor": EVALUATION_CONFIDENCE,
        "images": {
            "positive_backpack": len(positive_images),
            "hard_negative": len(negative_images),
        },
        "threshold_metrics": thresholds,
        "confidence_distribution": {
            "positive_backpack": _confidence_summary(positive_scores),
            "hard_negative": _confidence_summary(negative_scores),
            "hard_negative_groups": {
                group: _confidence_summary(group_scores)
                for group, group_scores in _negative_groups(
                    negative_images, scores
                ).items()
            },
        },
        "cpu_inference_ms": {
            "mean": round(statistics.mean(inference_seconds) * 1000, 2)
            if inference_seconds else None,
            "median": round(statistics.median(inference_seconds) * 1000, 2)
            if inference_seconds else None,
        },
        "model_load_seconds": round(load_seconds, 2),
        "process_rss_before_mb": round(memory_before / 1024**2, 1),
        "process_rss_model_loaded_mb": round(memory_after_load / 1024**2, 1),
        "process_peak_rss_model_load_and_inference_mb": round(
            memory.peak_bytes / 1024**2, 1
        ),
    }
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Folder containing positive_backpack/ and hard_negatives/ labeled images.",
    )
    parser.add_argument(
        "--challenger",
        default="yolo11s.pt",
        help="Second model to compare (default: yolo11s.pt).",
    )
    args = parser.parse_args()
    positive_dir = args.dataset / "positive_backpack"
    negative_dir = args.dataset / "hard_negatives"
    positive_images = _images(positive_dir)
    negative_images = _images(negative_dir)
    if not positive_images or not negative_images:
        parser.error(
            "Dataset must contain at least one image in both positive_backpack/ "
            "and hard_negatives/."
        )
    if len(positive_images) < 30:
        print(
            f"WARNING: only {len(positive_images)} positive images; "
            "30 or more diverse backpack photos are recommended.",
            flush=True,
        )

    report = {
        "conditions": {
            "device": "cpu",
            "image_size": IMAGE_SIZE,
            "prediction_confidence_floor": EVALUATION_CONFIDENCE,
            "verification_threshold": settings.YOLO_CONFIDENCE_THRESHOLD,
            "models": [settings.YOLO_MODEL, args.challenger],
        },
        "results": [
            evaluate_model(settings.YOLO_MODEL, positive_images, negative_images),
            evaluate_model(args.challenger, positive_images, negative_images),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
