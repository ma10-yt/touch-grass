from pathlib import Path

from app.ai.evaluate_yolo_models import (
    _confidence_summary,
    _negative_groups,
    _percentile,
)


def test_confidence_metrics_summarize_real_predictions_without_fabricating_misses():
    assert _confidence_summary([None, 0.45, 0.6, 0.8]) == {
        "detected": 3,
        "mean": 0.6167,
        "median": 0.6,
        "p90": 0.76,
    }
    assert _confidence_summary([None, None]) == {
        "detected": 0,
        "mean": None,
        "median": None,
        "p90": None,
    }


def test_percentile_and_hard_negative_groups_are_deterministic(tmp_path: Path):
    assert _percentile([1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([], 0.9) is None

    suitcase = tmp_path / "hard_negatives" / "suitcase" / "case.jpg"
    handbag = tmp_path / "hard_negatives" / "handbag" / "bag.png"
    suitcase.parent.mkdir(parents=True)
    handbag.parent.mkdir(parents=True)
    suitcase.touch()
    handbag.touch()
    grouped = _negative_groups(
        [suitcase, handbag],
        {suitcase: 0.2, handbag: None},
    )

    assert grouped == {"suitcase": [0.2], "handbag": [None]}
