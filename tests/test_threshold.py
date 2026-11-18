"""Validation-chosen decision thresholds."""

import numpy as np
import pytest

from tbmxai.evaluation.threshold import select_threshold


def test_youden_and_sensitivity_thresholds() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.35, 0.40, 0.45])
    thr = select_threshold(y, p, "youden")
    assert thr == pytest.approx(0.35)  # ≥0.35 → sens 0.75, spec 1.0 (J = 0.75, the maximum)
    thr_s = select_threshold(y, p, "sensitivity", target_sensitivity=1.0)
    assert thr_s == pytest.approx(0.25) and ((p >= thr_s)[y == 1]).all()
    with pytest.raises(ValueError):
        select_threshold(np.zeros(4), p[:4])


def test_sensitivity_threshold_is_the_highest_score_reaching_the_target() -> None:
    # Ten positives in a row above all negatives: the ROC curve is one vertical segment, so
    # 90 % sensitivity is first reached at the ninth-highest positive score.
    pos = np.linspace(0.99, 0.90, 10)
    neg = np.linspace(0.50, 0.10, 10)
    y = np.r_[np.ones(10), np.zeros(10)]
    thr = select_threshold(y, np.r_[pos, neg], "sensitivity", target_sensitivity=0.9)
    assert thr == pytest.approx(pos[8])
    assert (pos >= thr).mean() == pytest.approx(0.9)
