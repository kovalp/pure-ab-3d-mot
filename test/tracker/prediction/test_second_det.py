"""."""

from typing import Dict

import numpy as np
import pytest

from pure_ab_3d_mot.tracker import Ab3DMot


def test_second_det(tracker: Ab3DMot, det_reports1: Dict) -> None:
    """."""
    tracker.track(det_reports1)
    tracker.prediction()
    assert len(tracker.trackers) == 1
    x_ref = np.array([4.0, 5.0, 6.0, 0.71681469282, 3.0, 2.0, 8.0, 0.0, 0.0, 0.0]).reshape(10, 1)
    assert tracker.trackers[0].kf.x == pytest.approx(x_ref)
    assert tracker.trackers[0].time_since_update == 1
    assert tracker.trackers[0].initial_pose == pytest.approx([4.0, 5.0, 6.0, 7.0, 3.0, 2.0, 8.0])
