"""."""

from typing import Dict

import numpy as np
import pytest

from pure_ab_3d_mot.target import Target
from pure_ab_3d_mot.tracker import Ab3DMot


def test_first_det(tracker: Ab3DMot, det_reports1: Dict) -> None:
    """."""
    results = tracker.track(det_reports1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)
    result0_ref = [[8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.1, 2.1, 3.1, 4.1, 5.1]]
    assert results[0] == pytest.approx(np.array(result0_ref))
    assert len(tracker.trackers) == 1
    assert isinstance(tracker.trackers[0], Target)
    target = tracker.trackers[0]
    assert target.id == 1
    assert target.info == pytest.approx([1.1, 2.1, 3.1, 4.1, 5.1])
    assert target.hits == 1
    assert target.time_since_update == 0
    x_ref = np.array([4.0, 5.0, 6.0, 7.0, 3.0, 2.0, 8.0, 0.0, 0.0, 0.0]).reshape((10, 1))
    assert target.kf.x == pytest.approx(x_ref)
    ini_pose_ref = [4.0, 5.0, 6.0, 7.0, 3.0, 2.0, 8.0]
    assert target.initial_pose == pytest.approx(ini_pose_ref)
    assert tracker.id_now_output == pytest.approx([1.0])
    assert tracker.frame_count == 1
