"""."""

from typing import Dict

import numpy as np
import pytest

from pure_ab_3d_mot.tracker import Ab3DMot


def test_second_det(tracker: Ab3DMot, det_reports1: Dict) -> None:
    """."""
    tracker.track(det_reports1)
    results = tracker.track(det_reports1)
    assert len(results) == 1
    ref0 = [[8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7168146928204138, 1.0, 1.1, 2.1, 3.1, 4.1, 5.1]]
    assert results[0] == pytest.approx(np.array(ref0))
    assert len(tracker.trackers) == 1
    assert tracker.id_now_output == pytest.approx([1.0])
    assert tracker.frame_count == 2
