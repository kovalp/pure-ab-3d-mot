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
    ref0 = [[8., 2., 3., 4., 5., 6., 0.7168146928204138, 1., 1.1, 2.1, 3.1, 4.1, 5.1]]
    assert results[0] == pytest.approx(np.array(ref0))
    assert len(tracker.trackers) == 1
    assert tracker.id_now_output == pytest.approx([1.])

