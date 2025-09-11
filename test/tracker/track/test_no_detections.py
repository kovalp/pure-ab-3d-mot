"""."""

from typing import Dict

import numpy as np

from pure_ab_3d_mot.tracker import Ab3DMot


def test_no_detections(tracker: Ab3DMot, det_reports0: Dict) -> None:
    """."""
    results = tracker.track(det_reports0)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)
    assert results[0].shape == (0, 15)
    assert len(tracker.trackers) == 0
