"""."""

from typing import Dict

from pure_ab_3d_mot.target import Target
from pure_ab_3d_mot.tracker import Ab3DMot


def test_mahalanobis(tracker: Ab3DMot, det_reports1: Dict) -> None:
    """."""
    tracker.metric = tracker.metric.MAHALANOBIS_DIST
    tracker.track(det_reports1)
    assert len(tracker.trackers) == 1
    assert isinstance(tracker.trackers[0], Target)
