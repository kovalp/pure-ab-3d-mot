"""."""

from typing import Dict

from pure_ab_3d_mot.tracker import Ab3DMot


def test_second_det_with_ann_ids(tracker: Ab3DMot, det_ann_id1: Dict) -> None:
    """."""
    tracker.track(det_ann_id1)
    tracker.track(det_ann_id1)
    assert len(tracker.trackers) == 1
    target = tracker.trackers[0]
    assert target.ann_id == 1
    assert target.upd_id == 1
    assert isinstance(target.ann_id, int)
    assert isinstance(target.upd_id, int)
