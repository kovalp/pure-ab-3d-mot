"""."""

from typing import List

import numpy as np

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.tracker import Ab3DMot


def test_no_unmatched_det(tracker: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    tracker.birth(boxes, info, [])
    assert len(tracker.trackers) == 0


def test_1_unmatched_det(tracker: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    tracker.birth(boxes, info, [0])
    assert len(tracker.trackers) == 1
    target = tracker.trackers[0]
    assert target.ann_id == 123
    assert isinstance(target.ann_id, int)
    assert isinstance(target.upd_id, int)
