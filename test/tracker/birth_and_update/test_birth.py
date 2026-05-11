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


def test_kf_covariances(tracker: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    tracker.measurement_std_dev = 2.0
    tracker.proc_std_dev = 3.0
    tracker.proc_vel_std_dev = 0.2
    tracker.birth(boxes, info, [0])
    assert len(tracker.trackers) == 1
    kf = tracker.trackers[0].kf
    assert np.allclose(kf.R, 4.0 * np.eye(7))
    assert np.allclose(kf.Q[:7, :7], 9.0 * np.eye(7))
    assert np.allclose(kf.Q[7:, 7:], 0.04 * np.eye(3))
    assert np.allclose(kf.Q[7:, :7], 0.0)
    assert np.allclose(kf.Q[:7, 7:], 0.0)
