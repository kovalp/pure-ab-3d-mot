"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.tracker import Ab3DMot


@pytest.fixture
def boxes() -> List[Box3D]:
    return [Box3D(x=1, y=2, z=3, h=4, w=5, l=6, ry=0.678, s=0.789, ann_id=123)]


@pytest.fixture
def info() -> np.ndarray:
    return np.linspace(1, 5, num=5).reshape(1, 5)


def test_no_unmatched_det(tracker: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    tracker.birth(boxes, info, [])
    assert len(tracker.trackers) == 0


def test_1_unmatched_det(tracker: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    tracker.birth(boxes, info, [0])
    assert len(tracker.trackers) == 1
    assert tracker.trackers[0].ann_id == 123
