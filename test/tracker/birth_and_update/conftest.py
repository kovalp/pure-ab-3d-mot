"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.tracker import Ab3DMot


@pytest.fixture
def boxes() -> List[Box3D]:
    return [Box3D(x=2, y=3, z=4, h=4, w=5, l=6, ry=0.678, s=0.789, ann_id=123)]


@pytest.fixture
def info() -> np.ndarray:
    return np.linspace(1, 5, num=5).reshape(1, 5)


@pytest.fixture
def trk_t2(tracker1: Ab3DMot) -> Ab3DMot:
    tracker1.trackers[0].time_since_update = 2
    return tracker1
