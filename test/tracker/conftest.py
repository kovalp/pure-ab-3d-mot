"""."""

from typing import Dict, List, Union

import numpy as np
import pytest

from pure_ab_3d_mot.str_const import ANN_IDS, DETS, INFO
from pure_ab_3d_mot.target import Target
from pure_ab_3d_mot.tracker import Ab3DMot


@pytest.fixture
def tracker() -> Ab3DMot:
    return Ab3DMot()


@pytest.fixture
def det_reports0() -> Dict[str, Union[List[List[float]], np.ndarray]]:
    """."""
    return {DETS: [], INFO: []}


@pytest.fixture
def det_reports1() -> Dict[str, Union[List[List[float]], np.ndarray]]:
    """."""
    return {
        DETS: np.array([[8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
        INFO: np.array([[1.1, 2.1, 3.1, 4.1, 5.1]]),
    }


@pytest.fixture
def det_ann_id1(det_reports1) -> Dict[str, Union[List[List[float]], np.ndarray]]:
    """."""
    det_reports1[ANN_IDS] = np.array([1])
    return det_reports1


@pytest.fixture
def tracker1(tracker: Ab3DMot) -> Ab3DMot:
    pose = np.linspace(1.0, 7.0, num=7)
    info = np.linspace(8.0, 8.0 + 4.0, num=5)
    target = Target(pose, info, 123, ann_id=234)
    tracker.trackers.append(target)
    return tracker
