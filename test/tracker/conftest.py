"""."""

from typing import Dict, List, Union

import numpy as np
import pytest

from pure_ab_3d_mot.tracker import Ab3DMot


@pytest.fixture
def tracker() -> Ab3DMot:
    return Ab3DMot()


@pytest.fixture
def det_reports0() -> Dict[str, Union[List[List[float]], np.ndarray]]:
    """."""
    return {'dets': [], 'info': []}


@pytest.fixture
def det_reports1() -> Dict[str, Union[List[List[float]], np.ndarray]]:
    """."""
    return {
        'dets': np.array([[8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
        'info': np.array([[1.1, 2.1, 3.1, 4.1, 5.1]]),
    }
