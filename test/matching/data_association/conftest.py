"""."""

from typing import List

import pytest

from pure_ab_3d_mot.box import Box3D


@pytest.fixture
def detections2() -> List[Box3D]:
    return [Box3D.array2bbox([1, 2, 3, 0.3, 5, 6, 7]), Box3D.array2bbox([1, 2, 4, 0.7, 5, 6, 7])]


@pytest.fixture
def tracks1() -> List[Box3D]:
    return [Box3D.array2bbox([1, 2, 3, 0.3, 5, 6, 7])]
