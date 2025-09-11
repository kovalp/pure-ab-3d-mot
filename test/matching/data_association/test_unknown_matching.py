"""."""

from typing import List

import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import data_association


def test_unknown_matching(detections2: List[Box3D], tracks1: List[Box3D]) -> None:
    """."""
    with pytest.raises(AssertionError):
        data_association(detections2, tracks1, MetricKind.IOU_3D, -0.2, 'unknown')
