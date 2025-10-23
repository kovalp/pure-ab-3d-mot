"""."""

import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.iou import iou


@pytest.fixture
def detection() -> Box3D:
    return Box3D.array2bbox([1, 2, 3, 0.7, 5, 6, 7])


@pytest.fixture
def track() -> Box3D:
    return Box3D.array2bbox([1, 2, 3, 0.3, 5, 6, 7])


def test_iou_unknown_metric(detection: Box3D, track: Box3D) -> None:
    with pytest.raises(ValueError):
        iou(detection, track, MetricKind.UNKNOWN)
