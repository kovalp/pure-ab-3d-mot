"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


METRIC = MetricKind.EULER


def test_everything_equal(detections: List[Box3D]) -> None:
    """."""
    assert compute_affinity(detections, detections, METRIC, []) == pytest.approx(np.zeros((1, 1)))


def test_diff_ry(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    assert compute_affinity(detections, tracks, METRIC, []) == pytest.approx(-0.4)
