"""."""

import math

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


METRIC = MetricKind.MAHALANOBIS_DIST


def test_m_dist_everything_equal(detections: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, detections, METRIC, [np.eye(7)])
    assert distance == pytest.approx(np.zeros((1, 1)))


def test_m_dist_everything_equal_no_inn_mat(detections: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, detections, METRIC, [None])
    assert distance == pytest.approx(np.zeros((1, 1)))


def test_m_dist_diff_ry_no_inn_mat(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, tracks, METRIC, [None])
    assert distance == pytest.approx(-0.4)


def test_m_dist_diff_ry(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, tracks, METRIC, [np.eye(7) * 0.123])
    assert distance == pytest.approx(-0.4 * math.sqrt(0.123))
