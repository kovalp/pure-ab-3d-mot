"""."""

from typing import List
import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


def test_giou_3d_everything_equal(detections: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, detections, MetricKind.GIOU_3D, None)
    assert pytest.approx(distance) == np.ones((1, 1))


def test_giou_3d_diff_ry(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, tracks, MetricKind.GIOU_3D, None)
    assert pytest.approx(distance) == 0.4285714328289032


def test_giou_3d_diff_ry_and_x(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].x += 1
    distance = compute_affinity(detections, tracks, MetricKind.GIOU_3D, None)
    assert pytest.approx(distance) == 0.17274464666843414


def test_giou_3d_diff_ry_and_h(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].h += 1
    distance = compute_affinity(detections, tracks, MetricKind.GIOU_3D, None)
    assert pytest.approx(distance) == 0.5384615659713745
