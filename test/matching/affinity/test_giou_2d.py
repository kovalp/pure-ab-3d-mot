"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


METRIC = MetricKind.GIOU_2D


def test_giou_2d_everything_equal(detections: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, detections, METRIC, [])
    assert pytest.approx(distance) == np.ones((1, 1))


def test_giou_2d_diff_ry(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, tracks, METRIC, [])
    assert pytest.approx(distance) == 0.621537446975708


def test_giou_2d_diff_ry_and_x(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].x += 1
    distance = compute_affinity(detections, tracks, METRIC, [])
    assert pytest.approx(distance) == 0.5594278573989868


def test_giou_2d_diff_ry_and_y(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].y += 1
    distance = compute_affinity(detections, tracks, METRIC, [])
    assert pytest.approx(distance) == 0.621537446975708


def test_giou_2d_diff_ry_and_z(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].z += 1
    distance = compute_affinity(detections, tracks, METRIC, [])
    assert pytest.approx(distance) == 0.5463569760322571


def test_giou_2d_diff_ry_and_h(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].h += 1
    distance = compute_affinity(detections, tracks, METRIC, [])
    assert pytest.approx(distance) == 0.621537446975708
