"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


def test_iou_3d_everything_equal(detections: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, detections, MetricKind.IOU_3D, [])
    assert pytest.approx(distance) == np.ones((1, 1))


def test_iou_3d_diff_ry(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    distance = compute_affinity(detections, tracks, MetricKind.IOU_3D, [])
    assert pytest.approx(distance) == 0.757396936416626


def test_iou_3d_diff_ry_and_x(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].x += 1
    distance = compute_affinity(detections, tracks, MetricKind.IOU_3D, [])
    assert pytest.approx(distance) == 0.6247208714485168


def test_iou_3d_diff_ry_and_h(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].h += 1
    distance = compute_affinity(detections, tracks, MetricKind.IOU_3D, [])
    assert pytest.approx(distance) == 0.6729257106781006
