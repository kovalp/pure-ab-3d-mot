"""."""

from typing import List

import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import compute_affinity


METRIC = MetricKind.DIST_3D


def test_diff_ry_xyz(detections: List[Box3D], tracks: List[Box3D]) -> None:
    """."""
    tracks[0].x += 1
    assert compute_affinity(detections, tracks, METRIC, []) == pytest.approx(-1.0)
    tracks[0].y += 1
    assert compute_affinity(detections, tracks, METRIC, []) == pytest.approx(-1.4142135)
    tracks[0].z += 1
    assert compute_affinity(detections, tracks, METRIC, []) == pytest.approx(-1.7320508)
