"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import data_association


def test_large_threshold(detections2: List[Box3D], tracks1: List[Box3D]) -> None:
    """."""
    matches, loose_det, loose_trk, cost, aff_matrix = data_association(
        detections2, tracks1, MetricKind.GIOU_3D, 1.1
    )
    assert matches.shape == (0, 2)
    assert loose_trk == pytest.approx([0])
    assert loose_det == pytest.approx([1, 0])
    assert cost == pytest.approx(-1.0)
    assert aff_matrix == pytest.approx(np.array([[1.0], [0.546357]]))
