"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import data_association


def test_21(detections2: List[Box3D], tracks1: List[Box3D]) -> None:
    """."""
    matches, loose_det, _, cost, aff_matrix = data_association(
        detections2, tracks1, MetricKind.MAHALANOBIS_DIST, -0.2, trk_innovation_matrix=[np.eye(7)]
    )
    assert matches == pytest.approx(np.array([[0, 0]]))
    assert loose_det == pytest.approx([1])
    assert cost == pytest.approx(0.0)
    assert aff_matrix == pytest.approx(np.array([[0.0], [-1.0770329]]))
