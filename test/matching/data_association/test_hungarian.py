"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import MatchingAlgorithm, data_association


def test_hungarian(detections2: List[Box3D], tracks1: List[Box3D]) -> None:
    """."""
    matches, _det, _trk, _, _aff = data_association(
        detections2, tracks1, MetricKind.GIOU_3D, -0.2, MatchingAlgorithm.HUNGARIAN
    )
    assert matches == pytest.approx(np.array([[0, 0]]))
