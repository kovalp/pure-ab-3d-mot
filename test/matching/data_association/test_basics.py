"""."""

from typing import List

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.dist_metrics import MetricKind
from pure_ab_3d_mot.matching import data_association


def test_num_det_2_num_trk_1(detections2: List[Box3D], tracks1: List[Box3D]) -> None:
    """."""
    matches, loose_det, loose_trk, cost, aff_matrix = data_association(
        detections2, tracks1, MetricKind.GIOU_3D, -0.2
    )
    assert matches == pytest.approx(np.array([[0, 0]]))
    assert matches.dtype == int
    assert loose_det == pytest.approx([1])
    assert loose_det.dtype == int
    assert loose_trk.shape == (0,)
    assert loose_trk.dtype == int
    assert cost == pytest.approx(-1.0)
    assert aff_matrix == pytest.approx(np.array([[1.0], [0.546357]]))
    assert aff_matrix.dtype == 'float32'


def test_num_det_0_num_trk_1(tracks1: List[Box3D]) -> None:
    """."""
    matches, loose_det, loose_trk, cost, aff_matrix = data_association(
        [], tracks1, MetricKind.GIOU_3D, -0.2
    )
    assert matches.shape == (0, 2)
    assert loose_det.shape == (0,)
    assert loose_det.dtype == int
    assert loose_trk == pytest.approx([0])
    assert loose_trk.shape == (1,)
    assert loose_trk.dtype == int
    assert cost == pytest.approx(0.0)
    assert aff_matrix.shape == (0, 1)
    assert aff_matrix.dtype == 'float32'


def test_num_det_2_num_trk_0(detections2: List[Box3D]) -> None:
    """."""
    matches, loose_det, loose_trk, cost, aff_matrix = data_association(
        detections2, [], MetricKind.GIOU_3D, -0.2
    )
    assert matches.shape == (0, 2)
    assert loose_det == pytest.approx([0, 1])
    assert loose_det.shape == (2,)
    assert loose_det.dtype == int
    assert loose_trk.shape == (0,)
    assert loose_trk.dtype == int
    assert cost == pytest.approx(0.0)
    assert aff_matrix.shape == (2, 0)
    assert aff_matrix.dtype == 'float32'
