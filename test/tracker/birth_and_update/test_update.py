"""."""

from typing import List

import numpy as np
import pytest

from association_quality_clavia import UPD_ID_LOOSE

from pure_ab_3d_mot.box import Box3D
from pure_ab_3d_mot.tracker import Ab3DMot


def test_with_match(trk_t2: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    matches = np.zeros((1, 2), int)
    trk_t2.update(matches, [], boxes, info)
    assert len(trk_t2.trackers) == 1
    # fmt: off
    ref = [1.9090909090909092, 2.909090909090909, 3.909090909090909,
           0.6944006678554734, 5.909090909090909, 5.090909090909091,
           4.2727272727272725, 0.0, 0.0, 0.0]
    # fmt: on
    target = trk_t2.trackers[0]
    assert target.kf.x[:, 0] == pytest.approx(ref)
    assert target.time_since_update == 0
    assert target.hits == 2
    assert target.ann_id == 234
    assert target.upd_id == 123
    assert isinstance(target.ann_id, int)
    assert isinstance(target.upd_id, int)


def test_no_match(trk_t2: Ab3DMot, boxes: List[Box3D], info: np.ndarray) -> None:
    matches = np.zeros((0, 2), int)
    trk_t2.update(matches, [0], boxes, info)
    assert len(trk_t2.trackers) == 1
    ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0]
    target = trk_t2.trackers[0]
    assert target.kf.x[:, 0] == pytest.approx(ref)
    assert target.time_since_update == 2
    assert target.hits == 1
    assert target.ann_id == 234
    assert target.upd_id == UPD_ID_LOOSE
    assert isinstance(target.ann_id, int)
    assert isinstance(target.upd_id, int)
