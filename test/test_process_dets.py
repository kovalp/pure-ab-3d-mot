"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.process_dets import process_dets


def test_process_detections() -> None:
    """."""
    detections = np.linspace(1.0, 14.0, num=14).reshape((2, 7))
    boxes = process_dets(detections)
    assert len(boxes) == 2
    assert boxes[0].h == pytest.approx(1.0)
    assert boxes[0].w == pytest.approx(2.0)
    assert boxes[0].l == pytest.approx(3.0)
    assert boxes[0].x == pytest.approx(4.0)
    assert boxes[0].y == pytest.approx(5.0)
    assert boxes[0].z == pytest.approx(6.0)
    assert boxes[0].ry == pytest.approx(7.0)

    assert boxes[1].h == pytest.approx(8.0)
    assert boxes[1].w == pytest.approx(9.0)
    assert boxes[1].l == pytest.approx(10.0)
    assert boxes[1].x == pytest.approx(11.0)
    assert boxes[1].y == pytest.approx(12.0)
    assert boxes[1].z == pytest.approx(13.0)
    assert boxes[1].ry == pytest.approx(14.0)
