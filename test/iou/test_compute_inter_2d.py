import numpy as np
import pytest

from pure_ab_3d_mot.iou import compute_inter_2d


def test_compute_inter_2d() -> None:
    corners_a = np.array(
        [
            [6.6044949, 13.9222401],
            [6.67336983, 13.30712306],
            [8.4934111, 13.5109139],
            [8.42453617, 14.12603094],
        ]
    )
    corners_b = np.array(
        [
            [6.6319858, 13.8903007],
            [6.70094754, 13.29372935],
            [8.4944742, 13.5010553],
            [8.42551246, 14.09762665],
        ]
    )
    i_2d = compute_inter_2d(corners_a, corners_b)
    assert i_2d == pytest.approx(1.0605266793005177)
