"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.kitti_oxts import roty


def test_rot_y() -> None:
    """."""
    mat_y = roty(0.567)
    ref = [
        [0.8435160803285807, 0.0, 0.537103921254637],
        [0.0, 1.0, 0.0],
        [-0.537103921254637, 0.0, 0.8435160803285807],
    ]
    assert mat_y == pytest.approx(np.array(ref))
