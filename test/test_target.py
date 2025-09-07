"""."""

import pytest
import numpy as np

from pure_ab_3d_mot.target import Target

@pytest.fixture
def target() -> Target:
    pose = np.array([1., 2., 3., 4., 5., 6., 7])
    return Target(pose, np.array([8., 9., 10., 11., 12.]), 123)


def test_target_init(target: Target) -> None:
    """."""
    assert target.id == 123
    assert target.initial_pose == pytest.approx(np.array([1., 2., 3., 4., 5., 6., 7]))
    assert target.time_since_update == 0
    assert target.hits == 1


def test_target_repr(target: Target) -> None:
    """."""
    ref = 'Target(id 123 state [1. 2. 3. 4. 5. 6. 7. 0. 0. 0.] info [ 8.  9. 10. 11. 12.])'
    assert repr(target) == ref


def test_compute_inn_mat(target: Target) -> None:
    """."""
    target.kf.P[:] = np.eye(10) * 10.0
    target.kf.R[:] = np.eye(7) * 2.0
    assert target.compute_innovation_matrix() == pytest.approx(12.0 * np.eye(7))

def test_get_velocity(target: Target) -> None:
    """."""
    target.kf.x[:] = np.linspace(1.0, 10.0, num=10).reshape(10, 1)
    assert target.get_velocity() == pytest.approx([8., 9., 10.])