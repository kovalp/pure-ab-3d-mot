"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.matching import greedy_matching


def test_greedy_matching() -> None:
    """."""
    cost_mat = -np.array((1.0, 2.0, 4.0, 3.0, 5.0, 6.0)).reshape(2, 3)
    matched_indices = greedy_matching(cost_mat)
    assert matched_indices == pytest.approx(np.array([[1, 2], [0, 1]]))


def test_greedy_matching_34() -> None:
    """."""
    cost_mat = -np.array((1.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0, 8.0, 9.0, 1.0, 7.0, 3.0)).reshape(3, 4)
    matched_indices = greedy_matching(cost_mat)
    assert matched_indices == pytest.approx(np.array([[2, 0], [1, 3], [0, 2]]))
