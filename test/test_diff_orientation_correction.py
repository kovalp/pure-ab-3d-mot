"""."""

import pytest

from pure_ab_3d_mot.dist_metrics import diff_orientation_correction


def test_diff_orientation_correction() -> None:
    """."""
    assert diff_orientation_correction(0.5) == pytest.approx(0.5)
    assert diff_orientation_correction(2.5) == pytest.approx(-0.6415926535897931)
    assert diff_orientation_correction(-2.5) == pytest.approx(0.6415926535897931)
