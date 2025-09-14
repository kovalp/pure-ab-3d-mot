"""."""

import pytest

from pure_ab_3d_mot.orientation_correction import orientation_correction, within_range


def test_within_range() -> None:
    """."""
    assert within_range(0.1) == pytest.approx(0.1)
    assert within_range(4.1) == pytest.approx(-2.1831853071795866)
    assert within_range(-4.1) == pytest.approx(2.1831853071795866)


def test_orientation_correction() -> None:
    """."""
    t_p, t_d = orientation_correction(0.321, 0.123)
    assert t_p == pytest.approx(0.321)
    assert t_d == pytest.approx(0.123)

    t_p, t_d = orientation_correction(0.321, 4.123)
    assert t_p == pytest.approx(-2.820592653589793)
    assert t_d == pytest.approx(-2.160185307179586)

    t_p, t_d = orientation_correction(-6.321, 4.123)
    assert t_p == pytest.approx(-3.1794073464102066)
    assert t_d == pytest.approx(-2.160185307179586)

    t_p, t_d = orientation_correction(7.0, 9.0)
    assert t_p == pytest.approx(3.858407346410207)
    assert t_d == pytest.approx(2.7168146928204138)

    t_p, t_d = orientation_correction(9.0, 4.0)
    assert t_p == pytest.approx(-3.5663706143591725)
    assert t_d == pytest.approx(-2.2831853071795862)
