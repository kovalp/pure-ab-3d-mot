"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.iou import convex_hull_intersection


def test_all_equal(poly: np.ndarray) -> None:
    """."""
    inter_poly, area = convex_hull_intersection(poly, poly)
    assert inter_poly == pytest.approx(poly)
    assert area == pytest.approx(30.0)


def test_no_intersection(poly: np.ndarray) -> None:
    """."""
    poly2 = poly.copy()
    poly2[:, 1] += 10.0
    inter_poly, area = convex_hull_intersection(poly, poly2)
    assert inter_poly is None
    assert area == pytest.approx(0.0)
