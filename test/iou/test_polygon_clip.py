"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.iou import Poly2DT, polygon_clip


def test_polygon_clip_everything_equal(poly: Poly2DT) -> None:
    """."""
    inter_poly = polygon_clip(poly, poly)
    assert inter_poly == pytest.approx(poly)


def test_polygon_clip_10_diff_down(poly: Poly2DT) -> None:
    """."""
    poly2 = poly.copy()
    poly2[0] = poly[0][0], -1.0
    inter_poly = polygon_clip(poly, poly2)
    assert inter_poly == pytest.approx(poly)


def test_polygon_clip_10_diff_up(poly: Poly2DT) -> None:
    """."""
    poly2 = poly.copy()
    poly2[0] = poly[0][0], +1.0
    inter_poly = polygon_clip(poly, poly2)
    ref = [
        (-2.9045165380828144, 2.9310753756789847),
        (0.36370156623524513, 1.0),
        (4.904516538082815, 3.0689246243210153),
        (4.904516538082815, 3.0689246243210153),
        (1.6362984337647548, 6.8529371008606565),
        (-2.9045165380828144, 2.9310753756789847),
    ]
    assert inter_poly == pytest.approx(np.array(ref))


def test_no_intersection(poly: Poly2DT) -> None:
    """."""
    poly2 = poly.copy()
    poly2[:, 1] += 10.0
    assert polygon_clip(poly, poly2) is None
