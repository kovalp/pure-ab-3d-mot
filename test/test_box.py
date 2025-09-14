"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.box import Box3D, box2corners3d_camcoord


@pytest.fixture
def box() -> Box3D:
    return Box3D(x=1.0, y=2.0, z=3.0, h=4.0, w=5.0, l=6.0, ry=0.678, s=0.789)


def test_repr(box: Box3D) -> None:
    assert repr(box) == 'Box3D(pose 1.0 2.0 3.0 0.678 size 6.0 5.0 4.0 score 0.789)'


def test_basics(box: Box3D) -> None:
    assert box.s == pytest.approx(0.789)


def test_to_dict(box: Box3D) -> None:
    """."""
    dct = box.bbox2dict(box)
    assert dct['center_x'] == pytest.approx(1.0)
    assert dct['center_y'] == pytest.approx(2.0)
    assert dct['center_z'] == pytest.approx(3.0)
    assert dct['height'] == pytest.approx(4.0)
    assert dct['width'] == pytest.approx(5.0)
    assert dct['length'] == pytest.approx(6.0)
    assert dct['heading'] == pytest.approx(0.678)


def test_to_ab_3d_mot_array(box: Box3D) -> None:
    assert box.bbox2array(box) == pytest.approx([1.0, 2.0, 3.0, 0.678, 6.0, 5.0, 4.0, 0.789])
    box.s = None
    assert box.bbox2array(box) == pytest.approx([1.0, 2.0, 3.0, 0.678, 6.0, 5.0, 4.0])


def test_to_kitti_array(box: Box3D) -> None:
    assert box.bbox2array_raw(box) == pytest.approx([4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 0.678, 0.789])
    box.s = None
    assert box.bbox2array_raw(box) == pytest.approx([4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 0.678])


def test_from_kitti_det() -> None:
    box = Box3D.from_kitti([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert repr(box) == 'Box3D(pose 4.0 5.0 6.0 7.0 size 3.0 2.0 1.0 score None)'
    box = Box3D.from_kitti([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.45])
    assert repr(box) == 'Box3D(pose 4.0 5.0 6.0 7.0 size 3.0 2.0 1.0 score 0.45)'


def test_from_ab_3d_mot_det() -> None:
    box = Box3D.array2bbox([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert repr(box) == 'Box3D(pose 1.0 2.0 3.0 4.0 size 5.0 6.0 7.0 score None)'
    box = Box3D.array2bbox([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.45])
    assert repr(box) == 'Box3D(pose 1.0 2.0 3.0 4.0 size 5.0 6.0 7.0 score 0.45)'


def test_box2corners3d_camcoord(box: Box3D) -> None:
    """."""
    corners = box2corners3d_camcoord(box)
    ref = [
        [4.9045778015272194, 2.0, 3.065362005942834],
        [1.7683946913667126, 2.0, -0.8287817381354419],
        [-2.904577801527219, 2.0, 2.934637994057166],
        [0.23160530863328743, 2.0, 6.828781738135442],
        [4.9045778015272194, -2.0, 3.065362005942834],
        [1.7683946913667126, -2.0, -0.8287817381354419],
        [-2.904577801527219, -2.0, 2.934637994057166],
        [0.23160530863328743, -2.0, 6.828781738135442],
    ]
    assert corners == pytest.approx(np.array(ref))
