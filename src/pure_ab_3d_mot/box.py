"""."""

from typing import Sequence

import numpy as np

from .kitti_oxts import roty


class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None) -> None:  # noqa: E741
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.ry = ry  # orientation
        self.s = s  # detection score

    def __repr__(self) -> str:
        return 'Box3D(pose {} {} {} {} size {} {} {} score {})'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s
        )

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x,
            'center_y': bbox.y,
            'center_z': bbox.z,
            'height': bbox.h,
            'width': bbox.w,
            'length': bbox.l,
            'heading': bbox.ry,
        }

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def from_kitti(cls, data: Sequence[float]) -> 'Box3D':
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def array2bbox(cls, data: Sequence[float]) -> 'Box3D':
        # take the format of data of [x,y,z,theta,l,w,h]
        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox


def box2corners3d_camcoord(bbox: Box3D) -> np.ndarray:
    """Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
    convert it to the 8 corners of the 3D box, the box is in the camera coordinate
    with right x, down y, front z

     Returns:
         corners_3d: (8,3) array in rect camera coord

     box corner order is like follows
             1 -------- 0         top is bottom because y direction is negative
            /|         /|
           2 -------- 3 .
           | |        | |
           . 5 -------- 4
           |/         |/
           6 -------- 7

     rect/ref camera coord:
     right x, down y, front z

     x -> w, z -> l, y -> h
    """
    # 3d bounding box dimensions
    l_, w, h = bbox.l, bbox.w, bbox.h

    # 3d bounding box corners
    x_corners = [l_ / 2, l_ / 2, -l_ / 2, -l_ / 2, l_ / 2, l_ / 2, -l_ / 2, -l_ / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # compute rotational matrix around yaw axis
    # -1.57 means straight, so there is a rotation here
    rot_mat = roty(bbox.ry)
    # rotate and translate 3d bounding box
    corners_3d = np.dot(rot_mat, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox.x
    corners_3d[1, :] = corners_3d[1, :] + bbox.y
    corners_3d[2, :] = corners_3d[2, :] + bbox.z
    return np.transpose(corners_3d)
