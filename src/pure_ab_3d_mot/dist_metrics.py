"""."""

from enum import Enum

import numpy as np

import math

from .box import Box3D, box2corners3d_camcoord


class MetricKind(Enum):
    """."""
    IOU_3D = 'iou_3d'
    GIOU_3D = 'giou_3d'
    IOU_2D = 'iou_2d'
    GIOU_2D = 'giou_2d'
    MAHALANOBIS_DIST = 'm_dis'


#################### distance metric


def dist_ground(bbox1, bbox2):
    # Compute distance of bottom center in 3D space, NOT considering the difference in height

    c1 = Box3D.bbox2array(bbox1)[[0, 2]]
    c2 = Box3D.bbox2array(bbox2)[[0, 2]]
    dist = np.linalg.norm(c1 - c2)

    return dist


def dist3d_bottom(bbox1, bbox2):
    # Compute distance of bottom center in 3D space, considering the difference in height / 2

    c1 = Box3D.bbox2array(bbox1)[:3]
    c2 = Box3D.bbox2array(bbox2)[:3]
    dist = np.linalg.norm(c1 - c2)

    return dist


def dist3d(bbox1, bbox2):
    # Compute distance of actual center in 3D space, considering the difference in height

    corners1 = box2corners3d_camcoord(bbox1)  # 8 x 3
    corners2 = box2corners3d_camcoord(bbox2)  # 8 x 3

    # compute center point based on 8 corners
    c1 = np.average(corners1, axis=0)
    c2 = np.average(corners2, axis=0)

    dist = np.linalg.norm(c1 - c2)

    return dist


def diff_orientation_correction(diff: float) -> float:
    """
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    """
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    return diff


def m_distance(det, trk, trk_inv_innovation_matrix=None) -> float:
    # compute difference
    det_array = Box3D.bbox2array(det)[:7]
    trk_array = Box3D.bbox2array(trk)[:7]  # (7, )
    diff = np.expand_dims(det_array - trk_array, axis=1)  # 7 x 1

    # correct orientation
    corrected_yaw_diff = diff_orientation_correction(diff[3])
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        sqr = np.dot(diff.T, trk_inv_innovation_matrix.dot(diff))[0, 0]
    else:
        sqr = np.dot(diff.T, diff)[0, 0]  # distance along 7 dimension
    return float(math.sqrt(sqr))
