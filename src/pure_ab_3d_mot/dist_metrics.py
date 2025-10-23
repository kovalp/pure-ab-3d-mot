"""."""

import math

from enum import Enum

import numpy as np

from .box import Box3D, box2corners3d_camcoord


class MetricKind(Enum):
    """."""

    IOU_3D = 'iou_3d'
    GIOU_3D = 'giou_3d'
    IOU_2D = 'iou_2d'
    GIOU_2D = 'giou_2d'
    MAHALANOBIS_DIST = 'm_dis'
    EULER = 'euler'
    DIST_2D = 'dist_2d'
    DIST_3D = 'dist_3d'
    UNKNOWN = 'unknown'


def dist_ground(bbox1: Box3D, bbox2: Box3D) -> float:
    # Compute distance of bottom center in 3D space, NOT considering the difference in height
    c1 = Box3D.bbox2array(bbox1)[[0, 2]]
    c2 = Box3D.bbox2array(bbox2)[[0, 2]]
    return np.linalg.norm(c1 - c2)


def dist3d(bbox1: Box3D, bbox2: Box3D) -> float:
    # Compute distance of actual center in 3D space, considering the difference in height
    # compute center point based on 8 corners
    corners1 = box2corners3d_camcoord(bbox1)
    corners2 = box2corners3d_camcoord(bbox2)
    c1 = np.average(corners1, axis=0)
    c2 = np.average(corners2, axis=0)
    return np.linalg.norm(c1 - c2)


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
    corrected_yaw_diff = diff_orientation_correction(float(diff[3, 0]))
    diff[3] = corrected_yaw_diff

    if trk_inv_innovation_matrix is not None:
        sqr = np.dot(diff.T, trk_inv_innovation_matrix.dot(diff))[0, 0]
    else:
        sqr = np.dot(diff.T, diff)[0, 0]  # distance along 7 dimension
    return float(math.sqrt(sqr))
