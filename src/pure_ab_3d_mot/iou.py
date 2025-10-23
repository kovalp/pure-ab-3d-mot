"""."""

from typing import List, Optional, Tuple, Union

import numpy as np

from scipy.spatial import ConvexHull

from pure_ab_3d_mot.box import Box3D, box2corners3d_camcoord
from pure_ab_3d_mot.dist_metrics import MetricKind


Poly2DT = Union[List[Tuple[float, float]], np.ndarray]
TOL_INSIDE = 1e-14


def iou(box_a: Box3D, box_b: Box3D, metric: MetricKind = MetricKind.GIOU_3D) -> Union[float, None]:
    """Compute 3D/2D bounding box IoU, only working for object parallel to ground

    Input:
        Box3D instances
    Output:
        iou_3d: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    box corner order is like follows
            1 -------- 0 		 top is bottom because y direction is negative
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    rect/ref camera coord:
    right x, down y, front z
    """
    # compute 2D related measures
    corners_a = compute_bottom1(box_a)
    corners_b = compute_bottom1(box_b)
    i_2d = compute_inter_2d(corners_a, corners_b)

    if metric == MetricKind.IOU_2D:
        u_2d = box_a.w * box_a.l + box_b.w * box_b.l - i_2d
        return i_2d / u_2d
    elif metric == MetricKind.GIOU_2D:
        u_2d = box_a.w * box_a.l + box_b.w * box_b.l - i_2d
        c_2d = convex_area(corners_a, corners_b)
        return i_2d / u_2d - (c_2d - u_2d) / c_2d
    elif metric == MetricKind.IOU_3D:
        overlap_height = compute_height(box_a, box_b)
        i_3d = i_2d * overlap_height
        u_3d = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - i_3d
        return i_3d / u_3d
    else:
        if metric != MetricKind.GIOU_3D:
            raise ValueError(f'Metric: {metric} is not supported.')
        overlap_height = compute_height(box_a, box_b)
        i_3d = i_2d * overlap_height
        u_3d = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - i_3d
        union_height = compute_height(box_a, box_b, inter=False)
        c_2d = convex_area(corners_a, corners_b)
        c_3d = c_2d * union_height
        return i_3d / u_3d - (c_3d - u_3d) / c_3d


def compute_bottom1(box: Box3D) -> np.ndarray:
    # obtain ground corners and area, not containing the height
    corners = box2corners3d_camcoord(box)  # 8 x 3

    # get bottom corners and inverse order so that they are in the
    # counter-clockwise order to fulfill polygon_clip
    return corners[-5::-1, [0, 2]]  # 4 x 2


def compute_inter_2d(boxa_bottom, boxb_bottom):
    # computer intersection over union of two sets of bottom corner points

    _, I_2D = convex_hull_intersection(boxa_bottom, boxb_bottom)

    # a slower version
    # from shapely.geometry import Polygon
    # reca, recb = Polygon(boxa_bottom), Polygon(boxb_bottom)
    # I_2D = reca.intersection(recb).area

    return I_2D


def polygon_clip(subject_poly: Poly2DT, clip_poly: Poly2DT) -> Optional[Poly2DT]:
    """Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subject_poly: a list of (x,y) 2d points, any polygon.
        clip_poly: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p: Tuple[float, float]) -> bool:
        a = (cp2[0] - cp1[0]) * (p[1] - cp1[1])
        b = (cp2[1] - cp1[1]) * (p[0] - cp1[0])
        return a + TOL_INSIDE > b

    def compute_intersection() -> Tuple[float, float]:
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denominator = dc[0] * dp[1] - dc[1] * dp[0]
        n3 = 1.0 / denominator
        return (n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3

    output_list = subject_poly
    cp1 = clip_poly[-1]

    for cp2 in clip_poly:
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for subjectVertex in input_list:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    output_list.append(compute_intersection())
                output_list.append(e)
            elif inside(s):
                output_list.append(compute_intersection())
            s = e
        cp1 = cp2
        if len(output_list) == 0:
            return None
    return np.array(output_list)


def convex_hull_intersection(p1: Poly2DT, p2: Poly2DT) -> Tuple[Optional[Poly2DT], float]:
    """Compute area of two convex hull's intersection area.

    Args:
        p1: list of (x,y) tuples of hull vertices.
        p2: list of (x,y) tuples of hull vertices.

    Returns:
        a list of (x,y) for the intersection and its area
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def compute_height(box_a, box_b, inter=True) -> float:
    """."""
    corners1 = box2corners3d_camcoord(box_a)  # 8 x 3
    corners2 = box2corners3d_camcoord(box_b)  # 8 x 3

    if inter:  # compute overlap height
        y_max = min(corners1[0, 1], corners2[0, 1])
        y_min = max(corners1[4, 1], corners2[4, 1])
        height = max(0.0, y_max - y_min)
    else:  # compute union height
        y_max = max(corners1[0, 1], corners2[0, 1])
        y_min = min(corners1[4, 1], corners2[4, 1])
        height = max(0.0, y_max - y_min)

    return height


def convex_area(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    # compute the convex area
    all_corners = np.vstack((corners_a, corners_b))
    c_hull = ConvexHull(all_corners)
    convex_corners = all_corners[c_hull.vertices]
    return PolyArea2D(convex_corners)


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area
