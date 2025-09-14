"""."""

from typing import List

from pure_ab_3d_mot.box import Box3D


def process_dets(dets) -> List[Box3D]:
    """Convert to list of Box3D objects.

    Args:
        dets: detections in the KITTI format [[h,w,l,x,y,z,theta],...]

    Returns:
        The list.
    """
    dets_new = []
    for det in dets:
        det_tmp = Box3D.from_kitti(det)
        dets_new.append(det_tmp)

    return dets_new
