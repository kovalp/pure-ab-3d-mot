"""."""

from typing import List, Sequence

from pure_ab_3d_mot.box import Box3D


def process_dets(dets: Sequence[float], ann_ids: Sequence[int]) -> List[Box3D]:
    """Convert to list of Box3D objects.

    Args:
        dets: detections in the KITTI format [[h,w,l,x,y,z,theta],...]
        ann_ids: optional array of annotation ids.

    Returns:
        The list of Box3D objects.
    """
    boxes = []
    for i, det in enumerate(dets):
        box = Box3D.from_kitti(det)
        if len(ann_ids) > 0:
            box.ann_id = int(ann_ids[i])
        boxes.append(box)

    return boxes
