# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
# Refactored by <"Peter Koval" koval.peter@gmail.com> 2025


from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from association_quality_clavia import UPD_ID_LOOSE

from .box import Box3D
from .dist_metrics import MetricKind
from .matching import MatchingAlgorithm, data_association
from .orientation_correction import orientation_correction, within_range
from .process_dets import process_dets
from .str_const import ANN_IDS, DETS, INFO
from .target import Target


class Ab3DMot(object):  # A Baseline of 3D Multi-Object Tracking
    """."""

    def __init__(self) -> None:
        """."""
        self.trackers: List[Target] = []
        self.frame_count = 0
        self.id_now_output = []
        self.ego_com = False  # ego motion compensation
        self.ID_count = [1]
        self.algorithm: MatchingAlgorithm = MatchingAlgorithm.HUNGARIAN
        self.metric = MetricKind.GIOU_3D
        self.threshold = -0.2
        self.min_hits = 3
        self.max_age = 2
        self.min_sim = -1.0
        self.max_sim = 1.0

    def update(
        self,
        matched: Union[np.ndarray, Sequence[Tuple[int, int]]],
        unmatched_tracks: Union[np.ndarray, Sequence[int]],
        det_boxes: List[Box3D],
        info: Union[np.ndarray, Sequence[List[float]]],
    ) -> None:
        """Update matched trackers with assigned detections

        Args:
            matched: a list of associated detection-track pairs.
            unmatched_tracks: a list of unmatched tracks.
            det_boxes: the list of detections.
            info: the array of other info for each detection.
        """
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_tracks:
                det_idx = matched[matched[:, 1] == t, 0]  # a list of detection indices
                assert det_idx.size == 1
                det_box = det_boxes[det_idx[0]]

                # update statistics
                trk.time_since_update = 0  # reset because just updated
                trk.hits += 1
                trk.upd_id = det_box.ann_id

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                pose = Box3D.bbox2array(det_box)[:7]
                trk.kf.x[3], pose[3] = orientation_correction(trk.kf.x[3], pose[3])

                # kalman filter update with observation
                trk.kf.update(pose)
                trk.kf.x[3] = within_range(trk.kf.x[3])
                trk.info[:] = info[det_idx[0], :]
            else:
                trk.upd_id = UPD_ID_LOOSE

    def birth(
        self, det_boxes: List[Box3D], info: np.ndarray, unmatched_detections: Sequence[int]
    ) -> List[int]:
        # create and initialise new trackers for unmatched detections

        new_id_list = []  # new ID will be generated for unmatched detections
        for i in unmatched_detections:
            box = det_boxes[i]
            pose = Box3D.bbox2array(box)[:7]
            trk = Target(pose, info[i, :], self.ID_count[0], ann_id=box.ann_id)
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            self.ID_count[0] += 1

        return new_id_list

    def output(self) -> List[np.ndarray]:
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age

        track_num = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            my_box = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
            kitti_det = Box3D.bbox2array_kitti(my_box)

            if trk.time_since_update < self.max_age and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                results.append(np.concatenate((kitti_det, [trk.id], trk.info)).reshape(1, -1))

            track_num -= 1
            if trk.time_since_update >= self.max_age:
                self.trackers.pop(track_num)  # death, remove dead tracklet

        return results

    def prediction(self) -> None:
        # get predicted locations from existing tracks
        for track in self.trackers:
            track.kf.predict()  # propagate locations
            track.kf.x[3] = within_range(track.kf.x[3])  # correct the yaw angle
            track.time_since_update += 1  # update statistics

    def track(self, dets_all: Dict[str, Union[List[List[float]], np.ndarray]]) -> np.ndarray:
        """
        Args:
            dets_all: dictionary with keys
                'dets' - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
                'info' - an array of other info for each det
                'ann_ids' - optional array of annotation ids.
            frame:    str, frame number, used to query ego pose
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        self.prediction()  # tracks (targets) propagation with constant-velocity Kalman filter.

        # matching
        trk_innovation_mat = []
        if self.metric == MetricKind.MAHALANOBIS_DIST:
            trk_innovation_mat = [trk.compute_innovation_matrix() for trk in self.trackers]
        det_boxes = process_dets(dets_all[DETS], dets_all.get(ANN_IDS, []))
        matched, unmatched_dets, unmatched_trks, cost, affi = data_association(
            det_boxes,
            self.get_target_boxes(),
            self.metric,
            self.threshold,
            self.algorithm,
            trk_innovation_mat,
        )

        info = dets_all[INFO]
        self.update(matched, unmatched_trks, det_boxes, info)
        self.birth(det_boxes, info, unmatched_dets)  # create and initialise new trackers

        results = self.output()  # output existing valid tracks
        if len(results) > 0:
            results = [np.concatenate(results)]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 15))]
        self.id_now_output = results[0][:, 7].tolist()  # only the active tracks that are output
        return results

    def get_target_boxes(self) -> List[Box3D]:
        """."""
        return [Box3D.array2bbox(trk.kf.x[:7, 0]) for trk in self.trackers]
