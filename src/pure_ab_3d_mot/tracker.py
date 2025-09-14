# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
# Refactored by <"Peter Koval" koval.peter@gmail.com> 2025

import copy

from typing import Dict, List, Union

import numpy as np

from .box import Box3D
from .dist_metrics import MetricKind
from .matching import MatchingAlgorithm, data_association
from .orientation_correction import orientation_correction, within_range
from .process_dets import process_dets
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

    def update(self, matched, unmatched_trks, dets, info):
        # update matched trackers with assigned detections
        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # a list of index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0  # reset because just updated
                trk.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = orientation_correction(trk.kf.x[3], bbox3d[3])

                # kalman filter update with observation
                trk.kf.update(bbox3d)
                trk.kf.x[3] = within_range(trk.kf.x[3])
                trk.info = info[d, :][0]

    def birth(self, dets, info, unmatched_dets: np.ndarray) -> List[int]:
        # create and initialise new trackers for unmatched detections

        # dets = copy.copy(dets)
        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = Target(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id)

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
            kitti_det = Box3D.bbox2array_raw(my_box)

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
        Params:
              dets_all: dict
                dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
                info: a array of other info for each det
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
        det_boxes = process_dets(dets_all['dets'])  # process detection format
        matched, unmatched_dets, unmatched_trks, cost, affi = data_association(
            det_boxes,
            self.get_target_boxes(),
            self.metric,
            self.threshold,
            self.algorithm,
            trk_innovation_mat,
        )

        info = dets_all['info']
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
