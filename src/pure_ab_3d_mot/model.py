# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
# Refactored by <"Peter Koval" koval.peter@gmail.com> 2025

import numpy as np
import copy
from .box import Box3D
from .matching import data_association
from .kalman_filter import KF


def within_range(theta):
    # make sure the orientation is within a proper range

    if theta >= np.pi: theta -= np.pi * 2  # make the theta still in the range
    if theta < -np.pi: theta += np.pi * 2

    return theta


def orientation_correction(self, theta_pre, theta_obs):
    # update orientation in propagated tracks and detected boxes so that they are within 90 degree

    # make the theta still in the range
    theta_pre = within_range(theta_pre)
    theta_obs = within_range(theta_obs)

    # if the angle of two theta is not acute angle, then make it acute
    if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(
            theta_obs - theta_pre) < np.pi * 3 / 2.0:
        theta_pre += np.pi
        theta_pre = within_range(theta_pre)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
        if theta_obs > 0:
            theta_pre += np.pi * 2
        else:
            theta_pre -= np.pi * 2

    return theta_pre, theta_obs


def process_dets(dets):
    # convert each detection into the class Box3D
    # inputs:
    # 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

    dets_new = []
    for det in dets:
        det_tmp = Box3D.array2bbox_raw(det)
        dets_new.append(det_tmp)

    return dets_new


# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):
    """."""

    def __init__(self) -> None:
        self.trackers = []
        self.frame_count = 0
        self.id_now_output = []
        self.ego_com = False  # ego motion compensation
        self.ID_count = [1]
        self.algm = 'hungar'
        self.metric = 'giou_3d'
        self.thres = -0.2
        self.min_hits = 3
        self.max_age = 2
        self.min_sim = -1.0
        self.max_sim = 1.0

    def prediction(self):
        # get predicted locations from existing tracks
        trks = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.predict()
            kf_tmp.kf.x[3] = within_range(kf_tmp.kf.x[3])
            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))
        return trks

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

    def birth(self, dets, info, unmatched_dets):
        # create and initialise new trackers for unmatched detections

        # dets = copy.copy(dets)
        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id)

            self.ID_count[0] += 1

        return new_id_list

    def output(self):
        # output exiting tracks that have been stably associated, i.e., >= min_hits
        # and also delete tracks that have appeared for a long time, i.e., >= max_age

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
            d = Box3D.bbox2array_raw(d)

            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
            num_trks -= 1

            # death, remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(num_trks)

        return results

    def track(self, dets_all):
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
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array
        self.frame_count += 1

        # recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        # process detection format
        dets = process_dets(dets)

        # tracks propagation based on velocity
        trks = self.prediction()

        # matching
        trk_innovation_matrix = None
        if self.metric == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]
        matched, unmatched_dets, unmatched_trks, cost, affi = \
            data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

        self.update(matched, unmatched_trks, dets, info)

        # create and initialise new trackers for unmatched detections
        self.birth(dets, info, unmatched_dets)

        # output existing valid tracks
        results = self.output()
        if len(results) > 0:
            results = [np.concatenate(results)]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 15))]
        self.id_now_output = results[0][:, 7].tolist()  # only the active tracks that are outputed

        return results
