"""."""

import numpy as np

from filterpy.kalman import KalmanFilter


class Target:
    """."""

    def __init__(self, pose: np.ndarray, info: np.ndarray, track_id: int):
        self.initial_pose = pose
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # other information associated
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # There is no need to use EKF here as the measurement and state are in the same space with linear relationship

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix, dim_x * dim_x
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000.0
        self.kf.P *= 10.0

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        # initialize data
        self.kf.x[:7] = self.initial_pose.reshape((7, 1))

    def __repr__(self) -> str:
        """."""
        return f'Target(id {self.id} state {self.kf.x.reshape(-1)} info {self.info})'

    def compute_innovation_matrix(self) -> np.ndarray:
        """compute the innovation matrix for association with mahalanobis distance"""
        return self.kf.H.dot(self.kf.P.dot(self.kf.H.T)) + self.kf.R

    def get_velocity(self) -> np.ndarray:
        # return the object velocity in the state

        return self.kf.x[7:]
