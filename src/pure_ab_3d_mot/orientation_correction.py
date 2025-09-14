"""."""

from typing import Tuple

import numpy as np


def within_range(theta: float) -> float:
    # make sure the orientation is within a proper range

    if theta >= np.pi:
        theta -= np.pi * 2  # make the theta still in the range
    if theta < -np.pi:
        theta += np.pi * 2

    return theta


def orientation_correction(theta_pre: float, theta_obs: float) -> Tuple[float, float]:
    # update orientation in propagated tracks and detected boxes so that they are within 90 degree

    # make the theta still in the range
    theta_pre = within_range(theta_pre)
    theta_obs = within_range(theta_obs)

    # if the angle of two theta is not acute angle, then make it acute
    if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
        theta_pre += np.pi
        theta_pre = within_range(theta_pre)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
        if theta_obs > 0:
            theta_pre += np.pi * 2
        else:
            theta_pre -= np.pi * 2

    return theta_pre, theta_obs
