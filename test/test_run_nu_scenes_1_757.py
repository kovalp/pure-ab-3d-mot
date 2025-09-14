"""."""

from pathlib import Path

import numpy as np

from pure_ab_3d_mot.tracker import Ab3DMot


def test_run_1_757(files_dir: Path) -> None:
    """."""
    f_path = files_dir / 'annotation_task_1_757.csv'
    ann = np.genfromtxt(f_path, 'float32', '#', ',', 1, usecols=[2, 3, 4, 5, 6, 7, 8])
    t_id = np.genfromtxt(f_path, int, '#', ',', 1, usecols=[0, 1])
    time_stamps = np.unique(t_id[:, 0])

    to_kitti = 5, 4, 3, 0, 1, 2, 6
    tracker = Ab3DMot()
    for ts_num, time_stamp in enumerate(time_stamps):
        time_stamp_mask = np.where(t_id[:, 0] == time_stamp)
        ids_r = t_id[time_stamp_mask, 1]
        det_r = ann[time_stamp_mask][:, to_kitti]
        det_dct = {'dets': det_r, 'info': ids_r.T}
        tracker.track(det_dct)

    assert len(tracker.trackers) == 13
