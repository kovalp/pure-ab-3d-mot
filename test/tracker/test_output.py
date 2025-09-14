"""."""

import numpy as np
import pytest

from pure_ab_3d_mot.tracker import Ab3DMot


REF = ((7.0, 6.0, 5.0, 1.0, 2.0, 3.0, 4.0, 123.0, 8.0, 9.0, 10.0, 11.0, 12.0),)


def test_empty_tracker(tracker: Ab3DMot) -> None:
    assert tracker.output() == []
    assert tracker.trackers == []


def test_updated_but_not_persistent(tracker1: Ab3DMot) -> None:
    tracker1.frame_count = 10
    assert tracker1.output() == []
    assert len(tracker1.trackers) == 1


def test_updated_and_first_steps(tracker1: Ab3DMot) -> None:
    result = tracker1.output()
    assert len(result) == 1
    assert result[0] == pytest.approx(np.array(REF))
    assert len(tracker1.trackers) == 1


def test_updated_and_persistent(tracker1: Ab3DMot) -> None:
    tracker1.frame_count = 10
    tracker1.trackers[0].hits = 4
    result = tracker1.output()
    assert len(result) == 1
    assert result[0] == pytest.approx(np.array(REF))
    assert len(tracker1.trackers) == 1


def test_tired_to_wait_remove_target(tracker1: Ab3DMot) -> None:
    tracker1.trackers[0].time_since_update = 5
    assert tracker1.output() == []
    assert len(tracker1.trackers) == 0
