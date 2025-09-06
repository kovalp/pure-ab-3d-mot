"""."""

from pure_ab_3d_mot.tracker import Ab3DMot


def test_first_det(tracker: Ab3DMot) -> None:
    """."""
    tracker.prediction()
    assert len(tracker.trackers) == 0
