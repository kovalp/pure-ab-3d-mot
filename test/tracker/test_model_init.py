"""."""

from pure_ab_3d_mot.tracker import Ab3DMot


def test_ab_3d_mot_init(tracker: Ab3DMot) -> None:
    """."""
    assert tracker.trackers == []
