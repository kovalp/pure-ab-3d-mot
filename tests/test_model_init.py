"""."""

from pure_ab_3d_mot.model import AB3DMOT


def test_ab_3d_mot_init() -> None:
    """."""
    tracker = AB3DMOT()
    assert tracker.trackers == []
