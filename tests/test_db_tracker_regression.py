from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracker.db_tracker import DBTracker


def test_dbtracker_large_shift_keeps_track() -> None:
    tracker = DBTracker(
        max_age=10,
        min_hits=1,
        center_gate_frac=0.25,
        r_pos=0.25,
        r_sz=0.5,
    )

    det0 = ((100, 120, 220, 320), 0.9, "person")
    outputs0 = tracker.update([det0], img_size=(640, 480))
    assert len(outputs0) == 1
    track_id = outputs0[0][0]

    det1 = ((250, 120, 370, 320), 0.9, "person")
    outputs1 = tracker.update([det1], img_size=(640, 480))
    assert len(outputs1) == 1
    assert outputs1[0][0] == track_id
