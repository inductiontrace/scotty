from __future__ import annotations

import math

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry import core_center, dilate_box, shrink_box
from tracker.costs import class_mismatch_penalty, normalized_l2, within_asym_scale_gate


def test_class_mismatch_penalty() -> None:
    assert class_mismatch_penalty("person", "person", 100.0) == 0.0
    assert class_mismatch_penalty("person", "box", 100.0) == 100.0


def test_shrink_and_core_center() -> None:
    box = (0.0, 0.0, 10.0, 20.0)
    shrunk = shrink_box(*box, 0.2)
    # 20% trimmed from each side → width 6, height 12
    assert shrunk == (1.0, 2.0, 9.0, 18.0)
    cx, cy = core_center(*box, 0.2)
    assert cx == 5.0
    assert cy == 10.0


def test_within_asym_scale_gate() -> None:
    assert within_asym_scale_gate((10.0, 10.0), (14.0, 14.0), 1.6, 0.6)
    assert within_asym_scale_gate((10.0, 10.0), (6.0, 6.0), 1.6, 0.6)
    assert not within_asym_scale_gate((10.0, 10.0), (20.0, 20.0), 1.6, 0.6)
    assert not within_asym_scale_gate((10.0, 10.0), (5.0, 8.0), 1.6, 0.6)


def test_dilate_box() -> None:
    dilated = dilate_box(0.0, 0.0, 10.0, 20.0, 0.1, 0.05)
    assert dilated == (-1.0, -1.0, 11.0, 21.0)


def test_normalized_l2() -> None:
    dist = normalized_l2((5.0, 5.0), (6.0, 9.0), (10.0, 10.0))
    assert math.isclose(dist, math.hypot(0.1, 0.4))
    assert normalized_l2((0.0, 0.0), (0.0, 0.0), None) == 0.0

