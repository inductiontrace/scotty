"""Association helpers for SORT-like trackers."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def class_mismatch_penalty(trk_cls: str, det_cls: str, penalty: float) -> float:
    """Return penalty if classes differ."""

    return penalty if trk_cls != det_cls else 0.0


def within_asym_scale_gate(
    pred_wh: Tuple[float, float],
    det_wh: Tuple[float, float],
    grow_tol: float,
    shrink_tol: float,
) -> bool:
    """Check asymmetric scale tolerances for width/height."""

    pw, ph = pred_wh
    dw, dh = det_wh
    if pw <= 0 or ph <= 0:
        return True
    okw = shrink_tol * pw <= dw <= grow_tol * pw
    okh = shrink_tol * ph <= dh <= grow_tol * ph
    return okw and okh


def normalized_l2(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    norm: Iterable[float] | None,
) -> float:
    """Normalized Euclidean distance between two points."""

    if norm is None:
        nx = ny = 1.0
    else:
        norm_iter = list(norm)
        if len(norm_iter) < 2:
            nx = ny = 1.0
        else:
            nx = float(norm_iter[0]) or 1.0
            ny = float(norm_iter[1]) or 1.0
    dx = (p1[0] - p2[0]) / nx
    dy = (p1[1] - p2[1]) / ny
    return math.hypot(dx, dy)

