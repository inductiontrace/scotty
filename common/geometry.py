"""Geometry helpers for tracker association tweaks."""

from __future__ import annotations

from typing import Tuple

BBox = Tuple[float, float, float, float]


def box_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Return the center point of an ``(x1, y1, x2, y2)`` box."""

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def shrink_box(
    x1: float, y1: float, x2: float, y2: float, shrink: float
) -> Tuple[float, float, float, float]:
    """Shrink a box by ``shrink`` fraction on each side."""

    w = x2 - x1
    h = y2 - y1
    nx1 = x1 + 0.5 * w * shrink
    ny1 = y1 + 0.5 * h * shrink
    nx2 = x2 - 0.5 * w * shrink
    ny2 = y2 - 0.5 * h * shrink
    return nx1, ny1, nx2, ny2


def core_center(x1: float, y1: float, x2: float, y2: float, shrink: float) -> Tuple[float, float]:
    """Return the center of the shrunk ``core`` box."""

    sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, shrink)
    return box_center(sx1, sy1, sx2, sy2)


def dilate_box(
    x1: float, y1: float, x2: float, y2: float, dx: float, dy: float
) -> Tuple[float, float, float, float]:
    """Dilate a box by ``dx``/``dy`` fractions per axis for association only."""

    w = x2 - x1
    h = y2 - y1
    return x1 - w * dx, y1 - h * dy, x2 + w * dx, y2 + h * dy

