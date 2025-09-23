"""Utilities for drawing tracking overlays on video frames."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2

BBox = Tuple[int, int, int, int]
TrackTuple = Tuple[int, BBox, str, float]


def draw_tracks(frame, tracks: Iterable[TrackTuple], draw_scores: bool = True):
    """Annotate a frame with bounding boxes and labels.

    Args:
        frame: Frame to draw on. Modified in-place.
        tracks: Iterable of track tuples ``(track_id, bbox, label, score)``.
        draw_scores: Whether to append the confidence score to the label.

    Returns:
        The mutated frame for convenience.
    """

    for tid, (x1, y1, x2, y2), clazz, conf in tracks:
        color = (37 * tid % 255, 17 * tid % 255, 89 * tid % 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"ID{tid}:{clazz}"
        if draw_scores:
            text += f" {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame

