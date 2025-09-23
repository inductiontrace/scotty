"""Shared video source utilities."""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple

import cv2


FrameReadFn = Callable[[], Tuple[bool, Any]]
FrameCloseFn = Callable[[], None]
FrameSizeFn = Callable[[], Tuple[int, int]]


def to_bgr(frame):
    """Normalize frames from any source to a 3-channel BGR array."""
    if frame is None:
        return None
    if frame.ndim == 2:
        # GRAY -> BGR
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame ndim={frame.ndim}, expected 2 or 3.")
    height, width, channels = frame.shape
    if channels == 3:
        # Heuristic: Picamera2 default is RGB; convert to BGR for consistency.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if channels == 4:
        # Many Pi pipelines produce XRGB/BGRA; try BGRA->BGR first.
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unexpected channel count: {channels}, expected 1/3/4.")


def open_source(src, fps):
    """Return read/close/size helpers for a camera source."""

    if isinstance(src, str) and src.lower() == "picam":
        try:
            from picamera2 import Picamera2
        except Exception as exc:  # pragma: no cover - hardware dependency
            raise SystemExit(
                "Picamera2 not available. Install with: sudo apt install -y python3-picamera2"
            ) from exc

        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        picam.configure(config)
        picam.start()
        time.sleep(0.3)
        first = picam.capture_array()
        height, width = first.shape[:2]

        def read_fn():
            frame = picam.capture_array()
            return True, frame

        def close_fn():
            try:
                picam.stop()
            except Exception:
                pass

        def size_fn():
            return (width, height)

        return read_fn, close_fn, size_fn

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"Failed to read first frame from source: {src}")
    height, width = frame.shape[:2]

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception:
        pass

    def read_fn():
        ok, frame_local = cap.read()
        return ok, frame_local

    def close_fn():
        cap.release()

    def size_fn():
        return (width, height)

    return read_fn, close_fn, size_fn
