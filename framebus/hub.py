from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any, Dict, Tuple, Union

import cv2
import imagezmq

from common.video_utils import open_source


def _normalize_source_arg(source: str) -> Union[str, int]:
    """Return the value to pass into ``open_source``."""

    if source.lower() in {"picam", "picamera", "pi"}:
        return "picam"
    try:
        return int(source)
    except ValueError:
        return source


def _maybe_rgb_to_bgr(frame, assume_rgb: bool) -> Any:
    if frame is None:
        return None
    if assume_rgb and frame.ndim == 3 and frame.shape[2] == 3:
        return frame[:, :, ::-1]
    return frame


def main() -> None:
    """Publish frames from Picamera2 or OpenCV sources over ZeroMQ."""

    parser = argparse.ArgumentParser(description="FrameBus hub publisher")
    parser.add_argument(
        "--endpoint",
        default="tcp://*:5555",
        help="ZeroMQ endpoint to bind (default: tcp://*:5555)",
    )
    parser.add_argument(
        "--source",
        default="picam",
        help="Camera device index, video file path, or 'picam' for Picamera2",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target publish rate when throttling non-ZMQ sources",
    )
    parser.add_argument(
        "--camera-id",
        default="pi5_cam0",
        help="Camera identifier to include in frame metadata",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop a video file instead of exiting at EOF",
    )
    args = parser.parse_args()

    src = _normalize_source_arg(str(args.source))
    assume_rgb = src == "picam"

    try:
        read_fn, close_fn, size_fn = open_source(src, args.fps)
    except SystemExit as exc:
        raise SystemExit(str(exc))

    width, height = size_fn()

    sender = imagezmq.ImageSender(connect_to=args.endpoint, REQ_REP=False)
    host = socket.gethostname()
    frame_id = 0
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    next_ts = time.time()

    try:
        while True:
            if period > 0:
                now = time.time()
                if now < next_ts:
                    time.sleep(min(0.01, next_ts - now))
                    continue
                next_ts = now + period

            ok, frame = read_fn()
            if not ok:
                if args.loop:
                    close_fn()
                    read_fn, close_fn, size_fn = open_source(src, args.fps)
                    width, height = size_fn()
                    continue
                break

            frame_bgr = _maybe_rgb_to_bgr(frame, assume_rgb)
            if frame_bgr is None:
                continue

            ok, jpg = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                continue

            ts = time.time()
            meta: Dict[str, Any] = {
                "frame_id": frame_id,
                "ts": ts,
                "camera_id": args.camera_id,
                "w": int(width),
                "h": int(height),
                "fmt": "jpg",
                "host": host,
            }
            sender.send_jpg(json.dumps(meta), jpg)
            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        close_fn()


if __name__ == "__main__":
    main()
