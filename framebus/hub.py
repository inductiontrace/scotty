from __future__ import annotations
import time
import json
import socket
from typing import Dict, Any

import cv2
import imagezmq
from picamera2 import Picamera2


def main() -> None:
    """Publish frames from Picamera2 over ZeroMQ."""
    sender = imagezmq.ImageSender(connect_to="tcp://*:5555", REQ_REP=False)
    cam = Picamera2()
    cam_config = cam.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    cam.configure(cam_config)
    cam.start()
    host = socket.gethostname()
    frame_id = 0
    try:
        while True:
            arr = cam.capture_array()
            ts = time.time()
            ok, jpg = cv2.imencode(".jpg", arr[:, :, ::-1])
            if not ok:
                continue
            meta: Dict[str, Any] = {
                "frame_id": frame_id,
                "ts": ts,
                "camera_id": "pi5_cam0",
                "w": int(arr.shape[1]),
                "h": int(arr.shape[0]),
                "fmt": "jpg",
                "host": host,
            }
            sender.send_jpg(json.dumps(meta), jpg)
            frame_id += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
