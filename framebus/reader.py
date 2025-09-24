from __future__ import annotations
import json
from typing import Any, Dict, Iterator, Tuple

import cv2
import imagezmq
import numpy as np


class ZmqFrameReader:
    """Iterate over frames from a FrameBus publisher."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555") -> None:
        self.hub = imagezmq.ImageHub(open_port=endpoint, REQ_REP=False)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        while True:
            meta_json, jpg = self.hub.recv_jpg()
            meta = json.loads(meta_json)
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield frame, meta

    def close(self) -> None:
        close_fn = getattr(self.hub, "close", None)
        if callable(close_fn):
            close_fn()
